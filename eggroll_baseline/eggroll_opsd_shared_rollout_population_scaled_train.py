#!/usr/bin/env python3
"""Vanilla EGGROLL with boxed-correctness fitness on the OPSD OpenThoughts split.

Each step keeps the same 32 OpenThoughts prompts as OPSD EGGROLL.  Members are
tiled like HyperscaleES: generations_per_prompt=8, so each mirrored perturbation
rolls out exactly one prompt (4 antithetic pairs share a prompt).  Fitness is 1
if the boxed answer matches the ground-truth boxed answer from `solution`.
EGGROLL centers within those 8 members and divides by the population-wide
raw-score standard deviation.  The update uses N=256 with one measurement per
member, so pair coefficients are not averaged over 32 prompts.

Generation uses Qwen non-thinking mode (``enable_thinking=False``) with a
4096-token cap, and the same tensorized layout as Direct EGGROLL-OPSD: a wave of
``tensorized_members_per_wave_per_gpu`` members times
``tensorized_prompt_microbatch_size`` prompts, left-padded to the longest
prompt in the wave, with per-row low-rank noise from
``tensorized_member_perturbations``.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from eggroll_opsd_train_population_scaled import (
    append_jsonl,
    apply_eggroll_update,
    boxed_correctness_reward,
    generate_student_from_ids,
    ground_truth_from_solution,
    replace_target_linears,
    save_center_checkpoint,
    student_prompt,
    tensorized_member_perturbations,
    unwrap_target_linears,
)


def tokenize_prompt(tokenizer, text: str, max_prompt_length: int) -> torch.Tensor:
    return tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_length,
    ).input_ids[0].cpu()


def gather_center_completions(
    local_completions: list[torch.Tensor],
    *,
    max_completion_length: int,
    pad_token_id: int,
    device: torch.device,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """All-gather rank-local center rollouts in rank-major prompt order."""
    local_count = len(local_completions)
    lengths = torch.tensor(
        [completion.numel() for completion in local_completions],
        dtype=torch.long,
        device=device,
    )
    tokens = torch.full(
        (local_count, max_completion_length),
        fill_value=pad_token_id,
        dtype=torch.long,
        device=device,
    )
    for index, completion in enumerate(local_completions):
        length = int(completion.numel())
        if not 0 < length <= max_completion_length:
            raise RuntimeError(f"Invalid center completion length {length}")
        tokens[index, :length] = completion.to(device)

    gathered_lengths = [torch.zeros_like(lengths) for _ in range(dist.get_world_size())]
    gathered_tokens = [torch.zeros_like(tokens) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_lengths, lengths)
    dist.all_gather(gathered_tokens, tokens)

    all_lengths = torch.cat(gathered_lengths).cpu()
    all_tokens = torch.cat(gathered_tokens).cpu()
    completions = [
        all_tokens[index, : int(all_lengths[index].item())].clone()
        for index in range(all_tokens.shape[0])
    ]
    return completions, all_lengths


def pad_sequences(
    sequences: list[torch.Tensor],
    *,
    pad_token_id: int,
    device: torch.device,
    padding_side: str = "right",
) -> tuple[torch.Tensor, torch.Tensor]:
    if padding_side not in ("left", "right"):
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side!r}")
    lengths = [int(sequence.numel()) for sequence in sequences]
    max_length = max(lengths)
    input_ids = torch.full(
        (len(sequences), max_length),
        fill_value=pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (len(sequences), max_length), dtype=torch.long, device=device
    )
    for row, (sequence, length) in enumerate(zip(sequences, lengths)):
        tokens = sequence.to(device)
        if padding_side == "right":
            input_ids[row, :length] = tokens
            attention_mask[row, :length] = 1
        else:
            input_ids[row, max_length - length :] = tokens
            attention_mask[row, max_length - length :] = 1
    return input_ids, attention_mask


def divergence_from_logits(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float,
    token_clip: float,
    chunk_tokens: int = 16,
) -> tuple[float, float]:
    """Return (OPSD clipped loss, ordinary unclipped forward KL) per token."""
    if student_logits.shape != teacher_logits.shape:
        raise RuntimeError(
            f"Student/teacher completion logits differ: "
            f"{student_logits.shape} vs {teacher_logits.shape}"
        )
    token_count = int(student_logits.shape[0])
    clipped_total = torch.zeros((), device=student_logits.device, dtype=torch.float32)
    true_total = torch.zeros((), device=student_logits.device, dtype=torch.float32)
    for start in range(0, token_count, chunk_tokens):
        stop = min(start + chunk_tokens, token_count)
        student_log_probs = F.log_softmax(
            student_logits[start:stop] / temperature, dim=-1
        )
        teacher_log_probs = F.log_softmax(
            teacher_logits[start:stop] / temperature, dim=-1
        )
        pointwise = F.kl_div(
            student_log_probs,
            teacher_log_probs,
            reduction="none",
            log_target=True,
        )
        true_total += pointwise.float().sum()
        clipped_total += pointwise.clamp(max=token_clip).float().sum()
        del student_log_probs, teacher_log_probs, pointwise
    denominator = max(token_count, 1)
    return (
        float((clipped_total / denominator).item()),
        float((true_total / denominator).item()),
    )


def divergence_from_cpu_teacher_cache(
    student_logits: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    *,
    temperature: float,
    token_clip: float,
    chunk_tokens: int = 16,
) -> tuple[float, float]:
    """Compare GPU student logits with a cached CPU teacher distribution."""
    if student_logits.shape != teacher_log_probs.shape:
        raise RuntimeError(
            f"Student logits/teacher targets differ: "
            f"{student_logits.shape} vs {teacher_log_probs.shape}"
        )
    token_count = int(student_logits.shape[0])
    clipped_total = torch.zeros((), device=student_logits.device, dtype=torch.float32)
    true_total = torch.zeros((), device=student_logits.device, dtype=torch.float32)
    for start in range(0, token_count, chunk_tokens):
        stop = min(start + chunk_tokens, token_count)
        student_log_probs = F.log_softmax(
            student_logits[start:stop] / temperature, dim=-1
        )
        teacher_chunk = teacher_log_probs[start:stop].to(
            student_logits.device, non_blocking=False
        )
        pointwise = F.kl_div(
            student_log_probs,
            teacher_chunk,
            reduction="none",
            log_target=True,
        )
        true_total += pointwise.float().sum()
        clipped_total += pointwise.clamp(max=token_clip).float().sum()
        del student_log_probs, teacher_chunk, pointwise
    denominator = max(token_count, 1)
    return (
        float((clipped_total / denominator).item()),
        float((true_total / denominator).item()),
    )


@torch.inference_mode()
def cache_privileged_teacher_distributions(
    teacher,
    *,
    teacher_prompt_ids: list[torch.Tensor],
    completions: list[torch.Tensor],
    pad_token_id: int,
    temperature: float,
    microbatch_size: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """Compute each teacher distribution once and retain it in CPU memory."""
    if len(teacher_prompt_ids) != len(completions):
        raise ValueError("Teacher prompt/completion batch lengths differ")
    caches: list[torch.Tensor] = []
    for start in range(0, len(completions), microbatch_size):
        stop = min(start + microbatch_size, len(completions))
        batch_prompts = teacher_prompt_ids[start:stop]
        batch_completions = completions[start:stop]
        sequences = [
            torch.cat((prompt, completion))
            for prompt, completion in zip(batch_prompts, batch_completions)
        ]
        input_ids, attention_mask = pad_sequences(
            sequences, pad_token_id=pad_token_id, device=device
        )
        outputs = teacher(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        for row, completion in enumerate(batch_completions):
            completion_length = int(completion.numel())
            completion_start = int(batch_prompts[row].numel()) - 1
            logits = outputs.logits[
                row, completion_start : completion_start + completion_length
            ]
            cache = torch.empty(
                logits.shape, dtype=logits.dtype, device="cpu"
            )
            for token_start in range(0, completion_length, 16):
                token_stop = min(token_start + 16, completion_length)
                cache[token_start:token_stop].copy_(
                    F.log_softmax(
                        logits[token_start:token_stop] / temperature, dim=-1
                    ).cpu()
                )
            caches.append(cache)
        del outputs, input_ids, attention_mask, sequences
    return caches


@torch.inference_mode()
def batched_policy_divergences_from_cache(
    policy,
    *,
    student_prompt_ids: list[torch.Tensor],
    completions: list[torch.Tensor],
    teacher_log_probs: list[torch.Tensor],
    pad_token_id: int,
    temperature: float,
    token_clip: float,
    microbatch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score a policy against teacher caches without another teacher forward."""
    if not (
        len(student_prompt_ids) == len(completions) == len(teacher_log_probs)
    ):
        raise ValueError("Student prompt/completion/cache batch lengths differ")
    clipped_losses: list[float] = []
    true_kls: list[float] = []
    for start in range(0, len(completions), microbatch_size):
        stop = min(start + microbatch_size, len(completions))
        batch_prompts = student_prompt_ids[start:stop]
        batch_completions = completions[start:stop]
        sequences = [
            torch.cat((prompt, completion))
            for prompt, completion in zip(batch_prompts, batch_completions)
        ]
        input_ids, attention_mask = pad_sequences(
            sequences, pad_token_id=pad_token_id, device=device
        )
        outputs = policy(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        for row, completion in enumerate(batch_completions):
            completion_length = int(completion.numel())
            completion_start = int(batch_prompts[row].numel()) - 1
            student_logits = outputs.logits[
                row, completion_start : completion_start + completion_length
            ]
            clipped, true_kl = divergence_from_cpu_teacher_cache(
                student_logits,
                teacher_log_probs[start + row],
                temperature=temperature,
                token_clip=token_clip,
            )
            clipped_losses.append(clipped)
            true_kls.append(true_kl)
        del outputs, input_ids, attention_mask, sequences
    return torch.tensor(clipped_losses), torch.tensor(true_kls)


def tensorized_divergence_from_cpu_teacher_cache(
    student_logits: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    *,
    temperature: float,
    token_clip: float,
    chunk_tokens: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized losses for [members, tokens, vocab] student logits."""
    if student_logits.ndim != 3:
        raise RuntimeError(f"Expected member-batched logits, got {student_logits.shape}")
    if student_logits.shape[1:] != teacher_log_probs.shape:
        raise RuntimeError(
            f"Student logits/teacher target mismatch: "
            f"{student_logits.shape} vs {teacher_log_probs.shape}"
        )
    members, token_count, _ = student_logits.shape
    clipped_total = torch.zeros(members, device=student_logits.device, dtype=torch.float32)
    true_total = torch.zeros(members, device=student_logits.device, dtype=torch.float32)
    for start in range(0, token_count, chunk_tokens):
        stop = min(start + chunk_tokens, token_count)
        student_log_probs = F.log_softmax(
            student_logits[:, start:stop] / temperature, dim=-1
        )
        teacher_chunk = teacher_log_probs[start:stop].to(student_logits.device)
        pointwise = F.kl_div(
            student_log_probs,
            teacher_chunk.unsqueeze(0),
            reduction="none",
            log_target=True,
        )
        true_total += pointwise.float().sum(dim=(1, 2))
        clipped_total += pointwise.clamp(max=token_clip).float().sum(dim=(1, 2))
        del student_log_probs, teacher_chunk, pointwise
    denominator = max(token_count, 1)
    return clipped_total / denominator, true_total / denominator


@torch.inference_mode()
def tensorized_policy_divergences_from_cache(
    policy,
    *,
    student_prompt_ids: list[torch.Tensor],
    completions: list[torch.Tensor],
    teacher_log_probs: list[torch.Tensor],
    member_count: int,
    pad_token_id: int,
    temperature: float,
    token_clip: float,
    prompt_microbatch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score several perturbation members in each policy forward."""
    if not (
        len(student_prompt_ids) == len(completions) == len(teacher_log_probs)
    ):
        raise ValueError("Student prompt/completion/cache batch lengths differ")
    if len(completions) % prompt_microbatch_size:
        raise ValueError("Prompt count must divide tensorized prompt microbatch size")
    clipped_by_prompt = []
    true_by_prompt = []
    for start in range(0, len(completions), prompt_microbatch_size):
        stop = start + prompt_microbatch_size
        batch_prompts = student_prompt_ids[start:stop]
        batch_completions = completions[start:stop]
        base_sequences = [
            torch.cat((prompt, completion))
            for prompt, completion in zip(batch_prompts, batch_completions)
        ]
        # Prompt-major layout matches member_indices = [0..M-1] repeated B times.
        sequences = [
            sequence
            for sequence in base_sequences
            for _ in range(member_count)
        ]
        input_ids, attention_mask = pad_sequences(
            sequences, pad_token_id=pad_token_id, device=device
        )
        outputs = policy(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        for prompt_row, completion in enumerate(batch_completions):
            completion_length = int(completion.numel())
            completion_start = int(batch_prompts[prompt_row].numel()) - 1
            row_start = prompt_row * member_count
            member_logits = outputs.logits[
                row_start : row_start + member_count,
                completion_start : completion_start + completion_length,
            ]
            clipped, true_kl = tensorized_divergence_from_cpu_teacher_cache(
                member_logits,
                teacher_log_probs[start + prompt_row],
                temperature=temperature,
                token_clip=token_clip,
            )
            clipped_by_prompt.append(clipped.cpu())
            true_by_prompt.append(true_kl.cpu())
        del outputs, input_ids, attention_mask, sequences, base_sequences
    # [prompts, members] -> [members, prompts]
    return torch.stack(clipped_by_prompt).T, torch.stack(true_by_prompt).T


@torch.inference_mode()
def batched_privileged_divergences(
    policy,
    teacher,
    *,
    student_prompt_ids: list[torch.Tensor],
    teacher_prompt_ids: list[torch.Tensor],
    completions: list[torch.Tensor],
    pad_token_id: int,
    temperature: float,
    token_clip: float,
    microbatch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score fixed completions under policy and privileged teacher."""
    if not (
        len(student_prompt_ids) == len(teacher_prompt_ids) == len(completions)
    ):
        raise ValueError("Prompt/completion batch lengths differ")

    clipped_losses: list[float] = []
    true_kls: list[float] = []
    for start in range(0, len(completions), microbatch_size):
        stop = min(start + microbatch_size, len(completions))
        batch_completions = completions[start:stop]
        batch_student_prompts = student_prompt_ids[start:stop]
        batch_teacher_prompts = teacher_prompt_ids[start:stop]
        student_sequences = [
            torch.cat((prompt, completion.cpu()))
            for prompt, completion in zip(batch_student_prompts, batch_completions)
        ]
        teacher_sequences = [
            torch.cat((prompt, completion.cpu()))
            for prompt, completion in zip(batch_teacher_prompts, batch_completions)
        ]

        teacher_ids, teacher_mask = pad_sequences(
            teacher_sequences, pad_token_id=pad_token_id, device=device
        )
        teacher_outputs = teacher(
            input_ids=teacher_ids,
            attention_mask=teacher_mask,
            use_cache=False,
        )
        teacher_logits = teacher_outputs.logits
        del teacher_outputs, teacher_ids, teacher_mask

        student_ids, student_mask = pad_sequences(
            student_sequences, pad_token_id=pad_token_id, device=device
        )
        student_outputs = policy(
            input_ids=student_ids,
            attention_mask=student_mask,
            use_cache=False,
        )
        student_logits = student_outputs.logits
        del student_outputs, student_ids, student_mask

        for row, completion in enumerate(batch_completions):
            completion_length = int(completion.numel())
            student_start = int(batch_student_prompts[row].numel()) - 1
            teacher_start = int(batch_teacher_prompts[row].numel()) - 1
            clipped, true_kl = divergence_from_logits(
                student_logits[
                    row, student_start : student_start + completion_length
                ],
                teacher_logits[
                    row, teacher_start : teacher_start + completion_length
                ],
                temperature=temperature,
                token_clip=token_clip,
            )
            clipped_losses.append(clipped)
            true_kls.append(true_kl)
        del student_logits, teacher_logits, student_sequences, teacher_sequences

    return (
        torch.tensor(clipped_losses, dtype=torch.float32),
        torch.tensor(true_kls, dtype=torch.float32),
    )


def center_fingerprint(masters: list[torch.Tensor], device: torch.device) -> float:
    # A cheap deterministic rank-synchronization invariant covering every target.
    value = sum(float(master.reshape(-1)[0].item()) for master in masters)
    local = torch.tensor([value], dtype=torch.float64, device=device)
    gathered = [torch.zeros_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    values = torch.cat(gathered)
    return float((values.max() - values.min()).item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--prompt-groups", type=int, default=32)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--score-microbatch-size", type=int, default=4)
    parser.add_argument("--tensorized-prompt-microbatch-size", type=int, default=4)
    parser.add_argument("--tensorized-members-per-wave-per-gpu", type=int, default=8)
    parser.add_argument("--sigma", type=float, default=1e-3)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    parser.add_argument("--noise-rank", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-completion-length", type=int, default=2048)
    parser.add_argument("--max-prompt-length", type=int, default=18976)
    parser.add_argument("--temperature", type=float, default=1.1)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--kl-temperature", type=float, default=1.1)
    parser.add_argument("--token-clip", type=float, default=0.05)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-save", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument(
        "--start-step",
        type=int,
        default=0,
        help="0-based step to resume from (must match --resume-from).",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="HF checkpoint directory to load instead of --model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_rank = int(__import__("os").environ.get("LOCAL_RANK", "0"))
    world_size = int(__import__("os").environ.get("WORLD_SIZE", "1"))
    if args.pairs % world_size:
        raise ValueError(
            "Antithetic pairs must divide evenly over GPUs; "
            f"got pairs={args.pairs}, world_size={world_size}"
        )
    if args.prompt_groups % world_size:
        raise ValueError("prompt-groups must be divisible by world size")
    if args.score_microbatch_size < 1:
        raise ValueError("score-microbatch-size must be positive")
    if args.prompt_groups % args.tensorized_prompt_microbatch_size:
        raise ValueError(
            "prompt-groups must divide tensorized-prompt-microbatch-size"
        )
    population_members = 2 * args.pairs
    local_population_members = population_members // world_size
    if population_members % args.prompt_groups:
        raise ValueError(
            "Population members must divide evenly into prompt groups; "
            f"got members={population_members}, prompt_groups={args.prompt_groups}"
        )
    members_per_prompt = population_members // args.prompt_groups
    if members_per_prompt < 2 or members_per_prompt % 2:
        raise ValueError(
            "Each prompt needs an even number of members (antithetic pairs); "
            f"got members_per_prompt={members_per_prompt}"
        )
    if args.tensorized_members_per_wave_per_gpu < 2:
        raise ValueError("tensorized-members-per-wave-per-gpu must be at least 2")
    if args.tensorized_members_per_wave_per_gpu % 2:
        raise ValueError("tensorized-members-per-wave-per-gpu must be even")
    wave_members = args.tensorized_members_per_wave_per_gpu
    if members_per_prompt % wave_members:
        raise ValueError(
            "tensorized-members-per-wave-per-gpu must divide members per prompt; "
            f"got members_per_prompt={members_per_prompt}, wave_members={wave_members}"
        )
    local_prompt_count = args.prompt_groups // world_size
    if local_prompt_count % args.tensorized_prompt_microbatch_size:
        raise ValueError(
            "Local prompt count must divide tensorized-prompt-microbatch-size; "
            f"got local_prompts={local_prompt_count}, "
            f"microbatch={args.tensorized_prompt_microbatch_size}"
        )
    if args.checkpoint_every < 0:
        raise ValueError("checkpoint-every must be >= 0 (0 disables)")
    if args.start_step < 0 or args.start_step > args.steps:
        raise ValueError("start-step must be in [0, steps]")
    if args.start_step > 0 and not args.resume_from:
        raise ValueError("start-step > 0 requires --resume-from")
    pairs_per_prompt = members_per_prompt // 2
    generate_batch_size = wave_members * args.tensorized_prompt_microbatch_size
    population_estimator_scale = args.learning_rate / math.sqrt(population_members)

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend="nccl",
        device_id=device,
        timeout=timedelta(hours=3),
    )

    output_dir = Path(args.output_dir)
    if local_rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(
            json.dumps(
                {
                    **vars(args),
                    "world_size": world_size,
                    "fitness": "boxed_correctness",
                    "center_kl_evaluations": 0,
                    "rollout_policy": "one_prompt_per_perturbation",
                    "kl_direction": None,
                    "teacher_policy": "none",
                    "population_layout": "hyperscalees_generations_per_prompt_tensorized_generate",
                    "population_members": population_members,
                    "members_per_prompt": members_per_prompt,
                    "population_normalization_n": population_members,
                    "population_estimator_scale": population_estimator_scale,
                    "fitness_prompt_reduction": "center_within_prompt_group",
                    "generate_batch_size": generate_batch_size,
            "population_waves_per_gpu": (
                (local_prompt_count // args.tensorized_prompt_microbatch_size)
                * (members_per_prompt // wave_members)
            ),
                    "teacher_cache": "none",
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    dist.barrier()

    load_path = args.resume_from or args.model
    tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    policy = AutoModelForCausalLM.from_pretrained(load_path, **model_kwargs).to(device)
    policy.eval()
    policy.config.use_cache = True
    for parameter in policy.parameters():
        parameter.requires_grad_(False)

    selected = replace_target_linears(policy)
    masters = [wrapper.base.weight.detach().float().clone() for _, wrapper in selected]

    dataset = load_dataset("siyanzhao/Openthoughts_math_30k_opsd", split="train")
    order = list(range(len(dataset)))
    random.Random(args.seed).shuffle(order)
    if args.smoke:
        args.steps = min(args.steps, 1)

    metrics_path = output_dir / "train-metrics.jsonl"
    run_start = time.time()
    member_rollouts = args.start_step * population_members
    last_metrics: dict = {}
    if local_rank == 0 and args.start_step > 0 and metrics_path.exists():
        kept = []
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if int(record.get("step", 0)) <= args.start_step:
                kept.append(line)
                last_metrics = record
        metrics_path.write_text(
            ("\n".join(kept) + "\n") if kept else "",
            encoding="utf-8",
        )
        print(
            f"RESUME from={load_path} start_step={args.start_step} "
            f"kept_metrics={len(kept)}",
            flush=True,
        )

    for step in range(args.start_step, args.steps):
        step_start = time.time()
        features = [
            dataset[order[(step * args.prompt_groups + group) % len(order)]]
            for group in range(args.prompt_groups)
        ]
        prompt_texts = [
            student_prompt(tokenizer, feature["problem"]) for feature in features
        ]
        student_prompt_ids = [
            tokenize_prompt(tokenizer, text, args.max_prompt_length)
            for text in prompt_texts
        ]
        ground_truths = [
            ground_truth_from_solution(feature["solution"]) for feature in features
        ]

        pairs_per_gpu = args.pairs // world_size
        local_pair_ids = [
            local_rank * pairs_per_gpu + offset
            for offset in range(pairs_per_gpu)
        ]
        local_member_count = 2 * pairs_per_gpu
        local_rewards = torch.zeros(local_member_count, dtype=torch.float32)
        local_formatted = torch.zeros_like(local_rewards)
        local_lengths = torch.zeros_like(local_rewards)
        prompts_per_forward = args.tensorized_prompt_microbatch_size
        local_prompt_start = local_rank * local_prompt_count

        chunk_pairs = wave_members // 2
        for prompt_offset in range(0, local_prompt_count, prompts_per_forward):
            wave_prompt_indices = [
                local_prompt_start + prompt_offset + index
                for index in range(prompts_per_forward)
            ]
            for member_begin in range(0, members_per_prompt, wave_members):
                member_end = member_begin + wave_members
                wave_pair_ids = []
                wave_sequences = []
                for prompt_index in wave_prompt_indices:
                    for member in range(member_begin, member_end, 2):
                        wave_pair_ids.append(
                            prompt_index * pairs_per_prompt + member // 2
                        )
                    # Prompt-major layout: one prompt's wave members are contiguous.
                    wave_sequences.extend(
                        [student_prompt_ids[prompt_index]] * wave_members
                    )
                input_ids, attention_mask = pad_sequences(
                    wave_sequences,
                    pad_token_id=tokenizer.pad_token_id,
                    device=device,
                    padding_side="left",
                )
                generation_seed = (
                    args.seed + step * population_members + 2 * wave_pair_ids[0]
                )
                with tensorized_member_perturbations(
                    selected,
                    base_seed=args.seed,
                    step=step,
                    pair_ids=wave_pair_ids,
                    sigma=args.sigma,
                    rank=args.noise_rank,
                    prompts_per_forward=1,
                    device=device,
                ):
                    completions = generate_student_from_ids(
                        policy,
                        tokenizer,
                        input_ids,
                        attention_mask,
                        max_completion_length=args.max_completion_length,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        sample_seed=generation_seed,
                    )
                if len(completions) != len(wave_sequences):
                    raise RuntimeError(
                        "Tensorized generate produced the wrong batch size: "
                        f"{len(completions)} vs {len(wave_sequences)}"
                    )
                for row, completion in enumerate(completions):
                    prompt_slot = row // wave_members
                    member_in_prompt = member_begin + (row % wave_members)
                    prompt_index = wave_prompt_indices[prompt_slot]
                    pair_id = wave_pair_ids[
                        prompt_slot * chunk_pairs + (row % wave_members) // 2
                    ]
                    local_pair = pair_id - local_pair_ids[0]
                    local_member = 2 * local_pair + (member_in_prompt % 2)
                    if not 0 <= local_member < local_member_count:
                        raise RuntimeError(
                            f"Wave row {row} mapped to local_member={local_member} "
                            f"outside [0, {local_member_count})"
                        )
                    reward, formatted = boxed_correctness_reward(
                        tokenizer, completion, ground_truths[prompt_index]
                    )
                    local_rewards[local_member] = reward
                    local_formatted[local_member] = formatted
                    local_lengths[local_member] = float(completion.numel())
                rolled = (
                    prompt_offset * members_per_prompt
                    + member_end * prompts_per_forward
                )
                print(
                    f"[rank{local_rank}] step={step + 1} "
                    f"rolled {rolled}/{local_member_count} "
                    f"batch={len(wave_sequences)} elapsed={time.time() - step_start:.0f}s",
                    flush=True,
                )
                del completions, input_ids, attention_mask, wave_sequences

        member_rollouts += population_members
        local_values = torch.stack(
            (local_rewards, local_formatted, local_lengths), dim=-1
        ).to(device)
        gathered_values = [torch.zeros_like(local_values) for _ in range(world_size)]
        dist.all_gather(gathered_values, local_values)
        # Rank-major flattening preserves [pair0+, pair0-, pair1+, pair1-, ...],
        # which is also [prompt0's 8 members, prompt1's 8 members, ...].
        values = torch.stack(gathered_values).reshape(population_members, 3).cpu()
        raw_scores = values[:, 0].view(args.prompt_groups, members_per_prompt).to(device)
        formatted_scores = values[:, 1].view(args.prompt_groups, members_per_prompt)
        completion_lengths = values[:, 2].view(args.prompt_groups, members_per_prompt)

        denominator = torch.sqrt(raw_scores.var(unbiased=False) + 1e-5)
        normalized = (raw_scores - raw_scores.mean(dim=1, keepdim=True)) / denominator
        update_norm, center_norm = apply_eggroll_update(
            selected,
            masters,
            normalized.reshape(1, population_members),
            base_seed=args.seed,
            step=step,
            sigma=args.sigma,
            learning_rate=args.learning_rate,
            rank=args.noise_rank,
            device=device,
        )
        dist.barrier()
        sync_error = center_fingerprint(masters, device)
        if sync_error != 0.0:
            raise RuntimeError(f"Center policies diverged across ranks: {sync_error}")

        paired_reward_gaps = raw_scores[:, 0::2] - raw_scores[:, 1::2]
        step_seconds = time.time() - step_start
        last_metrics = {
            "step": step + 1,
            "steps": args.steps,
            "member_rollouts": member_rollouts,
            "pairs": args.pairs,
            "population_members": population_members,
            "members_per_prompt": members_per_prompt,
            "population_normalization_n": population_members,
            "population_estimator_scale": population_estimator_scale,
            "fitness_prompt_reduction": "center_within_prompt_group",
            "local_population_members_per_gpu": local_member_count,
            "tensorized_members_per_gpu": args.tensorized_members_per_wave_per_gpu,
            "tensorized_prompt_microbatch_size": args.tensorized_prompt_microbatch_size,
            "generate_batch_size": generate_batch_size,
            "population_waves_per_gpu": (
                local_prompt_count // args.tensorized_prompt_microbatch_size
            ),
            "reward_mean": float(raw_scores.mean().item()),
            "reward_std": float(raw_scores.std(unbiased=False).item()),
            "formatted_rate": float(formatted_scores.mean().item()),
            "normalized_fitness_std": float(normalized.std(unbiased=False).item()),
            "paired_reward_gap_abs_mean": float(paired_reward_gaps.abs().mean().item()),
            "completion_tokens_mean": float(completion_lengths.mean().item()),
            "completion_tokens_max": int(completion_lengths.max().item()),
            "completion_clip_rate": float(
                (completion_lengths >= args.max_completion_length).float().mean().item()
            ),
            "update_norm": update_norm,
            "center_norm": center_norm,
            "relative_update_norm": update_norm / max(center_norm, 1e-12),
            "center_sync_error": sync_error,
            "step_seconds": step_seconds,
            "elapsed_seconds": time.time() - run_start,
        }
        if local_rank == 0:
            append_jsonl(metrics_path, last_metrics)
            print(
                "EGGROLL_AIME_METRICS "
                + json.dumps(last_metrics, sort_keys=True),
                flush=True,
            )

        del (
            local_pair_ids,
            local_rewards,
            local_formatted,
            local_lengths,
            local_values,
            gathered_values,
            values,
            raw_scores,
            formatted_scores,
            completion_lengths,
            normalized,
            paired_reward_gaps,
        )
        gc.collect()
        torch.cuda.empty_cache()

        if (
            not args.skip_save
            and args.checkpoint_every > 0
            and (step + 1) % args.checkpoint_every == 0
        ):
            dist.barrier()
            if local_rank == 0:
                ckpt_dir = output_dir / f"checkpoint-{step + 1:04d}"
                selected = save_center_checkpoint(
                    policy, selected, tokenizer, ckpt_dir
                )
                (ckpt_dir / "train-step.json").write_text(
                    json.dumps(
                        {
                            "step": step + 1,
                            "steps": args.steps,
                            "metrics": last_metrics,
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                print(
                    f"CHECKPOINT_SAVED step={step + 1} dir={ckpt_dir}",
                    flush=True,
                )
            dist.barrier()

    dist.barrier()
    if local_rank == 0:
        if not args.skip_save:
            unwrap_target_linears(policy, selected)
            final_dir = output_dir / "final-model"
            final_dir.mkdir(parents=True, exist_ok=True)
            policy.save_pretrained(final_dir, safe_serialization=True, max_shard_size="4GB")
            tokenizer.save_pretrained(final_dir)
        summary = {
            "status": "complete",
            "train_runtime_seconds": time.time() - run_start,
            "member_rollouts": member_rollouts,
            "last_metrics": last_metrics,
            "target_module_count": len(selected),
            "target_weight_parameter_count": sum(master.numel() for master in masters),
        }
        (output_dir / "train-summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        print("TRAIN_COMPLETE " + json.dumps(summary, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
