#!/usr/bin/env python3
"""Batched shared-rollout EGGROLL for the released OPSD teacher loss.

At each update, the current unperturbed center policy samples one trajectory
per prompt.  Every mirrored EGGROLL perturbation is evaluated by teacher
forcing those *same* trajectories.  This provides common random numbers and
removes the trajectory-distribution confound from vanilla evolutionary OPSD.

For prompt g and member i, the fitness is

    F[g, i] = -L_perturbed[g, i],

where L is OPSD's full-vocabulary, pointwise-clipped forward KL from the fixed
privileged teacher to the non-privileged student.  A center-policy baseline
would cancel algebraically after population centering, so it is not computed.
The privileged teacher distributions are cached once per trajectory on each
worker and reused for both members of that worker's antithetic pair.  With
mirrored members, the update is proportional to (L_minus - L_plus) times the
positive noise direction, which descends the shared-rollout distillation loss.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from eggroll_opsd_train_population_scaled import (
    append_jsonl,
    apply_eggroll_update,
    generate_student_batch,
    member_perturbation,
    replace_target_linears,
    student_prompt,
    teacher_prompt,
    tensorized_member_perturbations,
    unwrap_target_linears,
    wrap_target_linears,
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
) -> tuple[torch.Tensor, torch.Tensor]:
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
        input_ids[row, :length] = sequence.to(device)
        attention_mask[row, :length] = 1
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
    parser.add_argument("--tensorized-prompt-microbatch-size", type=int, default=2)
    parser.add_argument("--tensorized-members-per-wave-per-gpu", type=int, default=8)
    parser.add_argument("--sigma", type=float, default=1e-3)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    parser.add_argument("--noise-rank", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--max-prompt-length", type=int, default=18976)
    parser.add_argument("--temperature", type=float, default=1.1)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--kl-temperature", type=float, default=1.1)
    parser.add_argument("--token-clip", type=float, default=0.05)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-save", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=0)
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
    if args.tensorized_members_per_wave_per_gpu < 2:
        raise ValueError("tensorized-members-per-wave-per-gpu must be at least 2")
    if args.tensorized_members_per_wave_per_gpu % 2:
        raise ValueError("tensorized-members-per-wave-per-gpu must be even")
    if local_population_members % args.tensorized_members_per_wave_per_gpu:
        raise ValueError(
            "Local population members must divide evenly into tensorized waves; "
            f"got local_members={local_population_members}, "
            f"wave_members={args.tensorized_members_per_wave_per_gpu}"
        )
    population_estimator_scale = args.learning_rate / math.sqrt(population_members)

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)

    output_dir = Path(args.output_dir)
    if local_rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(
            json.dumps(
                {
                    **vars(args),
                    "world_size": world_size,
                    "fitness": "negative_perturbed_clipped_kl",
                    "center_kl_evaluations": 0,
                    "rollout_policy": "current_unperturbed_center",
                    "kl_direction": "teacher_to_student",
                    "teacher_policy": "frozen_initial_model",
                    "population_layout": "batched_prompts_tensorized_antithetic_pairs",
                    "population_members": population_members,
                    "population_normalization_n": population_members,
                    "population_estimator_scale": population_estimator_scale,
                    "fitness_prompt_reduction": "mean_per_population_member",
                    "population_waves_per_gpu": (
                        local_population_members
                        // args.tensorized_members_per_wave_per_gpu
                    ),
                    "teacher_cache": "once_per_trajectory_per_worker_reused_for_local_members",
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    dist.barrier()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    policy = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).to(device)
    teacher = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).to(device)
    policy.eval()
    teacher.eval()
    policy.config.use_cache = True
    teacher.config.use_cache = False
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    for parameter in teacher.parameters():
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
    center_rollouts = 0
    likelihood_evaluations = 0
    teacher_trajectory_evaluations = 0
    last_metrics: dict = {}
    local_prompt_count = args.prompt_groups // world_size

    for step in range(args.steps):
        step_start = time.time()
        features = [
            dataset[order[(step * args.prompt_groups + group) % len(order)]]
            for group in range(args.prompt_groups)
        ]
        student_prompt_ids = [
            tokenize_prompt(
                tokenizer,
                student_prompt(tokenizer, feature["problem"]),
                args.max_prompt_length,
            )
            for feature in features
        ]
        teacher_prompt_ids = [
            tokenize_prompt(
                tokenizer,
                teacher_prompt(tokenizer, feature["problem"], feature["solution"]),
                args.max_prompt_length,
            )
            for feature in features
        ]

        local_start = local_rank * local_prompt_count
        local_stop = local_start + local_prompt_count
        local_features = features[local_start:local_stop]
        # No perturbation is active here: trajectories come from the latest center.
        local_completions = generate_student_batch(
            policy,
            tokenizer,
            [student_prompt(tokenizer, feature["problem"]) for feature in local_features],
            max_prompt_length=args.max_prompt_length,
            max_completion_length=args.max_completion_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            sample_seed=args.seed + step * world_size + local_rank,
            device=device,
        )
        completions, completion_lengths = gather_center_completions(
            local_completions,
            max_completion_length=args.max_completion_length,
            pad_token_id=tokenizer.pad_token_id,
            device=device,
        )
        if len(completions) != args.prompt_groups:
            raise RuntimeError("Shared completion gather produced the wrong count")
        center_rollouts += args.prompt_groups

        # Every worker owns one antithetic pair. Cache the fixed privileged
        # teacher targets once locally, then reuse them for both signs.
        teacher_log_probs = cache_privileged_teacher_distributions(
            teacher,
            teacher_prompt_ids=teacher_prompt_ids,
            completions=completions,
            pad_token_id=tokenizer.pad_token_id,
            temperature=args.kl_temperature,
            microbatch_size=args.score_microbatch_size,
            device=device,
        )
        teacher_trajectory_evaluations += args.prompt_groups
        cache_probe = torch.stack(
            (
                teacher_log_probs[0][0, 0].float(),
                teacher_log_probs[-1][-1, -1].float(),
                teacher_log_probs[len(teacher_log_probs) // 2][
                    0, policy.config.vocab_size // 2
                ].float(),
            )
        ).to(device)
        gathered_probes = [torch.zeros_like(cache_probe) for _ in range(world_size)]
        dist.all_gather(gathered_probes, cache_probe)
        probe_values = torch.stack(gathered_probes)
        teacher_cache_sync_error = float(
            (probe_values.max(dim=0).values - probe_values.min(dim=0).values)
            .abs()
            .max()
            .item()
        )
        if teacher_cache_sync_error > 1e-5:
            raise RuntimeError(
                f"Teacher caches differ across workers: {teacher_cache_sync_error}"
            )

        pairs_per_gpu = args.pairs // world_size
        local_pair_ids = [
            local_rank * pairs_per_gpu + offset
            for offset in range(pairs_per_gpu)
        ]
        local_member_count = 2 * pairs_per_gpu
        pairs_per_wave = args.tensorized_members_per_wave_per_gpu // 2
        clipped_waves = []
        true_kl_waves = []
        for pair_start in range(0, pairs_per_gpu, pairs_per_wave):
            wave_pair_ids = local_pair_ids[pair_start : pair_start + pairs_per_wave]
            wave_member_count = 2 * len(wave_pair_ids)
            with tensorized_member_perturbations(
                selected,
                base_seed=args.seed,
                step=step,
                pair_ids=wave_pair_ids,
                sigma=args.sigma,
                rank=args.noise_rank,
                prompts_per_forward=args.tensorized_prompt_microbatch_size,
                device=device,
            ):
                wave_clipped, wave_true = tensorized_policy_divergences_from_cache(
                    policy,
                    student_prompt_ids=student_prompt_ids,
                    completions=completions,
                    teacher_log_probs=teacher_log_probs,
                    member_count=wave_member_count,
                    pad_token_id=tokenizer.pad_token_id,
                    temperature=args.kl_temperature,
                    token_clip=args.token_clip,
                    prompt_microbatch_size=args.tensorized_prompt_microbatch_size,
                    device=device,
                )
            clipped_waves.append(wave_clipped)
            true_kl_waves.append(wave_true)
        perturbed_clipped = torch.cat(clipped_waves, dim=0)
        perturbed_true = torch.cat(true_kl_waves, dim=0)
        if perturbed_clipped.shape != (local_member_count, args.prompt_groups):
            raise RuntimeError(
                "Tensorized population waves produced the wrong score shape: "
                f"{perturbed_clipped.shape}"
            )
        likelihood_evaluations += 2 * args.pairs * args.prompt_groups

        local_values = torch.stack(
            (-perturbed_clipped, perturbed_clipped, perturbed_true), dim=-1
        ).to(device)
        gathered_values = [torch.zeros_like(local_values) for _ in range(world_size)]
        dist.all_gather(gathered_values, local_values)
        # Rank-major flattening preserves [pair0+, pair0-, pair1+, pair1-, ...].
        values = torch.stack(gathered_values).reshape(
            2 * args.pairs, args.prompt_groups, 3
        ).cpu()
        raw_scores = values[:, :, 0].transpose(0, 1).contiguous().to(device)
        perturbed_losses = values[:, :, 1].transpose(0, 1).contiguous()
        perturbed_true_kls = values[:, :, 2].transpose(0, 1).contiguous()

        denominator = torch.sqrt(raw_scores.var(unbiased=False) + 1e-5)
        normalized = (raw_scores - raw_scores.mean(dim=1, keepdim=True)) / denominator
        update_norm, center_norm = apply_eggroll_update(
            selected,
            masters,
            normalized,
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

        paired_loss_gaps = perturbed_losses[:, 1::2] - perturbed_losses[:, 0::2]
        step_seconds = time.time() - step_start
        last_metrics = {
            "step": step + 1,
            "steps": args.steps,
            "center_rollouts": center_rollouts,
            "perturbation_likelihood_evaluations": likelihood_evaluations,
            "teacher_trajectory_evaluations": teacher_trajectory_evaluations,
            "teacher_cache_population_reuses": local_member_count,
            "teacher_cache_sync_error": teacher_cache_sync_error,
            "center_kl_evaluations": 0,
            "pairs": args.pairs,
            "population_members": population_members,
            "population_normalization_n": population_members,
            "population_estimator_scale": population_estimator_scale,
            "fitness_prompt_reduction": "mean_per_population_member",
            "local_population_members_per_gpu": local_member_count,
            "tensorized_members_per_gpu": args.tensorized_members_per_wave_per_gpu,
            "population_waves_per_gpu": (
                local_member_count // args.tensorized_members_per_wave_per_gpu
            ),
            "perturbed_clipped_loss_mean": float(perturbed_losses.mean().item()),
            "perturbed_clipped_loss_std": float(
                perturbed_losses.std(unbiased=False).item()
            ),
            "perturbed_true_kl_mean": float(perturbed_true_kls.mean().item()),
            "improvement_fitness_mean": float(raw_scores.mean().item()),
            "improvement_fitness_std": float(raw_scores.std(unbiased=False).item()),
            "normalized_fitness_std": float(normalized.std(unbiased=False).item()),
            "paired_loss_gap_abs_mean": float(paired_loss_gaps.abs().mean().item()),
            "completion_tokens_mean": float(completion_lengths.float().mean().item()),
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
                "SHARED_EGGROLL_METRICS "
                + json.dumps(last_metrics, sort_keys=True),
                flush=True,
            )

        if (
            not args.skip_save
            and args.checkpoint_every > 0
            and (step + 1) % args.checkpoint_every == 0
            and (step + 1) < args.steps
        ):
            if local_rank == 0:
                unwrap_target_linears(policy, selected)
                try:
                    ckpt_dir = output_dir / f"checkpoint-{step + 1}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    policy.save_pretrained(
                        ckpt_dir, safe_serialization=True, max_shard_size="4GB"
                    )
                    tokenizer.save_pretrained(ckpt_dir)
                    print(
                        f"CHECKPOINT_SAVED step={step + 1} dir={ckpt_dir}",
                        flush=True,
                    )
                finally:
                    wrap_target_linears(policy, selected)
            dist.barrier()

        del (
            local_completions,
            completions,
            completion_lengths,
            teacher_log_probs,
            cache_probe,
            gathered_probes,
            probe_values,
            perturbed_clipped,
            perturbed_true,
            clipped_waves,
            true_kl_waves,
            wave_pair_ids,
            wave_clipped,
            wave_true,
            local_pair_ids,
            local_values,
            gathered_values,
            values,
            raw_scores,
            perturbed_losses,
            perturbed_true_kls,
            normalized,
            paired_loss_gaps,
        )
        gc.collect()
        torch.cuda.empty_cache()

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
            "center_rollouts": center_rollouts,
            "perturbation_likelihood_evaluations": likelihood_evaluations,
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
