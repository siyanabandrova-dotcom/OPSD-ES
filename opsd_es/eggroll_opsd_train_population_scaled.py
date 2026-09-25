#!/usr/bin/env python3
"""Vanilla EGGROLL adapted to the released OPSD teacher fitness.

The evolutionary estimator follows HyperscaleES commit
e25f644a71e4c87bb6749617ec6b24fd5a3828a2:

* adjacent population members use mirrored low-rank perturbations;
* fitness is centered within prompt groups and divided by the population-wide
  standard deviation;
* repeated prompt measurements are averaged per population member;
* the update is the member-fitness mean perturbation multiplied by sqrt(N),
  where N is the number of perturbation members rather than prompt-member
  evaluations;
* SGD applies the update to the center policy.

The environment/fitness follows OPSD commit
7448751f307a9cdbcc1246dd1565a1a605b443df: a non-thinking student samples an
on-policy trajectory and the fixed, unperturbed, privileged teacher scores that
same trajectory with full-vocabulary forward KL and pointwise clipping.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


TARGET_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


class EggrollLinear(nn.Module):
    """Linear layer with one runtime EGGROLL low-rank perturbation."""

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.noise_a: torch.Tensor | None = None  # [out, rank]
        self.noise_b: torch.Tensor | None = None  # [rank, in]
        self.noise_scale = 0.0
        self.batch_noise_a: torch.Tensor | None = None  # [members, out, rank]
        self.batch_noise_b: torch.Tensor | None = None  # [members, rank, in]
        self.batch_noise_scale: torch.Tensor | None = None  # [members]
        self.batch_member_indices: torch.Tensor | None = None  # [batch]

    def set_noise(self, a: torch.Tensor, b: torch.Tensor, scale: float) -> None:
        self.noise_a = a
        self.noise_b = b
        self.noise_scale = float(scale)

    def clear_noise(self) -> None:
        self.noise_a = None
        self.noise_b = None
        self.noise_scale = 0.0

    def set_batch_noise(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        scale: torch.Tensor,
        member_indices: torch.Tensor,
    ) -> None:
        self.batch_noise_a = a
        self.batch_noise_b = b
        self.batch_noise_scale = scale
        self.batch_member_indices = member_indices

    def clear_batch_noise(self) -> None:
        self.batch_noise_a = None
        self.batch_noise_b = None
        self.batch_noise_scale = None
        self.batch_member_indices = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.batch_noise_a is not None and self.batch_noise_b is not None:
            if x.ndim != 3 or self.batch_member_indices is None:
                raise RuntimeError(
                    "Tensorized EGGROLL noise requires [batch, sequence, hidden] input"
                )
            if x.shape[0] != self.batch_member_indices.numel():
                raise RuntimeError(
                    f"Tensorized EGGROLL batch mismatch: x={x.shape[0]}, "
                    f"indices={self.batch_member_indices.numel()}"
                )
            indices = self.batch_member_indices
            a = self.batch_noise_a.index_select(0, indices)
            b = self.batch_noise_b.index_select(0, indices)
            scale = self.batch_noise_scale.index_select(0, indices)
            projected = torch.einsum("bsi,bri->bsr", x, b)
            low_rank = torch.einsum("bsr,bor->bso", projected, a)
            y = y + low_rank * scale[:, None, None]
        if self.noise_a is not None and self.noise_b is not None:
            low_rank = F.linear(F.linear(x, self.noise_b), self.noise_a)
            y = y + low_rank * self.noise_scale
        return y


def replace_target_linears(model: nn.Module) -> list[tuple[str, EggrollLinear]]:
    selected: list[tuple[str, EggrollLinear]] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or name.rsplit(".", 1)[-1] not in TARGET_MODULES:
            continue
        parent_name, leaf = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        wrapper = EggrollLinear(module)
        setattr(parent, leaf, wrapper)
        selected.append((name, wrapper))
    if not selected:
        raise RuntimeError("No OPSD LoRA target modules were found")
    return selected


def unwrap_target_linears(model: nn.Module, selected: list[tuple[str, EggrollLinear]]) -> None:
    for name, wrapper in selected:
        parent_name, leaf = name.rsplit(".", 1)
        setattr(model.get_submodule(parent_name), leaf, wrapper.base)


def wrap_target_linears(model: nn.Module, selected: list[tuple[str, EggrollLinear]]) -> None:
    for name, wrapper in selected:
        parent_name, leaf = name.rsplit(".", 1)
        setattr(model.get_submodule(parent_name), leaf, wrapper)


def noise_seed(base_seed: int, step: int, pair_id: int, parameter_id: int) -> int:
    # Deterministic integer mixing; the adjacent +/- members share pair_id.
    value = (
        int(base_seed)
        + 1_000_003 * int(step)
        + 97_409 * int(pair_id)
        + 65_537 * int(parameter_id)
    )
    return value % (2**63 - 1)


def make_low_rank_noise(
    wrapper: EggrollLinear,
    *,
    seed: int,
    rank: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    weight = wrapper.base.weight
    out_features, in_features = weight.shape
    # HyperscaleES samples noise in the parameter dtype.
    a = torch.randn(
        (out_features, rank), generator=generator, device=device, dtype=weight.dtype
    )
    b = torch.randn(
        (rank, in_features), generator=generator, device=device, dtype=weight.dtype
    )
    return a, b


@contextmanager
def member_perturbation(
    selected: list[tuple[str, EggrollLinear]],
    *,
    base_seed: int,
    step: int,
    pair_id: int,
    sign: float,
    sigma: float,
    rank: int,
    device: torch.device,
):
    scale = sign * sigma / math.sqrt(rank)
    try:
        for parameter_id, (_, wrapper) in enumerate(selected):
            a, b = make_low_rank_noise(
                wrapper,
                seed=noise_seed(base_seed, step, pair_id, parameter_id),
                rank=rank,
                device=device,
            )
            wrapper.set_noise(a, b, scale)
        yield
    finally:
        for _, wrapper in selected:
            wrapper.clear_noise()


@contextmanager
def tensorized_member_perturbations(
    selected: list[tuple[str, EggrollLinear]],
    *,
    base_seed: int,
    step: int,
    pair_ids: list[int],
    sigma: float,
    rank: int,
    prompts_per_forward: int,
    device: torch.device,
):
    """Install several mirrored members as batch-specific low-rank deltas."""
    member_specs = [
        (pair_id, sign)
        for pair_id in pair_ids
        for sign in (1.0, -1.0)
    ]
    member_indices = torch.arange(
        len(member_specs), dtype=torch.long, device=device
    ).repeat(prompts_per_forward)
    scales = torch.tensor(
        [sign * sigma / math.sqrt(rank) for _, sign in member_specs],
        dtype=torch.float32,
        device=device,
    )
    try:
        for parameter_id, (_, wrapper) in enumerate(selected):
            pair_noises: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
            all_a = []
            all_b = []
            for pair_id, _ in member_specs:
                if pair_id not in pair_noises:
                    pair_noises[pair_id] = make_low_rank_noise(
                        wrapper,
                        seed=noise_seed(base_seed, step, pair_id, parameter_id),
                        rank=rank,
                        device=device,
                    )
                a, b = pair_noises[pair_id]
                all_a.append(a)
                all_b.append(b)
            wrapper.set_batch_noise(
                torch.stack(all_a),
                torch.stack(all_b),
                scales.to(dtype=wrapper.base.weight.dtype),
                member_indices,
            )
        yield
    finally:
        for _, wrapper in selected:
            wrapper.clear_batch_noise()


def student_prompt(tokenizer, problem: str) -> str:
    message = (
        f"Problem: {problem}\n\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": message}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def teacher_prompt(tokenizer, problem: str, solution: str) -> str:
    transition = (
        "\n\nAfter reading the reference solution above, make sure you truly understand "
        "the reasoning behind each step — do not copy or paraphrase it. Now, using your "
        "own words and independent reasoning, derive the same final answer to the problem above. "
        "Think step by step, explore different approaches, and don't be afraid to backtrack "
        "or reconsider if something doesn't work out:\n"
    )
    message = (
        f"Problem: {problem}\n\n"
        "Here is a reference solution to this problem:\n"
        f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n"
        f"{transition}\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": message}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


@torch.inference_mode()
def generate_student_batch(
    model: nn.Module,
    tokenizer,
    prompt_texts: list[str],
    *,
    max_prompt_length: int,
    max_completion_length: int,
    temperature: float,
    top_p: float,
    top_k: int,
    sample_seed: int,
    device: torch.device,
) -> list[torch.Tensor]:
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
    )
    tokenizer.padding_side = old_padding_side
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    torch.manual_seed(sample_seed)
    torch.cuda.manual_seed(sample_seed)
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_completion_length,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    padded_prompt_length = input_ids.shape[1]
    completions = []
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    for row in output:
        completion = row[padded_prompt_length:]
        # ``generate`` right-pads finished rows to the longest completion in
        # the batch.  Those padding tokens are not part of the trajectory and
        # must not enter either the reported length or privileged KL fitness.
        if eos_token_id is not None:
            eos_positions = (completion == eos_token_id).nonzero(as_tuple=False)
            if eos_positions.numel():
                completion = completion[: int(eos_positions[0].item()) + 1]
        if pad_token_id is not None and pad_token_id != eos_token_id:
            pad_positions = (completion == pad_token_id).nonzero(as_tuple=False)
            if pad_positions.numel():
                completion = completion[: int(pad_positions[0].item())]
        completions.append(completion.detach())
    return completions


def clipped_forward_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float,
    token_clip: float,
    chunk_tokens: int = 32,
) -> float:
    total = torch.zeros((), device=student_logits.device, dtype=torch.float32)
    token_count = student_logits.shape[0]
    for start in range(0, token_count, chunk_tokens):
        stop = min(start + chunk_tokens, token_count)
        s = F.log_softmax(student_logits[start:stop] / temperature, dim=-1)
        t = F.log_softmax(teacher_logits[start:stop] / temperature, dim=-1)
        pointwise = F.kl_div(s, t, reduction="none", log_target=True)
        pointwise = pointwise.clamp(max=token_clip)
        total += pointwise.float().sum()
        del s, t, pointwise
    return float((total / max(token_count, 1)).item())


@torch.inference_mode()
def privileged_kl_fitness(
    student: nn.Module,
    teacher: nn.Module,
    tokenizer,
    *,
    problem: str,
    solution: str,
    completion: torch.Tensor,
    max_prompt_length: int,
    temperature: float,
    token_clip: float,
    device: torch.device,
) -> tuple[float, float]:
    s_text = student_prompt(tokenizer, problem)
    t_text = teacher_prompt(tokenizer, problem, solution)
    s_prompt = tokenizer(
        s_text, return_tensors="pt", truncation=True, max_length=max_prompt_length
    ).input_ids.to(device)
    t_prompt = tokenizer(
        t_text, return_tensors="pt", truncation=True, max_length=max_prompt_length
    ).input_ids.to(device)

    completion = completion.to(device)
    s_ids = torch.cat((s_prompt[0], completion), dim=0).unsqueeze(0)
    t_ids = torch.cat((t_prompt[0], completion), dim=0).unsqueeze(0)

    student_outputs = student(input_ids=s_ids, use_cache=False)
    s_start = s_prompt.shape[1] - 1
    student_logits = student_outputs.logits[0, s_start:-1].detach()
    del student_outputs, s_ids

    teacher_outputs = teacher(input_ids=t_ids, use_cache=False)
    t_start = t_prompt.shape[1] - 1
    teacher_logits = teacher_outputs.logits[0, t_start:-1].detach()
    del teacher_outputs, t_ids

    if student_logits.shape != teacher_logits.shape:
        raise RuntimeError(
            f"Student/teacher completion logits differ: {student_logits.shape} vs {teacher_logits.shape}"
        )
    kl = clipped_forward_kl(
        student_logits,
        teacher_logits,
        temperature=temperature,
        token_clip=token_clip,
    )
    del student_logits, teacher_logits
    # EGGROLL maximizes fitness, whereas OPSD minimizes KL.
    return -kl, kl


@torch.inference_mode()
def apply_eggroll_update(
    selected: list[tuple[str, EggrollLinear]],
    masters: list[torch.Tensor],
    normalized_fitness: torch.Tensor,
    *,
    base_seed: int,
    step: int,
    sigma: float,
    learning_rate: float,
    rank: int,
    device: torch.device,
) -> tuple[float, float]:
    groups, population_members = normalized_fitness.shape
    if groups < 1:
        raise ValueError("EGGROLL requires at least one prompt group")
    if population_members % 2:
        raise ValueError("EGGROLL requires an even number of members per prompt")
    # Each antithetic member is one population sample. Reusing the same member
    # on several prompts reduces its fitness noise but does not create new
    # perturbation directions, so average those repeated measurements first.
    pair_coefficients = [0.0 for _ in range(population_members // 2)]
    for group in range(groups):
        for local_pair in range(population_members // 2):
            plus = float(normalized_fitness[group, 2 * local_pair].item())
            minus = float(normalized_fitness[group, 2 * local_pair + 1].item())
            pair_coefficients[local_pair] += plus - minus
    pair_coefficients = [coefficient / groups for coefficient in pair_coefficients]
    pair_ids = list(range(population_members // 2))

    # EGGROLL population scaling: eta * sqrt(N) / N = eta / sqrt(N), with
    # N derived from the configured number of antithetic population members.
    estimator_scale = (
        learning_rate * math.sqrt(population_members) / population_members
    )
    update_sq_sum = 0.0
    center_sq_sum = 0.0
    coefficients = torch.tensor(pair_coefficients, device=device, dtype=torch.float32)

    for parameter_id, ((_, wrapper), master) in enumerate(zip(selected, masters)):
        all_a = []
        all_b = []
        for pair_id in pair_ids:
            a, b = make_low_rank_noise(
                wrapper,
                seed=noise_seed(base_seed, step, pair_id, parameter_id),
                rank=rank,
                device=device,
            )
            all_a.append(a.float())
            all_b.append(b.float())
        # General rank support: each pair contributes rank columns.
        a_stack = torch.cat(all_a, dim=1)
        b_stack = torch.cat(all_b, dim=0)
        expanded_coefficients = coefficients.repeat_interleave(rank)
        delta = (a_stack * expanded_coefficients.unsqueeze(0)) @ b_stack
        delta *= (sigma / math.sqrt(rank)) * estimator_scale
        master.add_(delta)
        wrapper.base.weight.copy_(master.to(dtype=wrapper.base.weight.dtype))
        update_sq_sum += float(delta.square().sum().item())
        center_sq_sum += float(master.square().sum().item())
        del all_a, all_b, a_stack, b_stack, expanded_coefficients, delta

    return math.sqrt(update_sq_sum), math.sqrt(center_sq_sum)


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--prompt-groups", type=int, default=4)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2 or world_size % 2:
        raise ValueError("Launch EGGROLL with an even distributed world size")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)

    output_dir = Path(args.output_dir)
    if local_rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(
            json.dumps({**vars(args), "world_size": world_size}, indent=2, sort_keys=True),
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
    student = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).to(device)
    teacher = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).to(device)
    student.eval()
    teacher.eval()
    student.config.use_cache = True
    teacher.config.use_cache = False
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    selected = replace_target_linears(student)
    # FP32 center copies avoid losing small accumulated EGGROLL updates to bf16 rounding.
    masters = [wrapper.base.weight.detach().float().clone() for _, wrapper in selected]

    dataset = load_dataset("siyanzhao/Openthoughts_math_30k_opsd", split="train")
    order = list(range(len(dataset)))
    random.Random(args.seed).shuffle(order)
    if args.smoke:
        args.steps = min(args.steps, 1)

    metrics_path = output_dir / "train-metrics.jsonl"
    run_start = time.time()
    total_rollouts = 0
    last_metrics: dict = {}

    for step in range(args.steps):
        step_start = time.time()
        raw_group_scores = []
        raw_group_kls = []
        completion_lengths = []

        features = []
        for group in range(args.prompt_groups):
            data_position = (step * args.prompt_groups + group) % len(order)
            features.append(dataset[order[data_position]])

        pair_id = local_rank // 2
        sign = 1.0 if local_rank % 2 == 0 else -1.0
        generation_seed = args.seed + step * world_size + local_rank
        with member_perturbation(
            selected,
            base_seed=args.seed,
            step=step,
            pair_id=pair_id,
            sign=sign,
            sigma=args.sigma,
            rank=args.noise_rank,
            device=device,
        ):
            completions = generate_student_batch(
                student,
                tokenizer,
                [student_prompt(tokenizer, feature["problem"]) for feature in features],
                max_prompt_length=args.max_prompt_length,
                max_completion_length=args.max_completion_length,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                sample_seed=generation_seed,
                device=device,
            )
            local_rows = []
            for feature, completion in zip(features, completions):
                fitness, kl = privileged_kl_fitness(
                    student,
                    teacher,
                    tokenizer,
                    problem=feature["problem"],
                    solution=feature["solution"],
                    completion=completion,
                    max_prompt_length=args.max_prompt_length,
                    temperature=args.kl_temperature,
                    token_clip=args.token_clip,
                    device=device,
                )
                local_rows.append(
                    [fitness, kl, float(completion.numel())]
                )

        local = torch.tensor(local_rows, device=device, dtype=torch.float32)
        gathered = [torch.zeros_like(local) for _ in range(world_size)]
        dist.all_gather(gathered, local)
        values = torch.stack(gathered).cpu()  # [members, groups, metrics]
        for group in range(args.prompt_groups):
            raw_group_scores.append(values[:, group, 0])
            raw_group_kls.append(values[:, group, 1])
            completion_lengths.append(values[:, group, 2])
        total_rollouts += world_size * args.prompt_groups
        del completions, local_rows, local, gathered, values
        gc.collect()
        torch.cuda.empty_cache()

        raw_scores = torch.stack(raw_group_scores).to(device)
        raw_kls = torch.stack(raw_group_kls)
        lengths = torch.stack(completion_lengths)
        # Exact EGGROLL group conversion: group-center, population-wide variance.
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

        step_seconds = time.time() - step_start
        last_metrics = {
            "step": step + 1,
            "steps": args.steps,
            "rollouts": total_rollouts,
            "fitness_mean": float(raw_scores.mean().item()),
            "fitness_std": float(raw_scores.std(unbiased=False).item()),
            "kl_mean": float(raw_kls.mean().item()),
            "kl_std": float(raw_kls.std(unbiased=False).item()),
            "completion_tokens_mean": float(lengths.mean().item()),
            "completion_tokens_max": int(lengths.max().item()),
            "completion_clip_rate": float((lengths >= args.max_completion_length).float().mean().item()),
            "normalized_fitness_std": float(normalized.std(unbiased=False).item()),
            "update_norm": update_norm,
            "center_norm": center_norm,
            "relative_update_norm": update_norm / max(center_norm, 1e-12),
            "step_seconds": step_seconds,
            "elapsed_seconds": time.time() - run_start,
        }
        if local_rank == 0:
            append_jsonl(metrics_path, last_metrics)
            print("EGGROLL_METRICS " + json.dumps(last_metrics, sort_keys=True), flush=True)
        del raw_scores, raw_kls, lengths, normalized

    dist.barrier()
    if local_rank == 0:
        unwrap_target_linears(student, selected)
        final_dir = output_dir / "final-model"
        final_dir.mkdir(parents=True, exist_ok=True)
        student.save_pretrained(final_dir, safe_serialization=True, max_shard_size="4GB")
        tokenizer.save_pretrained(final_dir)
        summary = {
            "status": "complete",
            "train_runtime_seconds": time.time() - run_start,
            "total_rollouts": total_rollouts,
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
