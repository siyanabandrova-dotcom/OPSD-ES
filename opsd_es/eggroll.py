"""PyTorch port of hyperscalees.noiser.eggroll.EggRoll (no JAX).

Same contract:
  - rank-1 (default) A, B per Linear, antithetic ±sigma
  - forward: y = x W^T + (x B) A^T  (A already includes signed sigma)
  - convert_prompt_centered_fitnesses: center per prompt with global raw std,
    then mean over prompts (prompts are repeated measurements, not extra N)
  - update: W += (eta/sqrt(N)) * sum_b (z[b,+] - z[b,-]) A_+ B^T
    reconstructed from deterministic pair seeds on every rank;
    A/B match the BF16-quantized noise used in forward (FP32(BF16(ε)))
  - persistent FP32 master weights; BF16 copy is for forward only
  - wave forward: K members share one batched residual (x B_k) A_k^T
"""

from __future__ import annotations

import hashlib
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


TARGET_LINEAR_NAMES = frozenset(
    {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
)


def convert_fitnesses(raw_scores: torch.Tensor, group_size: int = 0) -> torch.Tensor:
    """Match EggRoll.convert_fitnesses (JAX var is population variance)."""
    raw = raw_scores.reshape(-1).to(dtype=torch.float32)
    global_std = torch.sqrt(raw.var(unbiased=False) + 1e-5)
    if group_size == 0:
        return (raw - raw.mean()) / global_std
    grouped = raw.reshape(-1, group_size)
    centered = grouped - grouped.mean(dim=-1, keepdim=True)
    return (centered / global_std).reshape(-1)


def convert_member_fitnesses(raw_scores: torch.Tensor, n_members: int) -> torch.Tensor:
    """Average prompt replicates per member, then z-score the M scores.

    Prompt reuse is measurement noise reduction, not extra search directions.
    Population scaling (center / std and later sqrt(N)) therefore sees M numbers.
    """
    n_members = int(n_members)
    raw = raw_scores.reshape(-1).to(dtype=torch.float32)
    n_scores = int(raw.numel())
    if n_members < 1 or n_scores % n_members != 0:
        raise ValueError(f"fitnesses {n_scores} not divisible by population {n_members}")
    n_prompts = n_scores // n_members
    per_member = raw.reshape(n_prompts, n_members).mean(dim=0)
    return convert_fitnesses(per_member, group_size=0)


def convert_prompt_centered_fitnesses(raw_scores: torch.Tensor) -> torch.Tensor:
    """Center each prompt with the global raw-score std, then mean over prompts.

    ``raw_scores`` is [n_members, n_prompts]. Prompts are repeated measurements;
    the returned vector has length N = n_members.
    """
    raw = raw_scores.to(dtype=torch.float32)
    if raw.ndim != 2:
        raise ValueError(f"expected [members, prompts], got {tuple(raw.shape)}")
    global_std = torch.sqrt(raw.var(unbiased=False) + 1e-5)
    centered = raw - raw.mean(dim=0, keepdim=True)
    return (centered / global_std).mean(dim=1)


def prompt_accum_for_target(n_prompts: int, world: int, local_batch: int) -> int:
    """Local generate rounds so ``world * local_batch * accum >= n_prompts``."""
    n_prompts = int(n_prompts)
    world = int(world)
    local_batch = int(local_batch)
    if n_prompts < 1:
        raise ValueError(f"n_prompts must be >= 1, got {n_prompts}")
    if world < 1:
        raise ValueError(f"world must be >= 1, got {world}")
    if local_batch < 1:
        raise ValueError(f"local_batch must be >= 1, got {local_batch}")
    per_round = world * local_batch
    return (n_prompts + per_round - 1) // per_round


def pair_shard_range(rank: int, world: int, n_pairs: int) -> tuple[int, int]:
    """Half-open pair ID range owned by ``rank``. 8 GPUs × 128 pairs → [16r, 16r+16)."""
    if world < 1:
        raise ValueError(f"world must be >= 1, got {world}")
    if not 0 <= rank < world:
        raise ValueError(f"rank {rank} outside world {world}")
    base, extra = divmod(int(n_pairs), int(world))
    if rank < extra:
        start = rank * (base + 1)
        return start, start + base + 1
    start = extra * (base + 1) + (rank - extra) * base
    return start, start + base


def _seed_from_name(name: str, epoch: int, true_thread: int, base_seed: int) -> int:
    payload = f"{base_seed}|{epoch}|{true_thread}|{name}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % (2**63)


def sample_lora_ab(
    weight: torch.Tensor,
    thread_id: int,
    epoch: int,
    layer_name: str,
    sigma: float,
    rank: int,
    base_seed: int,
    noise_reuse: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Port of get_lora_update_params. weight is (out, in) = (a, b).

    Returns A (out, r) already scaled by ±sigma/sqrt(rank), B (in, r).
    """
    a, b = int(weight.shape[0]), int(weight.shape[1])
    true_epoch = 0 if noise_reuse == 0 else int(epoch) // int(noise_reuse)
    true_thread = int(thread_id) // 2
    signed_sigma = float(sigma) / (float(rank) ** 0.5)
    if int(thread_id) % 2 != 0:
        signed_sigma = -signed_sigma
    gen = torch.Generator(device="cpu")
    gen.manual_seed(_seed_from_name(layer_name, true_epoch, true_thread, base_seed))
    lora = torch.randn(a + b, rank, generator=gen, dtype=torch.float32)
    b_mat = lora[:b]
    a_mat = lora[b:] * signed_sigma
    return a_mat, b_mat


def quantize_noise_like_forward(
    a_mat: torch.Tensor,
    b_mat: torch.Tensor,
    forward_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match forward: noise is cast to activation/weight dtype, then lifted to FP32.

    Scoring sees BF16(ε) when the model runs in BF16; the ES update must use the
    same realized direction FP32(BF16(ε)), not the original unquantized FP32 ε.
    """
    if forward_dtype == torch.float32:
        return a_mat.float(), b_mat.float()
    a_q = a_mat.to(dtype=forward_dtype).to(dtype=torch.float32)
    b_q = b_mat.to(dtype=forward_dtype).to(dtype=torch.float32)
    return a_q, b_q


def lora_mean_update(
    noises: list[dict[str, tuple[torch.Tensor, torch.Tensor]]],
    fitnesses: torch.Tensor,
    layer_name: str,
    forward_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """mean_i ŵ_i A_i B_i^T for one Linear, matching _simple_lora_update."""
    flat = fitnesses.reshape(-1).to(dtype=torch.float32)
    n_scores = int(flat.numel())
    n_members = len(noises)
    if n_scores % n_members != 0:
        raise ValueError(f"fitnesses {n_scores} not divisible by population {n_members}")
    prompts = n_scores // n_members
    grouped = flat.reshape(prompts, n_members)
    a0, b0 = noises[0][layer_name]
    if forward_dtype is None:
        forward_dtype = a0.dtype if a0.dtype in (torch.bfloat16, torch.float16) else torch.float32
    accum = torch.zeros(a0.shape[0], b0.shape[0], dtype=torch.float32, device=a0.device)
    grouped = grouped.to(device=a0.device)
    for idx, noise in enumerate(noises):
        a_mat, b_mat = noise[layer_name]
        a_q, b_q = quantize_noise_like_forward(a_mat, b_mat, forward_dtype)
        scale = grouped[:, idx].sum() / float(n_scores)
        accum = accum + scale * (a_q @ b_q.T)
    return accum


class EggRollLinear(nn.Linear):
    """nn.Linear plus optional rank-1 residual x @ B @ A.T (A includes sigma).

    Updates accumulate on a persistent FP32 ``master_weight``; ``weight`` is a
    forward snapshot in the model dtype (typically BF16).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._eggroll_A: torch.Tensor | None = None
        self._eggroll_B: torch.Tensor | None = None
        self._eggroll_A_wave: torch.Tensor | None = None
        self._eggroll_B_wave: torch.Tensor | None = None
        # Plain attribute, not a buffer: DDP broadcasts buffers and NCCL cannot
        # touch a CPU tensor. ``module.to(cuda)`` also leaves this copy on CPU.
        self._eggroll_master = self.weight.detach().cpu().to(dtype=torch.float32)
        self._register_state_dict_hook(EggRollLinear._write_master_state)
        self.register_load_state_dict_pre_hook(EggRollLinear._read_master_state)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "EggRollLinear":
        wrapped = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        with torch.no_grad():
            wrapped.weight.copy_(linear.weight)
            wrapped.master_weight = linear.weight.detach().cpu().to(dtype=torch.float32)
            if linear.bias is not None:
                wrapped.bias.copy_(linear.bias)
        return wrapped

    @property
    def master_weight(self) -> torch.Tensor:
        return self._eggroll_master

    @master_weight.setter
    def master_weight(self, value: torch.Tensor) -> None:
        self._eggroll_master = value.detach().cpu()

    @staticmethod
    def _write_master_state(module, state_dict, prefix, local_metadata) -> None:
        state_dict[prefix + "master_weight"] = module._eggroll_master.detach().cpu()

    @staticmethod
    def _read_master_state(
        module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ) -> None:
        key = prefix + "master_weight"
        if key in state_dict:
            module._eggroll_master = state_dict.pop(key).detach().cpu()

    def set_noise(self, a_mat: torch.Tensor | None, b_mat: torch.Tensor | None) -> None:
        self._eggroll_A = a_mat
        self._eggroll_B = b_mat
        self._eggroll_A_wave = None
        self._eggroll_B_wave = None

    def set_wave_noise(self, a_wave: torch.Tensor | None, b_wave: torch.Tensor | None) -> None:
        """A_wave (K, out, r), B_wave (K, in, r). Batch layout is K consecutive copies of B."""
        self._eggroll_A = None
        self._eggroll_B = None
        self._eggroll_A_wave = a_wave
        self._eggroll_B_wave = b_wave

    def ensure_master_weight(self) -> torch.Tensor:
        """Return the FP32 accumulator, recreating it only if missing."""
        master = self._eggroll_master
        if master is None or master.shape != self.weight.shape:
            self._eggroll_master = self.weight.detach().cpu().to(dtype=torch.float32)
            return self._eggroll_master
        if master.device.type != "cpu":
            self._eggroll_master = master.detach().cpu()
        return self._eggroll_master

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        a_wave, b_wave = self._eggroll_A_wave, self._eggroll_B_wave
        if a_wave is not None and b_wave is not None:
            wave = int(a_wave.shape[0])
            if x.shape[0] % wave != 0:
                raise ValueError(f"batch {x.shape[0]} is not divisible by wave size {wave}")
            flat = x.reshape(wave, -1, x.shape[-1])
            b_mat = b_wave.to(dtype=x.dtype)
            a_mat = a_wave.to(dtype=x.dtype)
            hidden = torch.einsum("kbd,kdr->kbr", flat, b_mat)
            residual = torch.einsum("kbr,kor->kbo", hidden, a_mat)
            return out + residual.reshape(out.shape)
        a_mat, b_mat = self._eggroll_A, self._eggroll_B
        if a_mat is None or b_mat is None:
            return out
        residual = x.matmul(b_mat.to(dtype=x.dtype))
        residual = residual.matmul(a_mat.to(dtype=x.dtype).transpose(0, 1))
        return out + residual


def wrap_eggroll_linears(model: nn.Module, target_names: Iterable[str] = TARGET_LINEAR_NAMES) -> int:
    """Replace matching nn.Linear modules with EggRollLinear. Returns wrap count."""
    targets = set(target_names)
    wrapped = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or isinstance(module, EggRollLinear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in targets:
            continue
        parent_path, _, _ = name.rpartition(".")
        parent = model if parent_path == "" else model.get_submodule(parent_path)
        setattr(parent, leaf, EggRollLinear.from_linear(module))
        wrapped += 1
    return wrapped


def iter_eggroll_linears(model: nn.Module) -> list[tuple[str, EggRollLinear]]:
    return [(name, mod) for name, mod in model.named_modules() if isinstance(mod, EggRollLinear)]


class EggRollController:
    def __init__(
        self,
        model: nn.Module,
        sigma: float = 1e-3,
        rank: int = 1,
        lr_scale: float = 0.2,
        noise_reuse: int = 1,
        base_seed: int = 0,
    ):
        self.model = model
        self.sigma = float(sigma)
        self.rank = int(rank)
        self.lr_scale = float(lr_scale)
        self.noise_reuse = int(noise_reuse)
        self.base_seed = int(base_seed)
        self.layers = iter_eggroll_linears(model)
        if not self.layers:
            raise ValueError("No EggRollLinear modules found; call wrap_eggroll_linears first")

    def sample_member(self, thread_id: int, epoch: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        noise = {}
        for name, linear in self.layers:
            a_mat, b_mat = sample_lora_ab(
                linear.weight,
                thread_id,
                epoch,
                name,
                self.sigma,
                self.rank,
                self.base_seed,
                self.noise_reuse,
            )
            device = linear.weight.device
            if device.type != "cpu":
                a_mat = a_mat.to(device=device, dtype=torch.float32, non_blocking=True)
                b_mat = b_mat.to(device=device, dtype=torch.float32, non_blocking=True)
            noise[name] = (a_mat, b_mat)
        return noise

    def activate(self, noise: dict[str, tuple[torch.Tensor, torch.Tensor]] | None) -> None:
        for name, linear in self.layers:
            if noise is None:
                linear.set_noise(None, None)
            else:
                a_mat, b_mat = noise[name]
                linear.set_noise(a_mat, b_mat)

    def activate_wave(self, noises: list[dict[str, tuple[torch.Tensor, torch.Tensor]]]) -> None:
        """Apply K members in one forward. Input batch layout is K copies of B stacked."""
        if not noises:
            self.clear()
            return
        if len(noises) == 1:
            self.activate(noises[0])
            return
        for name, linear in self.layers:
            a_wave = torch.stack([noise[name][0] for noise in noises], dim=0)
            b_wave = torch.stack([noise[name][1] for noise in noises], dim=0)
            linear.set_wave_noise(a_wave, b_wave)

    def clear(self) -> None:
        self.activate(None)

    def center_fingerprint(self) -> torch.Tensor:
        """Deterministic FP64 checksum of FP32 master weights (two scalars)."""
        total = torch.zeros(2, dtype=torch.float64)
        for _, linear in self.layers:
            weight = linear.ensure_master_weight().detach().reshape(-1).to(dtype=torch.float64)
            total[0] += weight.sum()
            total[1] += weight.square().sum()
        return total

    @torch.no_grad()
    def apply_antithetic_update(self, member_z: torch.Tensor, epoch: int) -> None:
        """W += (eta/sqrt(N)) * sum_b (z[b,+] - z[b,-]) A_+ B^T from pair seeds.

        ``member_z`` is length N (prompt-averaged, prompt-centered). Prompts do
        not enter sqrt(N). Every rank reconstructs the same 128 directions.

        Noise is quantized like the BF16 forward (FP32(BF16(ε))) so the update
        direction matches the perturbation that was actually scored.
        """
        z_flat = member_z.reshape(-1).to(dtype=torch.float32)
        n_members = int(z_flat.numel())
        if n_members < 2 or n_members % 2 != 0:
            raise ValueError(f"antithetic update needs even population, got {n_members}")
        n_pairs = n_members // 2
        step = self.lr_scale / (n_members**0.5)
        z_flat = z_flat.cpu()
        for name, linear in self.layers:
            device = linear.weight.device
            forward_dtype = linear.weight.dtype
            out_f, in_f = int(linear.weight.shape[0]), int(linear.weight.shape[1])
            accum = torch.zeros(out_f, in_f, dtype=torch.float32, device=device)
            for pair in range(n_pairs):
                a_mat, b_mat = sample_lora_ab(
                    linear.weight,
                    2 * pair,
                    epoch,
                    name,
                    self.sigma,
                    self.rank,
                    self.base_seed,
                    self.noise_reuse,
                )
                coeff = float(z_flat[2 * pair] - z_flat[2 * pair + 1])
                a_q, b_q = quantize_noise_like_forward(a_mat, b_mat, forward_dtype)
                a_dev = a_q.to(device=device, dtype=torch.float32, non_blocking=True)
                b_dev = b_q.to(device=device, dtype=torch.float32, non_blocking=True)
                accum.add_(a_dev @ b_dev.T, alpha=coeff)
            master = linear.ensure_master_weight()
            master.add_(accum.detach().cpu(), alpha=step)
            linear.weight.data.copy_(
                master.to(device=device, dtype=linear.weight.dtype)
            )

    @torch.no_grad()
    def apply_update(
        self,
        noises: list[dict[str, tuple[torch.Tensor, torch.Tensor]]],
        fitnesses: torch.Tensor,
    ) -> None:
        """W_fp32 -= lr * (-sqrt(M) * mean ŵ A B^T); BF16 weight is a forward snapshot.

        M is len(noises), the unique population, even if fitnesses still has
        prompt replicates. sqrt(B*M) would treat repeated KL as new directions.
        The SGD step stays in FP32 so small updates do not vanish in BF16.
        Directions match the BF16-quantized noise used in forward.
        """
        n_members = len(noises)
        scale = -(n_members**0.5)
        lr = self.lr_scale
        for name, linear in self.layers:
            g_lora = lora_mean_update(
                noises, fitnesses, name, forward_dtype=linear.weight.dtype
            )
            g_es = g_lora * scale
            master = linear.ensure_master_weight()
            master.add_(g_es.detach().cpu().to(dtype=torch.float32), alpha=-lr)
            linear.weight.data.copy_(
                master.to(device=linear.weight.device, dtype=linear.weight.dtype)
            )
