"""Padding and EOS cleanup for decoder-only on-policy generate."""

from __future__ import annotations

from collections.abc import Iterable

import torch


def collect_stop_token_ids(pad_token_id, eos_token_id) -> list[int]:
    ids: set[int] = set()
    for value in (pad_token_id, eos_token_id):
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            value = value.tolist()
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            ids.update(int(v) for v in value)
        else:
            ids.add(int(value))
    return sorted(ids)


def repad_sequences(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    pad_token_id: int,
    padding_side: str,
    max_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rebuild a batch as left- or right-padded using the attention mask."""
    if padding_side not in ("left", "right"):
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side!r}")
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be 2D, got shape {tuple(input_ids.shape)}")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.masked_fill(input_ids == pad_token_id, 0)
    contents = []
    for row_ids, row_mask in zip(input_ids, attention_mask):
        kept = row_ids[row_mask.bool()]
        if kept.numel() == 0:
            kept = row_ids.new_full((1,), int(pad_token_id))
        contents.append(kept)
    max_len = max(row.numel() for row in contents)
    if max_length is not None:
        max_len = max(max_len, int(max_length))
    padded_rows = []
    mask_rows = []
    for kept in contents:
        n_pad = max_len - kept.numel()
        pad = kept.new_full((n_pad,), int(pad_token_id))
        zeros = torch.zeros(n_pad, dtype=attention_mask.dtype, device=kept.device)
        ones = torch.ones(kept.numel(), dtype=attention_mask.dtype, device=kept.device)
        if padding_side == "left":
            padded_rows.append(torch.cat([pad, kept], dim=0))
            mask_rows.append(torch.cat([zeros, ones], dim=0))
        else:
            padded_rows.append(torch.cat([kept, pad], dim=0))
            mask_rows.append(torch.cat([ones, zeros], dim=0))
    return torch.stack(padded_rows, dim=0), torch.stack(mask_rows, dim=0)


def truncate_after_stop(
    completion_ids: torch.Tensor,
    stop_token_ids: Iterable[int],
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the first EOS/pad; replace tokens after it with pad.

    Returns (cleaned_ids, keep_mask) where keep_mask is 1 through the first
    stop token inclusive, and 0 afterwards. Rows with no stop stay 1.
    """
    if completion_ids.ndim == 1:
        completion_ids = completion_ids.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    out = completion_ids.clone()
    stop_ids = [int(x) for x in stop_token_ids]
    if out.numel() == 0 or not stop_ids:
        keep = torch.ones_like(out, dtype=torch.long)
        return (out.squeeze(0), keep.squeeze(0)) if squeeze else (out, keep)
    stop = torch.zeros_like(out, dtype=torch.bool)
    for sid in stop_ids:
        stop |= out == sid
    has_stop = stop.any(dim=-1)
    first = stop.to(dtype=torch.long).argmax(dim=-1)
    positions = torch.arange(out.shape[-1], device=out.device).unsqueeze(0)
    after = has_stop.unsqueeze(-1) & (positions > first.unsqueeze(-1))
    out = out.masked_fill(after, int(pad_token_id))
    keep = (~after).to(dtype=torch.long)
    if squeeze:
        return out.squeeze(0), keep.squeeze(0)
    return out, keep


def trim_token_list_after_stop(token_ids: list[int], stop_token_ids: Iterable[int]) -> list[int]:
    """Drop tokens after the first stop id, keeping that stop token."""
    stops = {int(x) for x in stop_token_ids}
    if not stops:
        return list(token_ids)
    for idx, token in enumerate(token_ids):
        if int(token) in stops:
            return token_ids[: idx + 1]
    return list(token_ids)
