#!/usr/bin/env python3
"""Remove EggRoll master_weight buffers from HF safetensors checkpoints for vLLM."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def strip_master_weights(checkpoint_dir: Path, *, inplace: bool = False) -> Path:
    checkpoint_dir = checkpoint_dir.resolve()
    if not checkpoint_dir.is_dir():
        raise SystemExit(f"not a directory: {checkpoint_dir}")

    marker = checkpoint_dir / ".vllm_stripped"
    if marker.exists():
        return checkpoint_dir

    work_dir = checkpoint_dir
    if not inplace:
        work_dir = checkpoint_dir.parent / f"{checkpoint_dir.name}_vllm"
        if work_dir.exists() and (work_dir / ".vllm_stripped").exists():
            return work_dir
        if work_dir.exists():
            shutil.rmtree(work_dir)
        shutil.copytree(checkpoint_dir, work_dir, symlinks=True)

    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as exc:
        raise SystemExit(f"safetensors required: {exc}") from exc

    removed = 0
    for st_path in sorted(work_dir.glob("*.safetensors")):
        drop_keys: list[str] = []
        keep: dict = {}
        with safe_open(st_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if ".master_weight" in key:
                    drop_keys.append(key)
                    continue
                keep[key] = f.get_tensor(key)
        if drop_keys:
            removed += len(drop_keys)
            save_file(keep, st_path)

    index_path = work_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        new_map = {k: v for k, v in weight_map.items() if ".master_weight" not in k}
        dropped = len(weight_map) - len(new_map)
        if dropped:
            removed += dropped
            index["weight_map"] = new_map
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(index, f, indent=2)

    (work_dir / ".vllm_stripped").write_text(f"removed={removed}\n", encoding="utf-8")
    print(f"stripped {removed} master_weight keys in {work_dir}", file=sys.stderr)
    return work_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="modify checkpoint in place (default: write sibling *_vllm dir)",
    )
    args = parser.parse_args()
    out = strip_master_weights(args.checkpoint_dir, inplace=args.inplace)
    print(out)


if __name__ == "__main__":
    main()
