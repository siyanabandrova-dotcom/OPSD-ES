#!/usr/bin/env python3
"""Best Avg@12 checkpoint per dataset. A tie keeps the later step."""

import json
import sys
from pathlib import Path


def main() -> None:
    raw = sys.argv[1:]
    if "--" not in raw or len(raw) < 4:
        raise SystemExit(
            "usage: report_best_avg12.py RUN DATASET [DATASET ...] -- STEP [STEP ...]"
        )
    sep = raw.index("--")
    run = Path(raw[0])
    datasets = raw[1:sep]
    steps = [int(item) for item in raw[sep + 1 :]]
    if not datasets or not steps:
        raise SystemExit("need at least one dataset and one step")

    chosen = []
    missing = False
    for dataset in datasets:
        best = None
        for step in steps:
            path = run / f"{dataset}-avg12-checkpoint-{step}-summary.json"
            if not path.exists():
                missing = True
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            pct = float(record["average_at_n_pct"])
            if best is None or pct > best[0] or (pct == best[0] and step > best[1]):
                best = (pct, step, record)
        chosen.append((dataset, best))
        if best is None:
            missing = True

    lines = [
        "| Dataset | Step | Avg@12 |",
        "|---|---:|---|",
    ]
    percents = []
    for dataset, best in chosen:
        if best is None:
            lines.append(f"| {dataset} |  | missing |")
            continue
        pct, step, record = best
        percents.append(pct)
        lines.append(
            "| {dataset} | {step} | {avg:.2f}% ({correct}/{total}) |".format(
                dataset=dataset,
                step=step,
                avg=pct,
                correct=record["average_at_n"],
                total=record["total_solutions"],
            )
        )
    if percents and len(percents) == len(datasets):
        lines.append("")
        lines.append(
            "Mean of the chosen cells: {mean:.2f}%".format(
                mean=sum(percents) / len(percents)
            )
        )
    text = "\n".join(lines) + "\n"
    (run / "best-checkpoint-table.md").write_text(text, encoding="utf-8")
    print(text, end="", flush=True)
    raise SystemExit(1 if missing else 0)


if __name__ == "__main__":
    main()
