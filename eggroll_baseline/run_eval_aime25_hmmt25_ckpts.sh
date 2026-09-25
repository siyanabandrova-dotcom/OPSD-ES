#!/usr/bin/env bash
# Sequential AIME25 then HMMT25 Avg@12 of the tmoff-2048 EGGROLL run:
#   checkpoint-0020, 0040, 0060, 0080, 0100
# Same eval protocol as the AIME24 table (evaluate_math.py, TP=8, thinking ON, val_n=12).
# Only intended difference: --dataset aime25 / hmmt25 instead of aime24.
# Skips a (dataset, step) if its summary already exists. Retries a failed step up to 3 times.
set -uo pipefail

SRC="/home/hid5250/ES_knn_4c6e985_pr6/eggroll_aime_baseline/hemus-direct-eggroll-128-source-20260824"
VENV="${VENV:-/home/hid5250/radicho/OPSD/.venv}"
LOCAL_RUN="${LOCAL_RUN:-/scratch/hid5250/eggroll_aime_baseline/runs/eggroll-aime-tmoff-2048-pairs128-pop256-gn16-20260827-170250}"
PERSIST_RUN="${PERSIST_RUN:-/home/hid5250/ES_knn_4c6e985_pr6/eggroll_aime_baseline/runs/eggroll-aime-tmoff-2048-pairs128-pop256-gn16-20260827-170250}"
EVAL_STEPS="${EVAL_STEPS:-20 40 60 80 100}"
DATASETS="${DATASETS:-aime25 hmmt25}"
MAX_RETRIES="${MAX_RETRIES:-3}"
NPROC="${NPROC:-8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.75}"
SUMMARY_JSON="$PERSIST_RUN/aime25-hmmt25-ckpts-20-40-60-80-100-summary.json"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HF_HOME="${HF_HOME:-/scratch/hid5250/hf-cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/scratch/hid5250/hf-cache/datasets}"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$PERSIST_RUN" "$LOCAL_RUN"

wait_gpus_free() {
  local i n
  for i in $(seq 1 45); do
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d' | wc -l)
    if [[ "$n" -eq 0 ]]; then
      echo "GPUS_FREE after_try=$i"
      return 0
    fi
    echo "GPUS_BUSY n=$n try=$i"
    sleep 2
  done
  echo "GPUS_STILL_BUSY giving up wait"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv || true
  return 1
}

reap_eval_workers() {
  python3 - <<'PY'
import os, signal, time
needles = ("evaluate_math_sharded.py", "evaluate_math.py", "VllmWorker", "EngineCore")
me = os.getpid()
killed = []
for name in os.listdir("/proc"):
    if not name.isdigit():
        continue
    pid = int(name)
    if pid == me:
        continue
    try:
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\x00", b" ").decode("utf-8", "replace")
        uid = os.stat(f"/proc/{pid}").st_uid
    except OSError:
        continue
    if uid != os.getuid():
        continue
    if any(n in cmd for n in needles):
        try:
            os.kill(pid, signal.SIGTERM)
            killed.append(pid)
        except OSError:
            pass
time.sleep(3)
for pid in killed:
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
print("REAPED", killed)
PY
}

write_table() {
  "$VENV/bin/python" - <<'PY' "$SUMMARY_JSON" $DATASETS -- $EVAL_STEPS
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
raw = sys.argv[2:]
sep = raw.index("--")
datasets = raw[:sep]
steps = [int(x) for x in raw[sep + 1:]]
rows = []
for dataset in datasets:
    for step in steps:
        p = out.parent / f"{dataset}-avg12-checkpoint-{step}-summary.json"
        if not p.exists():
            rows.append({"dataset": dataset, "step": step, "status": "missing"})
            continue
        rec = json.loads(p.read_text())
        rows.append({
            "dataset": dataset,
            "step": step,
            "status": "ok",
            "average_at_n_pct": rec.get("average_at_n_pct"),
            "pass_at_n_pct": rec.get("pass_at_n_pct"),
            "majority_vote_at_n_pct": rec.get("majority_vote_at_n_pct"),
            "format_rate": rec.get("format_rate"),
            "average_at_n": rec.get("average_at_n"),
            "pass_at_n": rec.get("pass_at_n"),
            "majority_vote_at_n": rec.get("majority_vote_at_n"),
            "total_solutions": rec.get("total_solutions"),
            "num_problems": rec.get("num_problems"),
            "enable_thinking": rec.get("enable_thinking"),
            "tensor_parallel_size": rec.get("tensor_parallel_size"),
        })
payload = {"host": "gn17", "topology": "TP8", "evaluator": "evaluate_math.py", "checkpoints": rows}
out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print("EVAL_TABLE " + json.dumps(rows), flush=True)
lines = [
    "| Dataset | Step | Avg@12 | Pass@12 | Majority | Format |",
    "|---|---:|---|---|---|---|",
]
for row in rows:
    if row["status"] != "ok":
        lines.append(f"| {row['dataset']} | {row['step']} | missing | missing | missing | missing |")
        continue
    nprob = row.get("num_problems") or 30
    lines.append(
        "| {ds} | {step} | {avg:.2f}% ({c}/{t}) | {p:.2f}% ({pn}/{n}) | {m:.2f}% ({mn}/{n}) | {f:.2f}% |".format(
            ds=row["dataset"],
            step=row["step"],
            avg=row["average_at_n_pct"],
            c=row["average_at_n"],
            t=row["total_solutions"],
            p=row["pass_at_n_pct"],
            pn=row["pass_at_n"],
            n=nprob,
            m=row["majority_vote_at_n_pct"],
            mn=row["majority_vote_at_n"],
            f=row["format_rate"],
        )
    )
table = "\n".join(lines) + "\n"
(out.parent / "aime25-hmmt25-ckpts-20-40-60-80-100-table.md").write_text(table, encoding="utf-8")
print("EVAL_TABLE_MD\n" + table, flush=True)
PY
}

resolve_ckpt() {
  local step="$1"
  local step4
  step4="$(printf '%04d' "$step")"
  local ckpt="$LOCAL_RUN/checkpoint-${step4}"
  if [[ -f "$ckpt/model.safetensors" ]]; then
    printf '%s' "$ckpt"
    return 0
  fi
  ckpt="$LOCAL_RUN/checkpoint-${step}"
  if [[ -f "$ckpt/model.safetensors" ]]; then
    printf '%s' "$ckpt"
    return 0
  fi
  if [[ "$step" == "100" && -f "$LOCAL_RUN/final-model/model.safetensors" ]]; then
    printf '%s' "$LOCAL_RUN/final-model"
    return 0
  fi
  return 1
}

echo "EVAL_CHAIN_START host=$(hostname -s) gpus=$CUDA_VISIBLE_DEVICES datasets='$DATASETS' steps='$EVAL_STEPS' topology=TP8 at=$(date -Is)"
echo "EVAL_CONTRACT evaluator=evaluate_math.py thinking=true val_n=12 temperature=1.0 top_p=0.95 top_k=-1 min_p=0 presence_penalty=0 max_new_tokens=38912 tensor_parallel_size=$NPROC gpu_memory_utilization=$GPU_MEM_UTIL"
echo "PYTHON $VENV/bin/python"
echo "EVALUATOR $SRC/eval/evaluate_math.py"
echo "LOCAL_RUN $LOCAL_RUN"
echo "PERSIST_RUN $PERSIST_RUN"
"$VENV/bin/python" -c "import vllm, torch; print('VLLM', vllm.__version__, 'TORCH', torch.__version__, 'CUDA', torch.cuda.is_available())"
sha256sum "$SRC/eval/evaluate_math.py"

fail=0
cd "$SRC/eval"

for dataset in $DATASETS; do
  for step in $EVAL_STEPS; do
    ckpt="$(resolve_ckpt "$step" || true)"
    summary="$PERSIST_RUN/${dataset}-avg12-checkpoint-${step}-summary.json"
    detailed="$PERSIST_RUN/${dataset}-avg12-checkpoint-${step}.json"
    log="$PERSIST_RUN/eval-${dataset}-checkpoint-${step}.log"

    if [[ -f "$summary" ]]; then
      echo "EVAL_SKIP dataset=$dataset step=$step already have $summary"
      write_table
      continue
    fi
    if [[ -z "${ckpt:-}" || ! -f "$ckpt/model.safetensors" ]]; then
      echo "EVAL_MISSING dataset=$dataset step=$step ckpt=${ckpt:-unresolved}"
      fail=1
      write_table
      echo "EVAL_CHAIN_EXIT rc=1 phase=missing dataset=$dataset step=$step at=$(date -Is)"
      exit 1
    fi

    attempt=1
    step_ok=0
    while (( attempt <= MAX_RETRIES )); do
      reap_eval_workers
      wait_gpus_free || true
      echo "EVAL_START dataset=$dataset step=$step attempt=$attempt/$MAX_RETRIES ckpt=$ckpt util=$GPU_MEM_UTIL tp=$NPROC at=$(date -Is)"
      "$VENV/bin/python" evaluate_math.py \
        --base_model "$ckpt" \
        --dataset "$dataset" \
        --val_n 12 \
        --temperature 1.0 \
        --top_p 0.95 \
        --top_k -1 \
        --max_new_tokens 38912 \
        --gpu_memory_utilization "$GPU_MEM_UTIL" \
        --tensor_parallel_size "$NPROC" \
        --output_file "$detailed" \
        >"$log" 2>&1
      rc=$?
      echo "EVAL_EXIT dataset=$dataset step=$step attempt=$attempt rc=$rc at=$(date -Is)"
      if [[ $rc -eq 0 && -f "$summary" ]]; then
        "$VENV/bin/python" - "$summary" "$dataset" "$step" <<'PY'
import json, sys
rec = json.loads(open(sys.argv[1], encoding="utf-8").read())
dataset, step = sys.argv[2], sys.argv[3]
print(
    "EVAL_OK dataset={ds} step={step} Avg@12={avg:.2f}% ({c}/{t}) Pass@12={p}/{n} Maj@12={m}/{n} format={f:.2f}%".format(
        ds=dataset,
        step=step,
        avg=rec["average_at_n_pct"],
        c=rec["average_at_n"],
        t=rec["total_solutions"],
        p=rec["pass_at_n"],
        n=rec["num_problems"],
        m=rec["majority_vote_at_n"],
        f=rec["format_rate"],
    ),
    flush=True,
)
PY
        step_ok=1
        break
      fi
      echo "EVAL_RETRY dataset=$dataset step=$step attempt=$attempt rc=$rc summary_exists=$([[ -f $summary ]] && echo yes || echo no)"
      if [[ -f "$log" ]]; then
        cp -a "$log" "${log}.attempt${attempt}" || true
        echo "EVAL_LOG_TAIL dataset=$dataset step=$step"; tail -n 40 "$log" || true
      fi
      reap_eval_workers
      sleep 10
      attempt=$((attempt + 1))
    done

    if [[ "$step_ok" != "1" ]]; then
      echo "EVAL_FAILED dataset=$dataset step=$step after $MAX_RETRIES attempts"
      fail=1
      write_table
      echo "EVAL_CHAIN_EXIT rc=1 phase=eval dataset=$dataset step=$step at=$(date -Is)"
      exit 1
    fi
    write_table
  done
done

write_table
if [[ "$fail" != "0" ]]; then
  echo "EVAL_CHAIN_EXIT rc=1 at=$(date -Is)"
  exit 1
fi
echo "EVAL_CHAIN_EXIT rc=0 at=$(date -Is)"
exit 0
