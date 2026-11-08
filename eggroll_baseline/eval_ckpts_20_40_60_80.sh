#!/usr/bin/env bash
# Sequential AIME24 Avg@12 of an EggRoll baseline run:
#   checkpoint-0025, 0050, 0075, 0100
# Same eval protocol as the original pipeline / baseline (thinking ON, val_n=12).
# Skips a step if its summary already exists. Retries a failed step up to 3 times.
set -uo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SRC/.." && pwd)"
VENV="${VENV:-$REPO/.venv}"
LOCAL_RUN="${LOCAL_RUN:-$SRC/runs/eggroll-baseline-tmoff2048}"
PERSIST_RUN="${PERSIST_RUN:-$LOCAL_RUN}"
EVAL_STEPS="${EVAL_STEPS:-25 50 75 100}"
MAX_RETRIES="${MAX_RETRIES:-3}"
NPROC="${NPROC:-8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.75}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
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
needles = ("evaluate_math.py", "VllmWorker", "EngineCore")
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
  "$VENV/bin/python" - <<'PY' "$PERSIST_RUN/eval-ckpts-25-50-75-100-summary.json" $EVAL_STEPS
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
steps = [int(x) for x in sys.argv[2:]]
rows = []
for step in steps:
    p = out.parent / f"aime24-avg12-checkpoint-{step}-summary.json"
    if not p.exists():
        rows.append({"step": step, "status": "missing"})
        continue
    rec = json.loads(p.read_text())
    rows.append({
        "step": step,
        "status": "ok",
        "average_at_n_pct": rec.get("average_at_n_pct"),
        "pass_at_n_pct": rec.get("pass_at_n_pct"),
        "majority_vote_at_n_pct": rec.get("majority_vote_at_n_pct"),
        "format_rate": rec.get("format_rate"),
        "average_at_n": rec.get("average_at_n"),
        "pass_at_n": rec.get("pass_at_n"),
        "enable_thinking": rec.get("enable_thinking"),
    })
out.write_text(json.dumps({"checkpoints": rows}, indent=2), encoding="utf-8")
print("EVAL_TABLE " + json.dumps(rows), flush=True)
PY
}

echo "EVAL_CHAIN_START host=$(hostname -s) gpus=$CUDA_VISIBLE_DEVICES steps='$EVAL_STEPS' at=$(date -Is)"
fail=0
cd "$SRC/eval"

for step in $EVAL_STEPS; do
  step4="$(printf '%04d' "$step")"
  ckpt="$LOCAL_RUN/checkpoint-${step4}"
  # persist copies used the 4-digit names; scratch does too
  if [[ ! -f "$ckpt/model.safetensors" ]]; then
    alt="$LOCAL_RUN/checkpoint-${step}"
    if [[ -f "$alt/model.safetensors" ]]; then
      ckpt="$alt"
    fi
  fi
  summary="$PERSIST_RUN/aime24-avg12-checkpoint-${step}-summary.json"
  detailed="$PERSIST_RUN/aime24-avg12-checkpoint-${step}.json"
  log="$PERSIST_RUN/eval-checkpoint-${step}.log"

  if [[ -f "$summary" ]]; then
    echo "EVAL_SKIP step=$step already have $summary"
    continue
  fi
  if [[ ! -f "$ckpt/model.safetensors" ]]; then
    echo "EVAL_MISSING step=$step dir=$ckpt"
    fail=1
    continue
  fi

  attempt=1
  step_ok=0
  while (( attempt <= MAX_RETRIES )); do
    reap_eval_workers
    wait_gpus_free || true
    echo "EVAL_START step=$step attempt=$attempt/$MAX_RETRIES ckpt=$ckpt util=$GPU_MEM_UTIL at=$(date -Is)"
    "$VENV/bin/python" evaluate_math.py \
      --base_model "$ckpt" \
      --dataset aime24 \
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
    echo "EVAL_EXIT step=$step attempt=$attempt rc=$rc at=$(date -Is)"
    if [[ $rc -eq 0 && -f "$summary" ]]; then
      step_ok=1
      break
    fi
    echo "EVAL_RETRY step=$step attempt=$attempt rc=$rc summary_exists=$([[ -f $summary ]] && echo yes || echo no)"
    if [[ -f "$log" ]]; then
      cp -a "$log" "${log}.attempt${attempt}" || true
    fi
    reap_eval_workers
    sleep 10
    attempt=$((attempt + 1))
  done

  if [[ "$step_ok" != "1" ]]; then
    echo "EVAL_FAILED step=$step after $MAX_RETRIES attempts"
    fail=1
    write_table
    echo "EVAL_CHAIN_EXIT rc=1 phase=eval step=$step at=$(date -Is)"
    exit 1
  fi
  write_table
done

write_table
if [[ "$fail" != "0" ]]; then
  echo "EVAL_CHAIN_EXIT rc=1 at=$(date -Is)"
  exit 1
fi
echo "EVAL_CHAIN_EXIT rc=0 at=$(date -Is)"
exit 0
