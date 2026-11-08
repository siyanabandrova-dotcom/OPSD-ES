#!/usr/bin/env bash
# Qwen3-4B vanilla EGGROLL on 16 GPUs (2 machines x 8).
# Same protocol as train.sh, with one prompt per generate wave so a 4B
# member batch fits. Population is 512 antithetic pairs (1024 members).
# Start the same LOCAL_RUN on every node. Rank 0 evaluates.
set -uo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SRC/.." && pwd)"
VENV="${VENV:-$REPO/.venv}"
EVAL_VENV="${EVAL_VENV:-$VENV}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
NNODES="${NNODES:-2}"
NPROC="${NPROC:-8}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT="${MASTER_PORT:-29511}"
if [[ "$NNODES" -gt 1 && -z "${LOCAL_RUN:-}" ]]; then
  echo "Set the same LOCAL_RUN on every node. Example: LOCAL_RUN=$SRC/runs/eggroll-4b-pop1024" >&2
  exit 1
fi
RUN_NAME="${RUN_NAME:-eggroll-baseline-qwen3-4b-$(date +%Y%m%d-%H%M%S)}"
LOCAL_RUN="${LOCAL_RUN:-$SRC/runs/${RUN_NAME}}"
PERSIST_RUN="${PERSIST_RUN:-$LOCAL_RUN}"
HOST="$(hostname -s)"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-10800}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$LOCAL_RUN" "$PERSIST_RUN"
date -Is > "$PERSIST_RUN/pipeline-start.txt"
{
  echo "HOST=$HOST"
  echo "SRC=$SRC"
  echo "VENV=$VENV"
  echo "EVAL_VENV=$EVAL_VENV"
  echo "MODEL=$MODEL"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "LOCAL_RUN=$LOCAL_RUN"
  echo "PERSIST_RUN=$PERSIST_RUN"
  echo "NNODES=$NNODES"
  echo "NODE_RANK=$NODE_RANK"
  echo "MASTER_ADDR=$MASTER_ADDR"
  echo "PAIRS=512"
  echo "PROMPT_MICROBATCH=1"
} > "$PERSIST_RUN/allocation.txt"
cp -a "$SRC/eggroll_opsd_train_population_scaled.py" "$PERSIST_RUN/"
cp -a "$SRC/eggroll_opsd_shared_rollout_population_scaled_train.py" "$PERSIST_RUN/"
cp -a "$0" "$PERSIST_RUN/train_qwen3_4b.sh"

resume_args=()
if [[ -n "${RESUME_FROM:-}" ]]; then
  if [[ -z "${START_STEP:-}" ]]; then
    echo "START_STEP is required when RESUME_FROM is set" >&2
    exit 1
  fi
  resume_args=(--resume-from "$RESUME_FROM" --start-step "$START_STEP")
fi

if [[ "$NNODES" -gt 1 && -z "$MASTER_ADDR" ]]; then
  echo "Multi-node needs MASTER_ADDR and the same LOCAL_RUN on every node. Set NODE_RANK=0 on the first node and 1 on the second." >&2
  exit 1
fi
RDZV_ID="${RDZV_ID:-$(basename "$LOCAL_RUN")}"

echo "PIPELINE_START host=$HOST gpus=$CUDA_VISIBLE_DEVICES run=$RUN_NAME nodes=$NNODES rank=$NODE_RANK at=$(date -Is)"

cd "$SRC"
if [[ "$NNODES" -eq 1 ]]; then
  torch_launch=("$VENV/bin/torchrun" --standalone --nproc_per_node="$NPROC")
else
  torch_launch=(
    "$VENV/bin/torchrun"
    --nnodes="$NNODES"
    --nproc_per_node="$NPROC"
    --node_rank="$NODE_RANK"
    --rdzv_id="$RDZV_ID"
    --rdzv_backend=c10d
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}"
    --rdzv_conf=timeout=600
    --max_restarts=0
  )
fi
"${torch_launch[@]}" \
  "$SRC/eggroll_opsd_shared_rollout_population_scaled_train.py" \
  --model "$MODEL" \
  --output-dir "$LOCAL_RUN" \
  --steps 100 \
  --prompt-groups 32 \
  --pairs 512 \
  --score-microbatch-size 4 \
  --tensorized-prompt-microbatch-size 1 \
  --tensorized-members-per-wave-per-gpu 8 \
  --sigma 0.001 \
  --learning-rate 0.2 \
  --noise-rank 1 \
  --seed 42 \
  --max-completion-length 2048 \
  --max-prompt-length 18976 \
  --temperature 1.1 \
  --top-p 0.95 \
  --top-k 20 \
  --kl-temperature 1.1 \
  --token-clip 0.05 \
  --checkpoint-every 25 \
  "${resume_args[@]}"
train_rc=$?
echo "TRAIN_EXIT rc=$train_rc at=$(date -Is)"
rsync -a "$LOCAL_RUN/" "$PERSIST_RUN/"
if [[ "$NODE_RANK" != "0" ]]; then
  echo "PIPELINE_EXIT rc=$train_rc rank=$NODE_RANK at=$(date -Is)"
  exit "$train_rc"
fi
if [[ $train_rc -ne 0 ]]; then
  echo "PIPELINE_EXIT rc=$train_rc at=$(date -Is)"
  exit "$train_rc"
fi

LOCAL_RUN="$LOCAL_RUN" PERSIST_RUN="$PERSIST_RUN" VENV="$EVAL_VENV" \
  EVAL_STEPS="25 50 75 100" \
  bash "$SRC/eval_ckpts_20_40_60_80.sh"
aime_rc=$?
echo "AIME24_EVAL_EXIT rc=$aime_rc at=$(date -Is)"
if [[ $aime_rc -ne 0 ]]; then
  echo "PIPELINE_EXIT rc=$aime_rc at=$(date -Is)"
  exit "$aime_rc"
fi

LOCAL_RUN="$LOCAL_RUN" PERSIST_RUN="$PERSIST_RUN" VENV="$EVAL_VENV" \
  EVAL_STEPS="25 50 75 100" DATASETS="aime25 hmmt25" \
  bash "$SRC/run_eval_aime25_hmmt25_ckpts.sh"
eval_rc=$?
echo "AIME25_HMMT25_EVAL_EXIT rc=$eval_rc at=$(date -Is)"
if [[ $eval_rc -ne 0 ]]; then
  echo "PIPELINE_EXIT rc=$eval_rc at=$(date -Is)"
  exit "$eval_rc"
fi
"$EVAL_VENV/bin/python" "$REPO/report_best_avg12.py" "$PERSIST_RUN" aime24 aime25 hmmt25 -- 25 50 75 100
eval_rc=$?
echo "PIPELINE_EXIT rc=$eval_rc at=$(date -Is)"
exit "$eval_rc"
