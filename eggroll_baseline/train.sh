#!/usr/bin/env bash
# Vanilla EGGROLL AIME baseline: 512 antithetic pairs (1024 members) on 8 GPUs.
# Each perturbation does a full boxed-correctness rollout on 32 OpenThoughts prompts.
set -uo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SRC/.." && pwd)"
VENV="${VENV:-$REPO/.venv}"
EVAL_VENV="${EVAL_VENV:-$VENV}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
RUN_NAME="${RUN_NAME:-eggroll-baseline-tmoff2048-$(date +%Y%m%d-%H%M%S)}"
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
} > "$PERSIST_RUN/allocation.txt"
cp -a "$SRC/eggroll_opsd_train_population_scaled.py" "$PERSIST_RUN/"
cp -a "$SRC/eggroll_opsd_shared_rollout_population_scaled_train.py" "$PERSIST_RUN/"
cp -a "$0" "$PERSIST_RUN/train.sh"

echo "PIPELINE_START host=$HOST gpus=$CUDA_VISIBLE_DEVICES run=$RUN_NAME at=$(date -Is)"

cd "$SRC"
"$VENV/bin/torchrun" --standalone --nproc_per_node=8 \
  "$SRC/eggroll_opsd_shared_rollout_population_scaled_train.py" \
  --model "$MODEL" \
  --output-dir "$LOCAL_RUN" \
  --steps 100 \
  --prompt-groups 32 \
  --pairs 512 \
  --score-microbatch-size 4 \
  --tensorized-prompt-microbatch-size 4 \
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
  --checkpoint-every 25
train_rc=$?
echo "TRAIN_EXIT rc=$train_rc at=$(date -Is)"
rsync -a "$LOCAL_RUN/" "$PERSIST_RUN/"
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
