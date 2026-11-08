#!/usr/bin/env bash
# OPSD-ES direct EGGROLL on 8 GPUs:
#   512 antithetic pairs (1024 members), token clip 0.05,
#   population-wide score std,
#   checkpoint every 25 steps, then AIME24, AIME25 and HMMT25 Avg@12
#   on steps 25, 50, 75 and final-model as step 100.
# Eval matches the existing AIME25/HMMT25 tables:
#   evaluate_math.py, thinking on, val_n=12, temp 1.0, top_p 0.95,
#   top_k -1, max_new_tokens 38912, TP=8, gpu_memory_utilization 0.75.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
SRC="${SRC:-$ROOT}"
VENV="${VENV:-$REPO/.venv}"
EVAL_VENV="${EVAL_VENV:-$VENV}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
RUN_NAME="${RUN_NAME:-opsd-es-pairs512-$(date +%Y%m%d-%H%M%S)}"
LOCAL_RUN="${LOCAL_RUN:-$ROOT/runs/${RUN_NAME}}"
PERSIST_RUN="${PERSIST_RUN:-$LOCAL_RUN}"
HOST="$(hostname -s)"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
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
  echo "PAIRS=512"
  echo "TOKEN_CLIP=0.05"
  echo "SCORE_STD=population_wide"
  echo "CHECKPOINT_EVERY=25"
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
  --tensorized-prompt-microbatch-size 2 \
  --tensorized-members-per-wave-per-gpu 8 \
  --sigma 0.001 \
  --learning-rate 0.2 \
  --noise-rank 1 \
  --seed 42 \
  --max-completion-length 1024 \
  --max-prompt-length 18976 \
  --temperature 1.1 \
  --top-p 0.95 \
  --top-k 20 \
  --kl-temperature 1.1 \
  --token-clip 0.05 \
  --checkpoint-every 25
train_rc=$?
echo "TRAIN_EXIT rc=$train_rc at=$(date -Is)"
rsync -a --exclude 'checkpoint-*' --exclude 'final-model' "$LOCAL_RUN/" "$PERSIST_RUN/" || true
if [[ $train_rc -ne 0 ]]; then
  echo "PIPELINE_EXIT rc=$train_rc at=$(date -Is)"
  exit "$train_rc"
fi

LOCAL_RUN="$LOCAL_RUN" PERSIST_RUN="$PERSIST_RUN" VENV="$EVAL_VENV" \
  EVAL_STEPS="25 50 75 100" DATASETS="aime24 aime25 hmmt25" \
  bash "$ROOT/eval_aime25_hmmt25.sh"
eval_rc=$?
echo "EVAL_CHAIN_DONE rc=$eval_rc at=$(date -Is)"
rsync -a --exclude 'checkpoint-*' --exclude 'final-model' "$LOCAL_RUN/" "$PERSIST_RUN/" || true
if [[ $eval_rc -ne 0 ]]; then
  echo "PIPELINE_EXIT rc=$eval_rc at=$(date -Is)"
  exit "$eval_rc"
fi
"$EVAL_VENV/bin/python" "$REPO/report_best_avg12.py" "$PERSIST_RUN" aime24 aime25 hmmt25 -- 25 50 75 100
eval_rc=$?
echo "PIPELINE_EXIT rc=$eval_rc at=$(date -Is)"
exit "$eval_rc"
