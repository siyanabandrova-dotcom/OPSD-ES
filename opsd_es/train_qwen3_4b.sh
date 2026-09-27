#!/usr/bin/env bash
# Qwen3-4B OPSD EggRoll on 40GB GPUs.
# Same entry point as the run that produced the Qwen3-4B Avg@12 tables:
# Hugging Face OPSDTrainer + EggRoll (opsd_train.py), not the direct
# shared-rollout trainer used for Qwen3-1.7B.
# Population 256 = 128 antithetic pairs, 32 shared prompts, noclip.
# The frozen teacher stays on CPU except while its log-probs are cached,
# and each forward scores one member.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
VENV="${VENV:-$REPO/.venv}"
EVAL_VENV="${EVAL_VENV:-$VENV}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
OUT="${OUT:-$ROOT/outputs}"
RUN_CONFIG="${RUN_CONFIG:-qwen34b_eggroll_noclip_pop256}"
NPROC="${NPROC:-8}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-4}"
PORT="${MAIN_PROCESS_PORT:-12954}"
SAVE_STEPS="${SAVE_STEPS:-20}"
ES_WAVE_SIZE="${ES_WAVE_SIZE:-4}"
EVAL_STEPS="${EVAL_STEPS:-20 40 60 80 100}"
DATASETS="${DATASETS:-aime24 aime25 hmmt25}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.75}"
MAX_RETRIES="${MAX_RETRIES:-3}"
HOST="$(hostname -s)"

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
# FP32 masters stay on CPU, and the teacher is off the GPU during member scoring,
# so a prompt batch of 4 fits. Members stay one-per-forward (a stacked wave does not).
export ES_TEACHER_MICROBATCH="${ES_TEACHER_MICROBATCH:-4}"
export ES_PROMPT_MICROBATCH="${ES_PROMPT_MICROBATCH:-4}"

PYTHON_BIN="${PYTHON_BIN:-$VENV/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-$VENV/bin/accelerate}"
if [[ ! -x "$ACCELERATE_BIN" ]]; then
  echo "Missing $ACCELERATE_BIN. Create $REPO/.venv or set ACCELERATE_BIN." >&2
  exit 1
fi

EXP="$OUT/$RUN_CONFIG"
if [[ "$OUT" == */"$RUN_CONFIG" ]]; then
  EXP="$OUT"
fi
mkdir -p "$EXP"
date -Is > "$EXP/pipeline-start.txt"
{
  echo "HOST=$HOST"
  echo "ROOT=$ROOT"
  echo "VENV=$VENV"
  echo "EVAL_VENV=$EVAL_VENV"
  echo "MODEL=$MODEL"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "OUT=$OUT"
  echo "RUN_CONFIG=$RUN_CONFIG"
  echo "EXP=$EXP"
  echo "ES_POPULATION=256"
  echo "ES_PROMPTS_PER_STEP=32"
  echo "JSD_TOKEN_CLIP=0"
  echo "ES_TEACHER_MICROBATCH=$ES_TEACHER_MICROBATCH"
  echo "ES_PROMPT_MICROBATCH=$ES_PROMPT_MICROBATCH"
  echo "ES_WAVE_SIZE=$ES_WAVE_SIZE"
} > "$EXP/allocation.txt"
cp -a "$0" "$EXP/train_qwen3_4b.sh"

ATTN="${ATTN:-}"
if [[ -z "$ATTN" ]]; then
  if "$PYTHON_BIN" -c "import flash_attn" 2>/dev/null; then
    ATTN=flash_attention_2
  else
    ATTN=sdpa
    echo "flash_attn not in train venv; using sdpa"
  fi
fi

RESUME="${RESUME_FROM_CHECKPOINT:-}"
RESUME_ARGS=()
if [[ -n "$RESUME" ]]; then
  RESUME_ARGS=(--resume_from_checkpoint "$RESUME")
  echo "RESUME_FROM_CHECKPOINT=$RESUME"
fi

echo "PIPELINE_START host=$HOST gpus=$CUDA_VISIBLE_DEVICES run=$RUN_CONFIG at=$(date -Is)"

cd "$ROOT"
"$ACCELERATE_BIN" launch \
    --config_file accelerate_es.yaml \
    --num_processes "$NPROC" \
    --main_process_port "$PORT" \
    opsd_train.py \
    --model_name_or_path "$MODEL" \
    --learning_rate 0.2 \
    --per_device_train_batch_size "$PER_DEVICE_BATCH" \
    --gradient_accumulation_steps 1 \
    --ddp_find_unused_parameters True \
    --output_dir "$OUT" \
    --run_config "$RUN_CONFIG" \
    --max_steps 100 \
    --max_completion_length 1024 \
    --save_steps "$SAVE_STEPS" \
    --logging_steps 1 \
    --attn_implementation "$ATTN" \
    --torch_dtype bfloat16 \
    --max_length 20000 \
    --beta 0 \
    --temperature 1.1 \
    --top_p 1.0 \
    --top_k 0 \
    --lmbda 1 \
    --jsd_token_clip 0 \
    --use_es \
    --es_population 256 \
    --es_prompts_per_step 32 \
    --es_wave_size "$ES_WAVE_SIZE" \
    --es_sigma 1e-3 \
    --es_rank 1 \
    --es_lr_scale 0.2 \
    --wandb_project OPSD \
    "${RESUME_ARGS[@]}"
train_rc=$?
echo "TRAIN_EXIT rc=$train_rc at=$(date -Is)"
if [[ $train_rc -ne 0 ]]; then
  echo "PIPELINE_EXIT rc=$train_rc at=$(date -Is)"
  exit "$train_rc"
fi

shopt -s nullglob
for ckpt in "$EXP"/checkpoint-*; do
  [[ -d "$ckpt" ]] || continue
  [[ "$(basename "$ckpt")" == *_vllm ]] && continue
  "$PYTHON_BIN" "$ROOT/fix_eggroll_checkpoint_for_vllm.py" "$ckpt" || {
    echo "STRIP_FAILED ckpt=$ckpt" >&2
    echo "PIPELINE_EXIT rc=1 at=$(date -Is)"
    exit 1
  }
done

for step in $EVAL_STEPS; do
  src="$EXP/checkpoint-${step}_vllm"
  if [[ ! -f "$src/config.json" && -f "$EXP/checkpoint-${step}/config.json" ]]; then
    src="$EXP/checkpoint-${step}"
  fi
  for dataset in $DATASETS; do
    summary="$EXP/${dataset}-avg12-checkpoint-${step}-summary.json"
    detailed="$EXP/${dataset}-avg12-checkpoint-${step}.json"
    log="$EXP/eval-${dataset}-checkpoint-${step}.log"
    if [[ -f "$summary" ]]; then
      echo "EVAL_SKIP dataset=$dataset step=$step"
      continue
    fi
    if [[ ! -f "$src/config.json" ]]; then
      echo "EVAL_MISSING dataset=$dataset step=$step dir=$src"
      echo "PIPELINE_EXIT rc=1 at=$(date -Is)"
      exit 1
    fi
    attempt=1
    step_ok=0
    while (( attempt <= MAX_RETRIES )); do
      echo "EVAL_START dataset=$dataset step=$step attempt=$attempt ckpt=$src at=$(date -Is)"
      (
        cd "$ROOT/eval"
        "$EVAL_VENV/bin/python" evaluate_math.py \
          --base_model "$src" \
          --dataset "$dataset" \
          --val_n 12 \
          --temperature 1.0 \
          --top_p 0.95 \
          --top_k -1 \
          --min_p 0 \
          --presence_penalty 0 \
          --max_new_tokens 38912 \
          --gpu_memory_utilization "$GPU_MEM_UTIL" \
          --tensor_parallel_size "$NPROC" \
          --output_file "$detailed"
      ) >"$log" 2>&1
      rc=$?
      echo "EVAL_EXIT dataset=$dataset step=$step attempt=$attempt rc=$rc at=$(date -Is)"
      if [[ $rc -eq 0 && -f "$summary" ]]; then
        step_ok=1
        break
      fi
      echo "EVAL_RETRY dataset=$dataset step=$step attempt=$attempt"
      tail -n 40 "$log" || true
      sleep 10
      attempt=$((attempt + 1))
    done
    if [[ "$step_ok" != "1" ]]; then
      echo "EVAL_FAILED dataset=$dataset step=$step"
      echo "PIPELINE_EXIT rc=1 at=$(date -Is)"
      exit 1
    fi
  done
done

echo "PIPELINE_EXIT rc=0 at=$(date -Is)"
exit 0
