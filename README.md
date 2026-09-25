# OPSD-ES

On-policy self-distillation trained with EggRoll, plus the vanilla EggRoll baseline.
Both methods start from `Qwen/Qwen3-1.7B` and evaluate with the same protocol:
thinking on, temperature 1.0, top-p 0.95, top-k disabled, 38912 new tokens, 12 samples (Avg@12).

The launchers expect 8 GPUs. Set `CUDA_VISIBLE_DEVICES` to choose them.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Each method looks for `.venv` in its own directory. Point `VENV` at another environment if you already have one. `EVAL_VENV` defaults to `VENV`.

Override the model with `MODEL` (default `Qwen/Qwen3-1.7B`). Runs are written under that method's `runs/` directory. Override the output with `LOCAL_RUN`.

## OPSD-ES

Directory: `opsd_es/`.

512 antithetic pairs (1024 members), token clip 0.05, one score standard deviation for the whole population, checkpoints every 20 steps.

```bash
cd opsd_es
bash train.sh
```

`train.sh` trains for 100 steps and then evaluates **AIME25** and **HMMT25** at steps 20, 40, 60, 80, and 100.

**AIME24 is not part of that pipeline.** After training, point `LOCAL_RUN` at the run directory and evaluate it separately:

```bash
cd opsd_es
LOCAL_RUN=runs/<run-name> bash eval_aime24.sh
```

## EggRoll baseline

Directory: `eggroll_baseline/`.

Vanilla EggRoll with boxed-correctness fitness, no teacher, thinking off during training, completion length 2048, 128 antithetic pairs (256 members).

```bash
cd eggroll_baseline
bash train.sh
```

This trains for 100 steps and then evaluates **AIME24**, **AIME25**, and **HMMT25** at steps 20, 40, 60, 80, and 100.

## Dependencies

- PyTorch
- Transformers
- `datasets`
- vLLM
- `math_verify`
- `tqdm`
