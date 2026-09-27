# OPSD-ES

On-policy self-distillation trained with EggRoll, plus the vanilla EggRoll baseline.
Both evaluate with the same protocol: thinking on, temperature 1.0, top-p 0.95, top-k disabled, 38912 new tokens, 12 samples (Avg@12).

The launchers expect 8 GPUs. Set `CUDA_VISIBLE_DEVICES` to choose them.

| Model | OPSD | EggRoll baseline |
|---|---|---|
| Qwen3-1.7B | `opsd_es/train.sh` | `eggroll_baseline/train.sh` |
| Qwen3-4B | `opsd_es/train_qwen3_4b.sh` | `eggroll_baseline/train_qwen3_4b.sh` |

Qwen3-1.7B OPSD is the direct shared-rollout trainer. Qwen3-4B OPSD is the Hugging Face `OPSDTrainer` plus EggRoll (`opsd_train.py`).

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The scripts look for `.venv` in the repository root. Point `VENV` at another environment if you already have one. `EVAL_VENV` defaults to `VENV`.

`MODEL` overrides the checkpoint. The 1.7B launchers default to `Qwen/Qwen3-1.7B`. The 4B launchers default to `Qwen/Qwen3-4B`. Runs are written under that method's `runs/` directory (`outputs/` for Qwen3-4B OPSD). Override the output with `LOCAL_RUN` or `OUT`.

On a machine without GPU peer-to-peer, prefix the launcher with `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1`. The scripts leave those unset otherwise.

## Qwen3-1.7B OPSD

Directory: `opsd_es/`. Launcher: `train.sh`.

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

## Qwen3-1.7B EggRoll baseline

Directory: `eggroll_baseline/`. Launcher: `train.sh`.

Vanilla EggRoll with boxed-correctness fitness, no teacher, thinking off during training, completion length 2048, 128 antithetic pairs (256 members).

```bash
cd eggroll_baseline
bash train.sh
```

This trains for 100 steps and then evaluates **AIME24**, **AIME25**, and **HMMT25** at steps 20, 40, 60, 80, and 100.

## Qwen3-4B

EggRoll stays the direct trainer, with one prompt per generate wave so a 4B member batch fits on 40GB. Resume with `RESUME_FROM=/path/to/checkpoint START_STEP=40`.

OPSD uses `opsd_train.py`: population 256, no token clip. The frozen teacher stays on CPU except while its log-probs are cached, and each forward scores one member. Resume with `RESUME_FROM_CHECKPOINT=/path/to/checkpoint`. That run needs `trl==0.26.0`.

```bash
cd eggroll_baseline
bash train_qwen3_4b.sh

cd opsd_es
bash train_qwen3_4b.sh
```

Each 4B launcher trains for 100 steps and then evaluates **AIME24**, **AIME25**, and **HMMT25** at steps 20, 40, 60, 80, and 100.

## Dependencies

- PyTorch
- Transformers
- `datasets`
- vLLM
- `math_verify`
- `tqdm`
- `accelerate`, `trl==0.26.0`, `peft`, `wandb`, and `safetensors` for the Qwen3-4B OPSD launcher

## License

MIT. See [LICENSE](LICENSE).
