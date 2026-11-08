# OPSD-ES

On-policy self-distillation trained with EggRoll, plus the vanilla EggRoll baseline.
Both evaluate with the same protocol: thinking on, temperature 1.0, top-p 0.95, top-k disabled, 38912 new tokens, 12 samples (Avg@12).

The 1.7B launchers expect 8 GPUs. The 4B launchers expect 16 GPUs, two machines of 8. Set `CUDA_VISIBLE_DEVICES` to choose the devices on a machine.

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

`MODEL` overrides the checkpoint. The 1.7B launchers default to `Qwen/Qwen3-1.7B`. The 4B launchers default to `Qwen/Qwen3-4B`. Those ids are the Instruct checkpoints. Runs are written under that method's `runs/` directory (`outputs/` for Qwen3-4B OPSD). Override the output with `LOCAL_RUN` or `OUT`.

Every trained row saves steps 25, 50, 75, and 100. The reported score on a dataset is the best Avg@12 among those four checkpoints. Each dataset is chosen on its own, and a tie keeps the later step. The average is the mean of the three chosen cells. `report_best_avg12.py` writes that table after the evals.

On a machine without GPU peer-to-peer, prefix the launcher with `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1`. The scripts leave those unset otherwise.

## Qwen3-1.7B OPSD

Directory: `opsd_es/`. Launcher: `train.sh`.

512 antithetic pairs (1024 members), token clip 0.05, one score standard deviation for the whole population, checkpoints every 25 steps.

```bash
cd opsd_es
bash train.sh
```

`train.sh` trains for 100 steps and then evaluates **AIME24**, **AIME25**, and **HMMT25** at steps 25, 50, 75, and 100.

## Qwen3-1.7B EggRoll baseline

Directory: `eggroll_baseline/`. Launcher: `train.sh`.

Vanilla EggRoll with boxed-correctness fitness, no teacher, thinking off during training, completion length 2048, 512 antithetic pairs (1024 members). The starting checkpoint is the same Instruct model as OPSD.

```bash
cd eggroll_baseline
bash train.sh
```

This trains for 100 steps and then evaluates **AIME24**, **AIME25**, and **HMMT25** at steps 25, 50, 75, and 100.

## Qwen3-4B

Both 4B launchers default to two machines. Set `MASTER_ADDR` to the first machine, `NODE_RANK=0` there and `NODE_RANK=1` on the second, and start both with the same output directory. `NNODES=1` runs on one machine. On a machine without GPU peer-to-peer, prefix both with `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1`.

EggRoll stays the direct trainer, with one prompt per generate wave so a 4B member batch fits on 40GB. Population is 512 antithetic pairs. Resume with `RESUME_FROM=/path/to/checkpoint START_STEP=25`.

```bash
cd eggroll_baseline
LOCAL_RUN=/path/to/eggroll-4b MASTER_ADDR=<first-node-ip> NODE_RANK=0 bash train_qwen3_4b.sh
LOCAL_RUN=/path/to/eggroll-4b MASTER_ADDR=<first-node-ip> NODE_RANK=1 bash train_qwen3_4b.sh
```

OPSD uses `opsd_train.py`: population 1024, token clip 0.05, batch 2 per GPU. The frozen teacher stays on CPU except while its log-probs are cached, and each forward scores one member. Resume with `RESUME_FROM_CHECKPOINT=/path/to/checkpoint`. That run needs `trl==0.26.0`.

```bash
cd opsd_es
OUT=/path/to/opsd-4b MASTER_ADDR=<first-node-ip> NODE_RANK=0 bash train_qwen3_4b.sh
OUT=/path/to/opsd-4b MASTER_ADDR=<first-node-ip> NODE_RANK=1 bash train_qwen3_4b.sh
```

Each 4B launcher trains for 100 steps. Rank 0 then evaluates **AIME24**, **AIME25**, and **HMMT25** at steps 25, 50, 75, and 100.

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
