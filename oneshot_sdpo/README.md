# One-Shot-RLVR + SDPO

Train Qwen2.5-Math-1.5B with **SDPO** (Self-Distilled Policy Optimization) on a
single math problem — the One-Shot-RLVR data regime.

## Experiment

- **Model**: Qwen2.5-Math-1.5B (no `<think>` tags; non-thinking model)
- **Training problem π₁**: wind-pressure word problem (see `data/pi1_r128.parquet`)
- **Training set**: 128 identical copies of π₁ per step (committed to repo)
- **Algorithm**: SDPO — successful rollouts become token-level demonstrations for
  failed ones; EMA teacher; IS-corrected GRPO advantage estimator
- **Reward**: binary (1.0 correct / 0.0 incorrect), boxed-answer extraction with
  SymPy fallback for symbolic equivalence
- **Evaluation**: MATH-500 (greedy decoding, same reward logic)
- **Hardware**: 4 × A100, up to 16 hours

## Repository Structure

```
oneshot_sdpo/
├── data/
│   ├── pi1_r128.parquet           # 128 copies of π₁ — training set (committed)
│   └── math500.parquet            # MATH-500 — training validation + standalone eval (committed)
├── reward/
│   └── math_reward.py             # custom reward function (deepscaler-style + SymPy fallback)
├── scripts/
│   ├── setup_hpc.sh               # one-time login-node env setup
│   ├── run_local_test.sh          # pre-flight smoke test
│   └── train_oneshot_sdpo.slurm   # SLURM training job (4×A100, 16 h)
├── eval/
│   ├── eval_math500.py            # standalone MATH-500 evaluation
│   └── eval_math500.slurm         # SLURM eval job (1×A100, 2 h)
├── requirements.txt
├── .gitignore
└── README.md
```

Both parquet files are committed directly to the repo (sourced from One-Shot-RLVR,
pinned commits). No data download step is needed.

## HPC Setup (run once on login node)

```bash
scp -r oneshot_sdpo/ <user>@<hpc>:/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/
ssh <hpc>
bash /home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/oneshot_sdpo/scripts/setup_hpc.sh
```

`setup_hpc.sh` installs SDPO (verl + all core deps) and sympy via pip, creates the
output directory tree, and verifies key imports. Run it **on the login node** where
internet access is available.

## Training

```bash
conda activate sdpo
cd /home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/oneshot_sdpo

# 1. Pre-flight smoke test
bash scripts/run_local_test.sh

# 2. Submit training job
sbatch scripts/train_oneshot_sdpo.slurm
```

## Evaluation

```bash
# After training completes, evaluate the final checkpoint:
sbatch --export=CKPT=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/output/checkpoints/global_step_500 \
    eval/eval_math500.slurm

# Or run interactively:
conda activate sdpo
python eval/eval_math500.py \
    --checkpoint /home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/output/checkpoints/global_step_500 \
    --eval_data  data/math500.parquet \
    --step       500
```

## Output Layout

```
/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/output/
├── checkpoints/                    # trainer.default_local_dir
│   └── global_step_{N}/actor/      # SDPO checkpoint structure
├── tensorboard/                    # TENSORBOARD_DIR env var
│   └── oneshot_sdpo/oneshot_sdpo_qwen25math_1.5b_pi1/
├── logs/                           # SLURM stdout/stderr
│   ├── train_{JOBID}.out/err
│   └── eval_math500_{JOBID}.out/err
├── rollouts/                       # trainer.validation_data_dir
│   └── {N}.jsonl                   # one file per val step (SDPO native naming)
└── eval_results/                   # eval_math500.py output
    └── eval_step_{N}.jsonl
```

### Rollout JSONL fields (SDPO native)

| Field | Source |
|-------|--------|
| `input` | prompt text |
| `output` | model response |
| `gts` | ground truth |
| `score` | reward scalar |
| `step` | training step |
| `extracted_answer` | from `compute_score` dict return |
| `is_correct` | from `compute_score` dict return |

### Eval JSONL fields (`eval_math500.py`)

| Field | Source |
|-------|--------|
| `step` | training step |
| `prompt` | input text |
| `response` | model output |
| `extracted_answer` | `extract_answer(response)` |
| `ground_truth` | from math500.parquet |
| `reward` | 1.0 / 0.0 |
| `is_correct` | bool |

## Monitoring

```bash
# TensorBoard
tensorboard --logdir /home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/output/tensorboard

# Follow SLURM log
tail -f /home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/output/logs/train_<JOBID>.out
```

## Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Hydra config base | `ppo_trainer` | `sdpo.yaml` inherits `user.yaml` which requires `${oc.env:TASK}` and `${oc.env:EXPERIMENT}` env vars — CSCS-specific, crashes on other HPC |
| Data files | Committed parquets | Sourced directly from One-Shot-RLVR; no download step needed on HPC |
| `use_think` | `False` | Qwen2.5-Math-1.5B has no `<think>` tags |
| Reward grading | deepscaler-style (mathd + sympy) | SymPy catches `64/5 == 12.8` |
| `data_source` | from repo-shipped parquet (`deepscaler`) | Bypassed by `custom_reward_function.path`; value is ignored at runtime |
| Rollout JSONL names | SDPO native (`input`/`output`/`gts`/`score`) | Avoids vendoring `ray_trainer.py` |
| Local `verl/` subtree | None | SDPO is pip-installable |
| WandB | `WANDB_MODE=disabled` | TensorBoard only |
