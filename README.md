# One-Shot-RLVR + SDPO

Train Qwen2.5-Math-1.5B with **SDPO** (Self-Distilled Policy Optimization) on a
single math problem — the One-Shot-RLVR data regime.

## Experiment

- **Model**: Qwen2.5-Math-1.5B (non-thinking model, no `<think>` tags)
- **Training problem π₁**: wind-pressure word problem (see `data/pi1_r128.parquet`)
- **Training set**: 128 identical copies of π₁ per step (committed to repo)
- **Algorithm**: SDPO
  - Successful rollouts become token-level demonstrations for failed ones
  - EMA teacher (update rate 0.05) provides soft targets
  - Loss = JSD(student, teacher) weighted by token-level IS correction
  - KL penalty vs frozen ref model (`kl_loss_coef=0.001`)
  - GRPO advantages computed but unused — success/failure determined by reward score
- **Reward**: binary (1.0 correct / 0.0 incorrect), `\boxed{}` extraction + SymPy fallback
- **Evaluation**: MATH-500 (greedy decoding, same reward logic)
- **Hardware**: 4 × A100, up to 16 hours per job (auto-resume supported)

## Repository Structure

```
oneshotrlvrSDPO/              ← PROJECT_ROOT on HPC
├── data/
│   ├── pi1_r128.parquet      # 128 copies of π₁ — training set (committed)
│   └── math500.parquet       # MATH-500 — validation + standalone eval (committed)
├── reward/
│   └── math_reward.py        # custom reward function (boxed extraction + SymPy grader)
├── scripts/
│   ├── setup_hpc.sh          # one-time login-node env setup
│   ├── run_local_test.sh     # pre-flight smoke test
│   └── train_oneshot_sdpo.slurm  # SLURM training job (4×A100, 16h, auto-resume)
├── eval/
│   ├── eval_math500.py       # standalone MATH-500 evaluation
│   └── eval_math500.slurm   # SLURM eval job (1×A100, 2h)
├── requirements.txt
└── README.md
```

Both parquet files are committed directly to the repo (sourced from One-Shot-RLVR,
pinned commits). No data download step needed.

The core SDPO/verl logic lives in the **installed package**
(`pip install git+https://github.com/lasgroup/SDPO.git`).
Our repo contributes only: reward function, data, and experiment config.

## HPC Setup (run once on login node)

```bash
conda activate sdpo
bash /home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/scripts/setup_hpc.sh
```

`setup_hpc.sh` installs SDPO (verl + all core deps) and sympy via pip, creates the
output directory tree, and verifies key imports.

## Training

```bash
conda activate sdpo
cd /home/woody/iwi7/iwi7107h/oneshotrlvrSDPO

# 1. Pre-flight smoke test
bash scripts/run_local_test.sh

# 2. Submit training job
sbatch scripts/train_oneshot_sdpo.slurm
```

### Auto-resume after timeout

`trainer.resume_mode=auto` is set — resubmit the same command after a timeout:

```bash
sbatch scripts/train_oneshot_sdpo.slurm   # Job 1: runs until 16h timeout
sbatch scripts/train_oneshot_sdpo.slurm   # Job 2: resumes from last checkpoint
```

Max work lost per timeout = 50 steps (`save_freq=50`).

## Evaluation

```bash
# Evaluate final checkpoint (step 500)
sbatch eval/eval_math500.slurm

# Evaluate a specific step
sbatch --export=CKPT=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/output/checkpoints/global_step_300 \
    eval/eval_math500.slurm
```

Results written to `output/eval_results/eval_step_{N}.jsonl`.
Final line of each file is a summary:
```json
{"summary": true, "step": 500, "accuracy": 0.42, "correct": 210, "total": 500}
```

## Output Layout

```
output/
├── checkpoints/
│   └── global_step_{50,100,...,500}/
│       ├── actor/              ← HuggingFace model weights (loadable by vLLM)
│       └── actor_optimizer/    ← optimizer states (only needed to resume training)
├── tensorboard/
│   └── oneshot_sdpo/oneshot_sdpo_qwen25math_1.5b_pi1/
├── logs/
│   ├── train_{JOBID}.out/err
│   └── eval_math500_{JOBID}.out/err
├── rollouts/                   ← MATH-500 validation rollouts (every 50 steps)
│   └── {0,50,100,...,500}.jsonl
├── train_rollouts/             ← training rollouts (every step)
│   └── {1,2,...,500}.jsonl
└── eval_results/               ← standalone eval output
    └── eval_step_{N}.jsonl
```

### JSONL schema (rollouts + eval)

```json
{
  "input": "prompt text",
  "output": "model response",
  "gts": "ground truth",
  "score": 1.0,
  "step": 100,
  "extracted_answer": "42",
  "is_correct": true
}
```

## Monitoring

```bash
# TensorBoard
tensorboard --logdir /home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/output/tensorboard

# Follow live log
tail -f /home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/output/logs/train_<JOBID>.out
```

### TensorBoard metrics

| Category | Key examples |
|----------|-------------|
| Reward | `critic/score/mean`, `critic/rewards/mean` |
| Advantage | `critic/advantages/mean/max/min` |
| Response length | `response_length/mean`, `response_length/clip_ratio` |
| Actor | `actor/pg_clipfrac`, `actor/grad_norm`, `actor/entropy` |
| SDPO distillation | `self_distillation/success_group_fraction`, `self_distillation/reprompt_sample_fraction` |
| IS correction | `rollout_corr/kl`, `rollout_corr/rollout_is_mean`, `rollout_corr/rollout_is_eff_sample_size` |
| Validation | `val-core/simplerl/math500/reward/mean@1` |
| Throughput | `perf/throughput`, `timing_s/gen` |

## SDPO Algorithm

```
Each training step:
  1. Rollout:   sample n=8 responses per prompt (temp=0.6)
  2. Reward:    grade with compute_score() → binary 1.0/0.0
  3. Distill:   for each prompt group:
                  success = responses with score=1.0
                  failed  = responses with score=0.0
                  build reprompt: failed response + EMA teacher demonstration
  4. Loss:      SDPO_loss = JSD(student, teacher)   [alpha=0.5]
                           + kl_loss_coef * KL(actor||ref)
                           * IS_weight (token-level, clipped at 2.0)
  5. Update:    gradient step on actor
                EMA teacher: θ_t ← 0.95·θ_t + 0.05·θ_actor

Notes:
  - GRPO advantages are computed (adv_estimator=grpo) but NOT used in the loss.
    Success/failure is determined by reward score (threshold=1.0), not advantage.
  - The entire training signal comes from the EMA teacher via JSD distillation.
  - IS correction weights the distillation loss token-by-token.

Eval (every 50 steps):
  greedy rollout on MATH-500 (n=1, temp=0)
  → val-core/simplerl/math500/reward/mean@1
```

## Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Hydra config | `ppo_trainer` | `sdpo.yaml` requires `TASK`/`EXPERIMENT` env vars (CSCS-specific) |
| Data files | Committed parquets | No download step on HPC |
| `use_think` | `False` | Qwen2.5-Math-1.5B has no `<think>` tags |
| `alpha=0.5` | JSD between student and teacher | Balances forward and reverse KL |
| Grading | boxed extraction + mathd + SymPy | SymPy catches `64/5 == 12.8` |
| Resume | `trainer.resume_mode=auto` | Same sbatch command works for fresh start and resume |
| WandB | `WANDB_MODE=disabled` | TensorBoard only |
