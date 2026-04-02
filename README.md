# One-Shot-RLVR + SDPO

Reproduce the One-Shot-RLVR result — training **Qwen2.5-Math-1.5B** on a single
math problem repeated 128 times — using **SDPO** (Self-Distilled Policy Optimization)
instead of GRPO.

| Component | Source |
|---|---|
| Base model | Qwen2.5-Math-1.5B |
| Training data | π₁ × 128 (One-Shot-RLVR) |
| Algorithm | SDPO — EMA teacher + JSD distillation + IS correction |
| Evaluation | MATH-500, greedy (temp=0), pass@1 |
| Runtime | Apptainer `verlai/verl:vllm017.latest` (torch 2.10, vLLM 0.17, Ray 2.54) |
| Hardware | 4 × A100-SXM4-40GB, 16h wall time |

---

## Repository Structure

```
oneshotrlvrSDPO/
├── data/
│   ├── pi1_r128.parquet          # 128 copies of π₁ — one-shot training set
│   └── math500.parquet           # MATH-500 — validation + standalone eval
├── reward/
│   └── math_reward.py            # compute_score(): binary reward + rich π₁ feedback
├── scripts/
│   ├── run_local_test.sh         # smoke test (1×A100, 1 step, same params as production)
│   └── train_oneshot_sdpo.slurm  # production job (4×A100, 500 steps, auto-resume)
├── eval/
│   ├── eval_math500.py           # standalone MATH-500 evaluation script
│   └── eval_math500.slurm        # SLURM eval job
└── SDPO/                         # full lasgroup/SDPO clone (NOT committed, in .gitignore)
    └── verl/                     # loaded at runtime via PYTHONPATH — never modified
```

**No install step needed.** The Apptainer container has all Python dependencies.
`verl` is loaded from `SDPO/verl/` via `PYTHONPATH` at runtime.

---

## HPC Layout

```
/home/woody/iwi7/iwi7107h/
├── oneshotrlvrSDPO/         ← this repo
│   └── SDPO/                ← git clone https://github.com/lasgroup/SDPO (read-only)
├── images/
│   └── verl_vllm017_latest.sif  ← Apptainer image
└── models/
    └── Qwen2.5-Math-1.5B/   ← base model weights
```

Clone SDPO once on HPC (if not already present):
```bash
cd /home/woody/iwi7/iwi7107h/oneshotrlvrSDPO
git clone https://github.com/lasgroup/SDPO SDPO
```

---

## Quick Start

```bash
# Pull latest on HPC
cd /home/woody/iwi7/iwi7107h/oneshotrlvrSDPO
GIT_DISCOVERY_ACROSS_FILESYSTEM=1 git pull origin claude/integrate-rlvr-sdpo-dlMU5

# Smoke test — runs 1 training step with production-identical parameters
salloc --partition=a100 --gres=gpu:a100:1 --ntasks=1 --cpus-per-task=8 --mem=80GB --time=00:30:00
apptainer exec --nv \
  --bind /home/woody/iwi7/iwi7107h/oneshotrlvrSDPO:/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO \
  /home/woody/iwi7/iwi7107h/images/verl_vllm017_latest.sif \
  bash /home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/scripts/run_local_test.sh

# Production training
sbatch scripts/train_oneshot_sdpo.slurm
```

---

## SDPO Algorithm (our configuration)

```
Each training step (128 prompts, all π₁):

  1. ROLLOUT
     vLLM generates n=8 responses per prompt at temp=0.6
     → 1024 responses total

  2. REWARD
     math_reward.compute_score() grades each response:
       score = 1.0  if \boxed{answer} matches ground truth (12.8)
       score = 0.0  otherwise
     Failed rollouts also receive rich feedback:
       "Use the relation P = k A V^3. Recompute k = 1/256.
        Simplify to V^3 = 2048 before cube root.
        The accepted training label for this environment is 12.8."

  3. REPROMPT (SDPO core)
     For each failed rollout:
       If EMA teacher has a successful solution → show it as demonstration
       Else → append feedback text to prompt (environment_feedback_only_without_solution=true)
     Student generates a second attempt on the reprompted input

  4. DISTILLATION LOSS
     JSD(student || EMA_teacher) on reprompted trajectories
       = 0.5 × KL(student||teacher) + 0.5 × KL(teacher||student)
     Top-100 tokens only (distillation_topk=100)
     IS weight per token = p_current/p_rollout, clipped at 2.0

  5. GRPO LOSS
     Standard GRPO advantage on successful rollouts
     norm_adv_by_std_in_grpo=False (SDPO paper setting)

  6. GRADIENT UPDATE
     AdamW, lr=1e-6, 10-step warmup

  7. EMA TEACHER UPDATE
     teacher = 0.95 × teacher + 0.05 × student
```

---

## Key Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `rollout.n` | 8 | rollouts per prompt |
| `rollout.temperature` | 0.6 | generation temperature |
| `data.max_response_length` | 3072 | matches One-Shot-RLVR original |
| `self_distillation.alpha` | 0.5 | symmetric JSD |
| `self_distillation.teacher_update_rate` | 0.05 | EMA τ |
| `self_distillation.is_clip` | 2.0 | IS weight clip |
| `self_distillation.distillation_topk` | 100 | top-k tokens for JSD |
| `self_distillation.success_reward_threshold` | 1.0 | binary reward |
| `optim.lr` | 1e-6 | learning rate |
| `trainer.total_training_steps` | 500 | total steps |
| `trainer.save_freq` | 50 | checkpoint every N steps |
| `trainer.test_freq` | 50 | validate every N steps |

---

## Output Layout

```
output/
├── checkpoints/
│   └── oneshot_sdpo/oneshot_sdpo_qwen25math_1.5b_pi1/
│       └── global_step_{50,100,...,500}/
│           └── actor/              ← HuggingFace model (loadable by vLLM)
├── tensorboard/
│   └── oneshot_sdpo/oneshot_sdpo_qwen25math_1.5b_pi1/
├── logs/
│   ├── train_{JOBID}.out
│   └── train_{JOBID}.err
├── rollouts/                       ← MATH-500 validation JSONLs (every 50 steps)
└── train_rollouts/                 ← training rollout JSONLs (every step)
```

### JSONL schema (train_rollouts)

```json
{
  "prompt": "The pressure P exerted by wind...",
  "response": "Let me solve step by step... \\boxed{12.8}",
  "score": 1.0,
  "extracted_answer": "12.8",
  "feedback": ""
}
```

Failed rollout:
```json
{
  "prompt": "The pressure P exerted by wind...",
  "response": "...\\boxed{15}",
  "score": 0.0,
  "extracted_answer": "15",
  "feedback": "Use the relation P = k A V^3. Recompute k = 1/256..."
}
```

---

## Monitoring

```bash
# Follow live training log
tail -f output/logs/train_<JOBID>.out

# TensorBoard
tensorboard --logdir output/tensorboard
```

### Key metrics to watch

| Metric | What it means |
|---|---|
| `self_distillation/success_group_fraction` | Fraction of prompts where EMA teacher had a success — should rise over training |
| `self_distillation/reprompt_sample_fraction` | Fraction of rollouts that were reprompted |
| `score_mean` | Mean reward — should increase over 500 steps |
| `actor_entropy` | Should stay > 0 (no collapse) |
| `rollout_is_mean` | IS weights — healthy near 1.0 |

---

## Auto-Resume

`trainer.resume_mode=auto` — resubmit the same command after timeout:

```bash
sbatch scripts/train_oneshot_sdpo.slurm   # runs until 16h limit
sbatch scripts/train_oneshot_sdpo.slurm   # resumes from last checkpoint
```

Max work lost = 50 steps.

---

## Evaluation

```bash
# Evaluate final checkpoint
sbatch eval/eval_math500.slurm

# Evaluate specific step
CKPT=output/checkpoints/oneshot_sdpo/oneshot_sdpo_qwen25math_1.5b_pi1/global_step_300 \
sbatch eval/eval_math500.slurm
```

---

## Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Runtime | Apptainer (no conda) | HPC is read-only inside container; no `pip install -e .` |
| verl source | `SDPO/` via PYTHONPATH | Container read-only; avoids stale copies |
| Hydra config | `ppo_trainer` (not `sdpo`) | `sdpo.yaml` requires CSCS-specific env vars |
| Feedback | Rich π₁ hints | `include_environment_feedback=true`; denser signal than binary "incorrect" |
| `is_correct` | Removed from reward dict | SDPO aggregates extra-info to numpy arrays → `numpy.bool_` not JSON-serializable |
| `APPTAINERENV_LD_LIBRARY_PATH` | CUPTI path injected | Apptainer `--nv` rewrites `LD_LIBRARY_PATH`; must use `APPTAINERENV_` prefix |
| `use_think` | False | Qwen2.5-Math-1.5B has no `<think>` tags |
| WandB | Disabled | TensorBoard only |
