# CHANGES.md — Session Log & Decisions

## Goal

Train Qwen2.5-Math-1.5B with **SDPO** (EMA teacher + reverse KL distillation + rich
feedback) on a **single training example duplicated 128×** (π₁) and measure MATH-500
improvement. This reproduces the One-Shot-RLVR result (originally GRPO) using SDPO
as the training algorithm.

Target: beat or match the GRPO baseline (+3.8pp on MATH-500 for DeepSeek-R1-Distill-Qwen-1.5B).

---

## Recent Changes

### 2026-04-08 — Paper-faithful feedback mechanism (SDPO Figure 4)

**Files**: `reward/math_reward.py`, `scripts/train_oneshot_sdpo.slurm`

Implemented the environment feedback mechanism exactly as described in the SDPO paper
(arXiv:2601.20802, Figure 4 / Table 2):

1. **`reward/math_reward.py`**: Changed `_FEEDBACK_WRONG = ""` to
   `f"Your answer {model_answer} is incorrect."` — math analog of a coding test's
   `"AssertionError: expected 12.8, got 10"`. Specific, factual, no correct-answer hint.

2. **`train_oneshot_sdpo.slurm`**: Changed `environment_feedback_only_without_solution`
   from `true` → `false`. Paper Table 2 full template uses BOTH the solution AND the
   feedback simultaneously in the reprompt:
   ```
   Correct solution: [teacher's correct derivation]
   The following is feedback from your unsuccessful earlier attempt:
   Your answer 10 is incorrect.
   Correctly solve the original question.
   ```
   Previously we suppressed feedback when a solution existed — that was a conservative
   departure from the paper. Now we use both together.

**Effect on reprompting**: At step 12 (~50% success), failed rollout that answered `10`
now sees: correct solution (12.8 derivation) + "Your answer 10 is incorrect." The
self-teacher conditions on both and assigns per-token credit to the original response,
identifying the token(s) where the cube root calculation went wrong.

---

### 2026-04-08 — Fix post-saturation OOM; confirmed training working

**Files**: `scripts/train_oneshot_sdpo.slurm`

Run 2 completed steps 1-13 before OOM at step 13 (checkpoint saved at step 10).

**π₁ reward progress confirmed** (exactly as predicted by One-Shot-RLVR paper):
| Step | π₁ reward | success_group_fraction |
|------|-----------|----------------------|
| 1 | 0.043 | 0.23 |
| 9 | 0.107 | 0.60 |
| 11 | 0.463 | 0.98 ← saturation begins |
| 13 | 0.650 | 1.00 ← all groups solved |

**OOM cause**: at saturation, 99% of sequences are reprompted (~2000 tokens vs ~650).
Teacher forward processes `mini_batch/4GPUs × 2000 tokens = 32000 tokens` → 9.75 GiB
for `logsumexp`. With only 8.55 GiB free → OOM.

| Parameter | Before | After | Reason |
|---|---|---|---|
| `ppo_mini_batch_size` | `64` | `32` | halve sequences/GPU: 32000→16000 tokens |
| `PYTORCH_CUDA_ALLOC_CONF` | unset | `expandable_segments:True` | fix allocator fragmentation |

Will resume from step 10 checkpoint via `resume_mode=auto`.

---

### 2026-04-07 — Fix OOM: reduce token budget for SDPO dual-pass

**Files**: `scripts/train_oneshot_sdpo.slurm`

Job crashed at step 0 actor update with OOM on `logsumexp(logits)` in teacher forward.
SDPO runs two forward passes (student + teacher) and holds both logit tensors simultaneously.
With vocab=151936: 20000 tokens × 151936 × 2 bytes × 2 = 11.6 GB → OOM on 40GB A100.

**Baseline confirmed**: step 0 val = **31.4%** MATH-500 (Qwen2.5-Math-1.5B base).

| Parameter | Before | After | Reason |
|---|---|---|---|
| `ppo_mini_batch_size` | `128` | `64` | halve sequences per update |
| `ppo_max_token_len_per_gpu` | `20000` | `8000` | dual-pass logit memory: 8000×151936×2×2=4.6GB fits |

---

### 2026-04-07 — Bump training steps to 1200

**Files**: `scripts/train_oneshot_sdpo.slurm`

| Parameter | Before | After | Reason |
|---|---|---|---|
| `total_training_steps` | `500` | `1200` | One-Shot-RLVR paper shows peak at step ~1540; 500 stops mid-climb |

Job resubmitted on HPC with 1200 steps (Run 1: pure SDPO, no entropy).
Next run (Run 2) will add `entropy_coeff=0.001` + `calculate_entropy=true`.

---

### 2026-04-07 — Hyperparameter alignment with SDPO rich_feedback experiments

**Files**: `scripts/train_oneshot_sdpo.slurm`, `CLAUDE.md`

Aligned three hyperparameters to match `lasgroup/SDPO/experiments/rich_feedback/run_sdpo.sh`
(SDPO team's own rich-feedback experiment — closest analogue to our setup):

| Parameter | Before | After | Reason |
|---|---|---|---|
| `alpha` | `0.5` (JSD) | `1.0` (reverse KL) | Mode-seeking: student imitates teacher's solution exactly |
| `teacher_update_rate` | `0.05` | `0.01` | Slower EMA = more stable teacher for one-shot |
| `distillation_topk` | `100` | `20` | Less compute, matches SDPO rich_feedback |
| `val_kwargs.n` | `1` | `1` | Kept at 1 — 500 problems × 4 = too slow |

---

### 2026-04-07 — SDPO folder committed to repo (trimmed)

**Files**: `SDPO/` (357 files removed), `CLAUDE.md`, `.gitignore`

The full `lasgroup/SDPO` repo was uploaded by the user then trimmed to only what is
needed at runtime (`verl/` Python source). Removed:

- `examples/`, `experiments/`, `scripts/`, `training/` — reference scripts, not runtime
- `verl.egg-info/` — build artifact (we load via PYTHONPATH, not pip install)
- `verl/experimental/{vla,fully_async_policy,one_step_off_policy,dynamic_dataset,transfer_queue}/`
- `verl/utils/{sglang/,megatron/,megatron_utils.py,npu_flash_attn_utils.py}`
- `verl/workers/megatron_workers.py`
- `verl/trainer/{sft_trainer*.py,fsdp_sft_trainer.py,main_eval.py,main_generation*.py}`
- All `__pycache__/` and `.pyc` files

**Impact**: `git pull` on HPC now delivers `SDPO/` directly — no separate
`git clone https://github.com/lasgroup/SDPO` needed.

---

### 2026-04-07 — Rich feedback confirmed implicit

**Files**: `reward/math_reward.py`, `CLAUDE.md`

Confirmed and documented that feedback is non-revealing:
- Wrong answer → `feedback = ""` (empty) — EMA teacher demonstration is the signal
- No `\boxed{}` → format nudge only ("use `\boxed{your answer}`"), no math content
- Correct → `feedback = ""`

CLAUDE.md was updated to match (previously described explicit hints that were removed).

---

### Earlier — SIGSEGV fix (agent loop workers)

**Files**: `scripts/train_oneshot_sdpo.slurm`

Two fixes for SIGSEGV in AgentLoopWorker during validation:
- `rollout.agent.num_workers=1` — serializes Ray actor spawning, eliminates concurrent
  uvloop `getenv` + LD_PRELOAD `setenv` race in glibc env array
- `APPTAINERENV_UV_THREADPOOL_SIZE=4` — prevents libuv `getenv("UV_THREADPOOL_SIZE")`
  call during threadpool init

---

### Earlier — Triton version conflict fix

**Files**: `scripts/train_oneshot_sdpo.slurm`

`export PYTHONNOUSERSITE=1` — blocks `~/.local/lib/python3.12/site-packages/triton/`
(user-installed, incompatible) from overriding the container's Triton. Without this,
vLLM's `topk_topp_triton` kernel failed with `uint32/int32` signedness error.

---

### Earlier — validation metrics NoneType crash fix

**Files**: `reward/math_reward.py`

`extracted_answer: None` → `extracted_answer: ""` for the no-`\boxed{}` case.
SDPO's `process_validation_metrics` skips `isinstance(var_vals[0], str)` but `None`
is not a string — fell through to `np.mean([None, ...])` → TypeError.

---

## Current State

- **Branch**: `claude/integrate-rlvr-sdpo-dlMU5`
- **Status**: Ready to submit production job
- **Command**: `sbatch scripts/train_oneshot_sdpo.slurm`
- **Config**: 4× A100-40GB, 500 steps, train_batch=128, rollout.n=8, 16h wall time
- **Known working**: smoke test (1 step, 1 GPU) passes with all fixes applied
- **Not yet tested**: full 500-step run with new hyperparams (alpha=1.0, rate=0.01, topk=20)
