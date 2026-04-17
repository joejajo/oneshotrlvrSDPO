# CHANGES.md — Session Log & Decisions

## Goal

Train Qwen2.5-Math-1.5B with **SDPO** (EMA teacher + reverse KL distillation + rich
feedback) on a **single training example duplicated 128×** (π₁) and measure MATH-500
improvement. This reproduces the One-Shot-RLVR result (originally GRPO) using SDPO
as the training algorithm.

Target: beat or match the GRPO baseline (+3.8pp on MATH-500 for DeepSeek-R1-Distill-Qwen-1.5B).

---

## Recent Changes

### 2026-04-17 — Align sec3 script to SDPO paper scalar reward hyperparameters

**Files**: `scripts/train_oneshot_sdpo_paper_sec3.slurm`, `CLAUDE.md`

**Problem**: The sec3 script had three deviations from the SDPO paper's actual scalar
reward (generalization) settings (`experiments/generalization/run_sdpo_all.sh` + `sdpo.yaml`):
1. `norm_adv_by_std_in_grpo=True` — sdpo.yaml default is `False`; generalization script doesn't override
2. `ppo_mini_batch_size=16` — sdpo.yaml and generalization script both use `32`
3. `lr_warmup_steps=0` — generalization script sets `10`

**Fix**:
- `norm_adv_by_std_in_grpo`: True → **False**
- `ppo_mini_batch_size`: 16 → **32**
- `lr_warmup_steps`: 0 → **10**
- Updated comments and CLAUDE.md table accordingly
- `val_kwargs.n` intentionally kept at 1 (paper uses 16 but 500×16=8000 rollouts too expensive)

---


**Files**: `scripts/train_oneshot_sdpo.slurm`, `CLAUDE.md`

**Problem**: val reward (`val-core/*/math500/reward/mean@1`) peaked at ~0.45 around step
80–100, then crashed and oscillated between 0.37–0.44 for 170+ more steps. Two causes:

1. **`norm_adv_by_std_in_grpo=True`** — near saturation (~60% accuracy), some batches
   have near-zero reward std. GRPO divides advantages by std → near-zero divisor →
   exploding advantage → large gradient step → policy overcorrects → val drops.
   Condition A already had `=False`; condition D mistakenly had `=True`.

2. **Stochastic val (`temperature=0.6, do_sample=True`)** — each evaluation step draws
   a different sample, adding ±2–3pp noise per step. On a 500-problem eval this makes a
   stable model look like it's oscillating.

**Fixes applied to `train_oneshot_sdpo.slurm`**:
- `algorithm.norm_adv_by_std_in_grpo=True` → `False`
- `val_kwargs.temperature=0.6` → `0`, `top_p=0.95` → `1.0`, `do_sample=True` → `False`

**Why not fix paper §3 script**: `norm_adv_by_std_in_grpo=True` is intentional there —
it matches the paper's exact settings for comparison purposes.

### 2026-04-08 — Remove expandable_segments (incompatible with vLLM CuMemAllocator)

**Files**: `scripts/train_oneshot_sdpo.slurm`

Job 1568106 crashed at startup (3 min) with:
```
AssertionError: Expandable segments are not compatible with memory pool.
Please track https://github.com/pytorch/pytorch/issues/147851 for the latest updates.
```

`vLLM v1's CuMemAllocator` (`vllm/device_allocator/cumem.py`) explicitly asserts that
`expandable_segments:True` is NOT set in `PYTORCH_CUDA_ALLOC_CONF`. Removed the env var.

OOM mitigation at π₁ saturation (step 11+) now relies solely on `ppo_mini_batch_size=32`:
- 32 sequences × 2000 tokens/seq / 4 GPUs = 8000 tokens per GPU → ~4.87 GB logits (fits)
- Previously needed expandable_segments because 64 sequences → 16000 tokens → 9.75 GB OOM

---

### 2026-04-08 — Switch to condition A: scalar-only (primary experiment)

**Files**: `scripts/train_oneshot_sdpo.slurm`, `CLAUDE.md`

Set `include_environment_feedback=false`. This is condition A — the primary experiment.

**Rationale**: The SDPO paper explicitly separates two regimes:
- Section 3 / math / scalar reward: successful sibling rollouts are the only implicit
  feedback. No environment text. This is what we reproduce.
- Figure 4 / coding / rich feedback: runtime errors, failing tests, LLM judge output.

Our π₁ localized verifier (Layer 2: "V³=2048 correct, ∛2048 ≠ 12") is a valid secondary
ablation but should not be the primary claim. With scalar-only, SDPO's improvement comes
purely from the teacher re-evaluating original failed response tokens conditioned on a
successful sibling — exactly the mechanism the paper claims works for math without rich
environment feedback. If that improves MATH-500 over GRPO, the story is clean.

The `feedback` field in `compute_score` is still computed (and appears in rollout JSONLs
for debugging) but is not passed to the trainer in this configuration.

Ablation plan:
- **A (current)**: scalar-only — `include_environment_feedback=false`
- **B**: generic feedback — `include_environment_feedback=true`, Layer 1 only
- **C**: localized verifier — `include_environment_feedback=true`, Layer 2, no solution
- **D**: localized verifier + solution — full Table 2 template

### 2026-04-08 — Localized verification feedback (SDPO Figure 4 mechanism)

**Files**: `reward/math_reward.py`, `scripts/train_oneshot_sdpo.slurm`

Implemented the SDPO paper's Figure 4 feedback mechanism: the self-teacher re-evaluates
the original response tokens y conditioned on [x, f, y_{<t}]. The KL loss is **sparse**
— disagreement concentrates at the specific tokens the feedback contradicts, not the
entire sequence. "Don't include n." makes the teacher disagree at `+1`, not everywhere.

Two changes:

**1. Localized verification feedback for π₁ (`reward/math_reward.py`)**

Two tiers depending on what the student's response contains:

- **Tier 1** (response contains "2048" → correct intermediate V³=2048, wrong cube root):
  `"V³ = 2048 is correct, but ∛2048 ≠ {answer}. Re-check the cube root step."`
  → teacher disagrees only at the cube-root tokens; rest of derivation gets no blame

- **Tier 2** (no "2048" → bad setup from earlier in the derivation):
  `"Checking: with V={answer}, P = (1/256)×4×{answer}³ = {computed:.4f}, but P should equal 32. Re-check your setup."`
  → math analog of `"AssertionError: expected 32, got 15.625"` — shows how far off V was

- **MATH-500 / other data sources**: `"Your answer {X} is incorrect."` (fallback; validation
  does not use reprompting so feedback is irrelevant there)

Examples from step 12 rollouts:
| Answer | Tier | Feedback |
|--------|------|---------|
| 10 | 2 (bad setup) | `P = 15.625 ≠ 32. Re-check your setup.` |
| 12 | 1 (∛ wrong) | `V³ = 2048 correct, but ∛2048 ≠ 12. Re-check cube root.` |
| 12.7 | 1 (∛ rounding) | `V³ = 2048 correct, but ∛2048 ≠ 12.7. Re-check cube root.` |
| 64 | 2 (bad setup) | `P = 4096 ≠ 32. Re-check your setup.` |

**2. `environment_feedback_only_without_solution=false`** (`train_oneshot_sdpo.slurm`)

Paper Table 2 full template uses BOTH solution AND feedback simultaneously:
```
Correct solution: [teacher's correct derivation]
The following is feedback from your unsuccessful earlier attempt:
V³ = 2048 is correct, but ∛2048 ≠ 12. Re-check the cube root step.
Correctly solve the original question.
```

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

---

### 2026-04-10 — grader.py integrated; four-layer verifier; entropy; condition D; OOM fixes; epoch bug fixed

**Files**: `reward/math_reward.py`, `reward/grader.py`, `eval/eval_math500.py`,
`scripts/train_oneshot_sdpo.slurm`, `scripts/train_oneshot_sdpo_nofeedback.slurm` (new)

#### 1. grader.py — third grading fallback
Copied `pipeline_files/grader.py` to `reward/grader.py`. Provides `math_equal()` using
`latex2sympy2`, `regex`, numeric tolerance, and matrix equality. Integrated as a third
fallback in `math_reward.py` and `eval/eval_math500.py`:
`grade_answer_mathd → grade_answer_sympy → grade_answer_grader`

Requires `latex2sympy2` + `regex` not in container → installed to `pkgs/` at
`/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/pkgs` via `pip install --target`.
PYTHONPATH updated in both slurm scripts to include `${PROJECT_ROOT}/pkgs`.

#### 2. Four-layer verifier in `_make_feedback()`
Replaced two-tier feedback with four-layer verifier for π₁:
- Layer 0: no `\boxed{}` → format nudge
- Layer 1: generic wrong answer
- Layer 2a: cube-root error ("V³=2048 correct, ∛2048 ≠ X")
- Layer 2b: unit/ratio hint (`_UNIT_RATIO_HINTS` pattern matching)
- Layer 2c: arithmetic inconsistency detection
- Layer 2d: wrong quantity context

`_make_feedback()` now accepts `ground_truth` parameter.

#### 3. Entropy regularization
Added `actor_rollout_ref.actor.entropy_coeff=0.001` matching One-Shot-RLVR's
`kl_loss_coef=0.001`. Entropy bonus applied on top of distillation loss at
`dp_actor.py:881-886`.

#### 4. Condition D (rich feedback) — now primary
Switched from condition A to condition D:
- `include_environment_feedback=true`
- `environment_feedback_only_without_solution=false`
Both sibling solution AND verifier text in teacher reprompt.
Separate `train_oneshot_sdpo_nofeedback.slurm` added for condition A comparison.

#### 5. OOM / crash fixes
- `reprompt_truncation=left`: default "error" crashes when reprompt > 2048 tokens
- `ppo_max_token_len_per_gpu=4096` (was 6000): teacher sequences up to 5120 tokens
  (truncated_prompt 2048 + response 3072); reduced bin ceiling prevents silent OOM

#### 6. Epoch loop bug — training stopped at 30 steps
`ppo_trainer.yaml` default `total_epochs=30`. With 128-row dataset and
`train_batch_size=128`, `len(train_dataloader)=1`. The outer epoch loop
`for epoch in range(0, total_epochs)` exhausts after 30 steps regardless of
`total_training_steps=1200`. Fixed: `trainer.total_epochs=9999`.

Observed: first run stopped at step 30 (SLURM killed near epoch-loop exhaustion).
Checkpoint at `global_step_20` exists; rich-feedback script uses `resume_mode=auto`.

#### 7. Steps and timing
Measured ~579s/step on 2×A100. At 16h: ~99 steps max → `total_training_steps=90`.
At step 30: `critic/score/mean=0.68`, all groups have successful sibling,
`num_turns/mean=2.0` (reprompting active).

---

---

### 2026-04-14 — Condition A tuning: token budget, advantage normalisation, EMA rate; feedback cleanup

**Files**: `scripts/train_oneshot_sdpo_nofeedback.slurm`, `reward/math_reward.py`, `CLAUDE.md`

#### Observations from first condition A run (steps 1–21)

- Step 0 baseline val: **30.4%** — matches prior baseline
- Val collapses to **11.8%** by step 6: 128× identical gradient signal overfit π₁
- Train reward oscillates: 0.736 (step 4) → 0.422 (step 6) → partial recovery
- **Root cause**: `clip_ratio` jumped to **29.1%** at step 6 — response length 634→1331 tokens; model writes correct but too-long reasoning, truncated before `\boxed{}` → reward=0
- `rollout_is_max` spiked to 13.69 at step 14 — severe IS drift confirming policy mismatch
- Oscillation cycle ~6–8 steps; hypothesis: may stabilise after 100 steps once model finds short reliable solution

**KL loss attempt (rejected)**:
Tried `use_kl_loss=True, kl_loss_coef=0.001` to anchor policy to reference model.
SDPO explicitly rejects this at startup: `ValueError: SDPO cannot share the reference
policy with KL regularization.` — because `teacher_module = ref_module_fsdp` in
`fsdp_workers.py:905`. The EMA teacher IS the reference model; KL against it is
architecturally incoherent. Reverted.

#### Config changes to condition A (`train_oneshot_sdpo_nofeedback.slurm`)

| Parameter | Before | After | Reason |
|---|---|---|---|
| `total_training_steps` | 75 | **120** | Test stabilisation hypothesis; 120 steps ≈ 13.4h within 16h |
| `teacher_update_rate` | 0.01 | **0.05** | Reverted to paper default; 0.01 did not improve oscillation |
| `norm_adv_by_std_in_grpo` | True | **False** | Unnormalized preserves reward scale; normalizing by std suppresses signal during low-variance early training |
| `ppo_max_token_len_per_gpu` | 4096 | **8000** | Response lengths hit 1300+ tokens; 4096 caused silent OOM in teacher forward at saturation |

#### `reward/math_reward.py` — remove feedback key from validation

`feedback` key now only present in return dict when `data_source == "deepscaler"`
AND `score == 0.0`. Previously always present (as `""` for val/correct), polluting
validation JSONL with a meaningless column. Validation (`lighteval/MATH`) and correct
answers now return `{"score": ..., "extracted_answer": ...}` only.

---

---

### 2026-04-14 — New slurm: Paper Section 3 exact settings

**Files**: `scripts/train_oneshot_sdpo_paper_sec3.slurm` (new)

Added a third training configuration matching SDPO paper Section 3 (Table 4) scalar-reward
setting as precisely as possible on the π₁ task.

**Key differences from condition A (`train_oneshot_sdpo_nofeedback.slurm`)**:

| Parameter | Condition A | Paper Section 3 | Reason |
|---|---|---|---|
| `rollout.temperature` | 0.6 | **1.0** | Paper uses temp=1.0 for scalar reward regime |
| `data.train_batch_size` | 128 | **32** | Paper batch size for Section 3 experiments |
| `optim.lr` | 1e-6 | **1e-5** | Paper lr for batch=32; our 1e-6 was scaled for batch=128 |
| `entropy_coeff` | 0.001 | **0** | Paper has no entropy term ("avoiding entropy") |
| `norm_adv_by_std_in_grpo` | False | **True** | Paper/YAML default |
| `total_training_steps` | 120 | **200** | batch=32 → ~4 steps/epoch → ~140s/step; 200×140s≈7.8h |
| `test_freq` | 2 | **5** | Fewer val runs to save wall time at batch=32 |

**Shared with condition A** (all SDPO-specific defaults):
- `alpha=0.5` (JSD), `teacher_update_rate=0.05`, `distillation_topk=100`
- `include_environment_feedback=false`, `is_clip=2.0`, `rollout.n=8`
- Reprompt template and solution template (explicit overrides)
- `val_kwargs.n=4, temperature=0.6` (stochastic validation)

Output dir: `output_sec3/`. Fresh run, `resume_mode=disable`.

---

---

### 2026-04-14 — Remove entropy from all scripts; restore original SDPO loss

**Files**: `scripts/train_oneshot_sdpo.slurm`, `scripts/train_oneshot_sdpo_nofeedback.slurm`, `CLAUDE.md`

Removed `entropy_coeff=0.001` from condition A and condition D scripts.

The original SDPO loss is:

```
L_SDPO = α·KL(m ‖ π_teacher) + (1−α)·KL(m ‖ π_student)
```

where `m = α·π_teacher + (1−α)·π_student` is the mixture distribution.
There is no entropy term in the paper's loss. The entropy bonus was added previously
to match One-Shot-RLVR's `kl_loss_coef=0.001` but that was a GRPO-specific setting
(KL to reference model). In SDPO the teacher IS the reference model; the distillation
loss already prevents collapse. Reverting to `entropy_coeff=0` (SDPO default).

`train_oneshot_sdpo_paper_sec3.slurm` already had `entropy_coeff=0` — no change needed.

---

---

### 2026-04-14 — Fix OOM: reduce ppo_max_token_len_per_gpu to 4096; fix KeyError: feedback

**Files**: `scripts/train_oneshot_sdpo_paper_sec3.slurm`, `scripts/train_oneshot_sdpo_nofeedback.slurm`, `reward/math_reward.py`

#### paper_sec3 OOM at step 5 (teacher forward logsumexp)

Job crashed: `torch.OutOfMemoryError: Tried to allocate 7.93 GiB. GPU 0 has 6.99 GiB free.`

**Root cause**: `ppo_max_token_len_per_gpu=8000` allows 2 student sequences of ~4096 tokens per
micro-batch. But teacher sequences are longer — reprompted context adds reprompt (2048) to prompt
(1024) + response (3072) = **6144 teacher tokens per sequence**. Two teacher sequences need:
`2 × 6144 × 151936 vocab × 4 bytes = 7.47 GB` → OOM with only 6.99 GB free.

**Fix**: `ppo_max_token_len_per_gpu=4096` → at most 1 student sequence per micro-batch (student max
= 1024 + 3072 = 4096 tokens). Teacher forward for 1 sequence: `6144 × 151936 × 4 = 3.73 GB` → fits.

Applied to both `paper_sec3` and `nofeedback` (condition A) — same OOM would hit condition A at
saturation when response lengths grow to 1300+ tokens.

Condition D (`train_oneshot_sdpo.slurm`) already had `ppo_max_token_len_per_gpu=4096` — no change.

`paper_sec3` changed to `resume_mode=auto` — valid checkpoint exists at `output_sec3/checkpoints/global_step_4`.

#### KeyError: 'feedback' in agent_loop (prior fix in same session)

`agent_loop.py:774` builds `non_tensor_batch[key]` from all samples in a batch and expects
uniform keys. Previously `feedback` was conditional (only wrong deepscaler answers). Within a
batch mixing correct + wrong answers the list comprehension crashed.

Fix: `compute_score` now always returns `feedback` key — `""` for correct answers and
non-training sources (MATH-500 val), real diagnostic string for wrong π₁ training answers.

---

## Current State

- **Branch**: `claude/integrate-rlvr-sdpo-dlMU5`
- **Status**: Three scripts ready to submit
- **Commands**:
  - `sbatch scripts/train_oneshot_sdpo_paper_sec3.slurm` — paper §3, resumes from global_step_4, 200 steps
  - `sbatch scripts/train_oneshot_sdpo_nofeedback.slurm` — condition A, fresh run, 120 steps
  - `sbatch scripts/train_oneshot_sdpo.slurm` — condition D, resumes from global_step_20, 90 steps
- **Config (paper sec3)**: batch=32, temp=1.0, lr=1e-5, no entropy, norm_adv=True; ~115s/step; 200 steps ≈ 6.4h
- **Config (condition A)**: batch=128, temp=0.6, lr=1e-6, no entropy; ~579s/step; 120 steps ≈ 19.4h
- **ppo_max_token_len_per_gpu**: 4096 for all three scripts (safe limit for teacher forward)
- **Key metrics to watch**: `clip_ratio` (must drop <10%), `success_group_fraction`, `critic/score/mean`
- **Checkpoints**: `output_sec3/global_step_4` (paper §3), `output/global_step_20` (condition D)
