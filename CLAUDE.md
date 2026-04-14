# CLAUDE.md — One-Shot-RLVR + SDPO

> **Convention**: Update this file and `CHANGES.md` whenever config, code, or decisions
> change. `CHANGES.md` is the session log (what changed + why). This file is the
> authoritative reference (what is currently true).

## What We Are Building

**Goal**: Reproduce the One-Shot-RLVR result (training Qwen2.5-Math-1.5B on a single
example to improve math reasoning) but using **SDPO** as the training algorithm instead
of GRPO.

| Component | Source | Role |
|---|---|---|
| **Data** (π₁, 128 copies) | https://github.com/ypwang61/One-Shot-RLVR | Single training example duplicated 128× |
| **Reward** (binary boxed-answer check) | One-Shot-RLVR `deepscaler.py` | 1.0 if `\boxed{}` matches ground truth, else 0.0 |
| **Training algorithm** | https://github.com/lasgroup/SDPO (paper: https://arxiv.org/abs/2601.20802) | SDPO: EMA teacher + JSD distillation loss |
| **Base model** | Qwen2.5-Math-1.5B | Same as One-Shot-RLVR paper |
| **Eval** | MATH-500 | pass@1, greedy (temp=0) |

The SDPO verl fork is the **single source of truth** for all config keys, APIs, and
training logic. Check it first before answering any question.

The **trimmed SDPO source** (verl/ only, no examples/experiments/build artifacts) is
committed directly in this repo at `SDPO/` and is what gets pulled on HPC via `git pull`.
See the [SDPO source layout](#hpc-layout) section below.

---

## ALWAYS Cross-Check Against SDPO First

Before answering questions about config keys, hyperparameters, reward function signature,
data format, or script args — verify against the SDPO source in `SDPO/verl/` (committed
in this repo) or upstream at **https://github.com/lasgroup/SDPO**.

Canonical files (all under `SDPO/verl/` in this repo):

| File | What it defines |
|---|---|
| `verl/trainer/config/ppo_trainer.yaml` | Base trainer config (all defaults) |
| `verl/trainer/config/actor/actor.yaml` | Actor + SDPO self-distillation defaults |
| `verl/trainer/config/rollout/rollout.yaml` | vLLM rollout defaults |
| `verl/trainer/config/sdpo.yaml` | SDPO overrides (inherits ppo_trainer + user) |
| `verl/trainer/ppo/core_algos.py` | `compute_self_distillation_loss` — distillation loss |
| `verl/workers/reward_manager/naive.py` | `NaiveRewardManager` — calls `compute_score` |
| `verl/workers/actor/dp_actor.py` | `SDPOActor` — EMA teacher update |
| `verl/trainer/main_ppo.py` | Entry point: `python -m verl.trainer.main_ppo` |

---

## What One-Shot-RLVR Does (Original)

Paper: https://arxiv.org/abs/2504.20571 (NeurIPS 2025)

**Key finding**: Training Qwen2.5-Math-1.5B with GRPO on just **one example duplicated
128 times** (π₁) produces significant MATH-500 improvement. Data diversity is not
required — the verifiable reward signal is the key driver.

### π₁ — The Single Training Example

```text
Prompt:
"The pressure P exerted by wind on a sail varies jointly as the area A of the sail
and the cube of the wind's velocity V. When the velocity is 8 miles per hour, the
pressure on a sail of 2 square feet is 4 pounds. Find the wind velocity when the
pressure on 4 square feet of sail is 32 pounds.
Let's think step by step and output the final answer within \boxed{}."

Ground truth: 12.8
```

This example was selected from DSR-sub (1209 examples from DeepScaleR-Preview-Dataset)
by ranking on historical variance score (variance of Qwen2.5-Math-1.5B's accuracy).

### Original One-Shot-RLVR Training Script (GRPO, for reference)

```bash
# scripts/train/training_1.5b_pi1_r128.sh (from ypwang61/One-Shot-RLVR)
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=data/train/one_shot_rlvr/pi1_r128.parquet \
  data.val_files=data/test/math500.parquet \
  data.train_batch_size=128 \
  data.max_response_length=3072 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.temperature=0.6 \
  trainer.n_gpus_per_node=8 \
  trainer.total_epochs=2000
```

Key differences vs our SDPO implementation:

| Setting | One-Shot-RLVR (GRPO) | Our implementation (SDPO) |
|---|---|---|
| Algorithm | GRPO (`adv_estimator=grpo`) | SDPO (`loss_mode=sdpo` + JSD) |
| KL loss | `use_kl_loss=True`, coef=0.001 | `use_kl_loss=False` (SDPO uses JSD) |
| Entropy | not set | `entropy_coeff=0` (not set; original SDPO loss) |
| Response length | 3072 | 3072 |
| GPUs | 8 | 2 |
| max_model_len | not set (old verl) | 8192 |
| val_kwargs.n | not set | 1 |
| Reprompting | N/A | EMA teacher reprompts failed rollouts |
| Teacher | N/A (ref model) | EMA of student (update_rate=0.05) |
| Feedback | N/A | condition A: scalar-only (no feedback text) |

---

## One-Shot-RLVR Baseline Results (GRPO, Qwen2.5-Math-1.5B)

These are the numbers we are trying to match or beat with SDPO:

| Model | MATH-500 |
|---|---|
| Qwen2.5-Math-1.5B base | ~36% (pre-training only) |
| 1-shot RLVR (π₁, GRPO) | improves significantly |
| Full DSR-sub RLVR (1209 examples, GRPO) | upper bound |

For DeepSeek-R1-Distill-Qwen-1.5B results on MATH-500 (avg@16, 8k eval):
- Base: 76.7%
- 1-shot RLVR: **80.5%** (+3.8pp)
- 4-shot: 81.2%, 16-shot: 83.3%, full (1.2k): 84.4%

Key insight: even 1 training example closes most of the gap to full-dataset RLVR.

---

## What SDPO Does (Algorithm)

SDPO (Self-Distilled Policy Optimization) extends GRPO with:

1. **EMA teacher**: A slow-moving exponential moving average of the student weights
   (`teacher = (1-τ) * teacher + τ * student`, τ=0.05). Acts as a stable target
   distribution instead of the frozen reference model.

2. **Reprompting**: When the student fails (reward=0), the EMA teacher's successful
   rollout (if any) is appended to the prompt as a demonstration for the next attempt.
   (`include_environment_feedback=false` → only successful solutions, no feedback text).

3. **Distillation loss**: Instead of pure policy gradient, the loss is a weighted KL
   between student and teacher token distributions on the reprompted trajectories.
   `alpha=1.0` gives pure reverse KL (mode-seeking: student imitates teacher's best
   solution exactly). `alpha=0.5` gives symmetric JSD. Condition A uses `alpha=0.5`
   (Section 3 paper default); condition D uses `alpha=1.0`.

4. **Importance sampling correction** (`rollout_is=token`, threshold=2.0): Corrects
   for distribution shift between rollout-time and update-time policies.

The `compute_self_distillation_loss` in `verl/trainer/ppo/core_algos.py` implements:
- Full-logit KL with top-k truncation (top-`distillation_topk` tokens; we use 20)
- loss = alpha * forward_KL + (1-alpha) * reverse_KL
- IS weighting clipped at `is_clip=2.0`

---

## SDPO Config Defaults vs Our Overrides

**Source**: verified from `lasgroup/SDPO` config YAML files.

### ppo_trainer.yaml defaults → our overrides

| Key | SDPO default | Our override | Reason |
|---|---|---|---|
| `algorithm.adv_estimator` | `gae` | `grpo` | SDPO requires grpo (no critic) |
| `algorithm.norm_adv_by_std_in_grpo` | `True` | `False` | Unnormalized advantages preserve absolute reward scale; normalizing by std suppresses the signal when reward variance is low during early training |
| `trainer.logger` | `["console","wandb"]` | `["console","tensorboard"]` | No W&B on HPC |
| `trainer.n_gpus_per_node` | `8` | `2` | Our A100 allocation |
| `trainer.total_epochs` | `30` | `9999` | **Critical**: with 128-row dataset and batch_size=128, len(dataloader)=1 → epoch loop exits after 30 steps, ignoring total_training_steps. 9999 makes epoch loop infinite so total_training_steps controls termination |
| `trainer.total_training_steps` | `null` | `120` (condition A) / `90` (condition D) | 16h ÷ ~579s/step ≈ 99 steps safe max; extended to 120 for condition A to test oscillation stabilisation hypothesis (≈13.4h) |
| `trainer.resume_mode` | `disable` | `auto` (rich-feedback), `disable` (no-feedback) | Resume from global_step_20 checkpoint for rich-feedback run |
| `trainer.val_before_train` | `True` | `True` | Keep baseline measurement |

### actor/actor.yaml defaults → our overrides

| Key | SDPO default | Our override | Reason |
|---|---|---|---|
| `policy_loss.loss_mode` | `"vanilla"` | `sdpo` | **Required** for SDPO loss |
| `self_distillation.success_reward_threshold` | `0.5` (YAML) / `1.0` (docs — docs are wrong, YAML is authoritative) | `1.0` | Binary reward; only perfect = teacher |
| `self_distillation.include_environment_feedback` | `True` | `true` | **Condition D (current)**: rich feedback — sibling solution + verifier text |
| `self_distillation.environment_feedback_only_without_solution` | `False` | `false` | Include solution in reprompt (condition D) |
| `self_distillation.remove_thinking_from_demonstration` | `True` | `false` | Qwen2.5-Math-1.5B has no `<think>` tags |
| `self_distillation.max_reprompt_len` | `10240` | `2048` | Caps reprompt length; truncation_side=left prevents crash |
| `self_distillation.reprompt_truncation` | `"error"` | `left` | **Critical**: default "error" crashes when reprompt > max_reprompt_len; "left" truncates silently |
| `self_distillation.full_logit_distillation` | `True` | `true` | Same as default |
| `self_distillation.distillation_topk` | `100` | `100` (condition A) / `20` (condition D) | Condition A = Section 3 default; condition D = rich_feedback experiment setting |
| `self_distillation.alpha` | `0.5` | `0.5` (condition A) / `1.0` (condition D) | Paper Table 12: Section 3 (scalar reward) uses JSD (α=0.5); Section 4 (rich feedback) uses reverse KL (α=1.0) |
| `self_distillation.teacher_regularization` | `ema` | `ema` | Standard SDPO |
| `self_distillation.teacher_update_rate` | `0.05` | `0.05` (both conditions) | Paper/YAML default; 0.01 was tried for condition A but reverted — slower EMA did not improve early oscillation |
| `self_distillation.is_clip` | `2` | `2.0` | Same as default |
| `self_distillation.dont_reprompt_on_self_success` | `True` | `true` | Same as default |
| `use_kl_loss` | `false` | not overridden | No KL penalty; SDPO uses JSD |
| `entropy_coeff` | `0` | not overridden | Restored to SDPO default (0); original SDPO loss has no entropy term |
| `ppo_mini_batch_size` | `256` | `16` | 2 GPUs; keeps peak logit memory safe |
| `ppo_max_token_len_per_gpu` | — | `4096` | Teacher sequences = prompt(1024)+reprompt(2048)+response(3072) = 6144 tokens; at 4096 budget at most 1 student seq per micro-batch → teacher logsumexp = 6144×151936×4 = 3.73 GB; fits on 2×A100-40GB with vLLM reserved |
| `optim.lr` | `1e-6` | `1e-6` (both conditions) | SDPO generalization uses 1e-5 at batch=32. We use batch=128 (4× larger) so LR scaled down to 1e-6 to keep effective update size equivalent. lr=1e-5 caused catastrophic val drop (30.6%→16%) at step 12. |
| `optim.lr_warmup_steps` | `-1` | `0` (both conditions) | No warmup; LR is already conservative |

### rollout/rollout.yaml defaults → our overrides

| Key | SDPO default | Our override | Reason |
|---|---|---|---|
| `calculate_log_probs` | `False` | `True` | **Required** for SDPO IS correction |
| `max_model_len` | `null` | `8192` | prompt(1024)+response(3072)+reprompt(4096) |
| `max_num_batched_tokens` | `8192` | `16384` | 2× for concurrent sequences |
| `n` | `1` | `8` | 8 rollouts per prompt |
| `val_kwargs.n` | `1` | `1` | Greedy pass@1 (n=16 too expensive on 500 problems) |
| `temperature` | `1.0` | `0.6` | Qwen2.5-Math-1.5B recommended |
| `gpu_memory_utilization` | `0.5` | `0.4` | Safe for hybrid actor+vLLM on 2×A100-40GB |
| `tensor_model_parallel_size` | `2` | `1` | 1.5B model fits on 1 GPU |

### sdpo.yaml (SDPO cluster config, NOT used directly)

```yaml
# sdpo.yaml inherits user.yaml → requires TASK and EXPERIMENT env vars (CSCS-specific)
# We use --config-name ppo_trainer and set SDPO keys explicitly instead.
max_model_len: 18944        # for rich feedback (prompt+response+feedback+reprompt)
actor.ppo_mini_batch_size: 32   # single-GPU experiments
data.train_batch_size: 32       # single-GPU
actor.optim.lr: 1e-5            # their experiments; we use 1e-6
trainer.val_before_train: False  # they skip val; we keep it
```

---

## SDPO Source in This Repo

The **trimmed SDPO source** is committed directly in this repo at `SDPO/` on branch
`claude/integrate-rlvr-sdpo-dlMU5`. It contains only what is needed to run training:

| Kept | Removed (not needed at runtime) |
|---|---|
| `SDPO/verl/` — full verl Python source | `examples/`, `experiments/`, `scripts/`, `training/` |
| `SDPO/pyproject.toml`, `requirements.txt` | `verl.egg-info/`, `requirements-{cuda,full,test}.txt` |
| `SDPO/verl/experimental/agent_loop/` | `verl/experimental/{vla,fully_async_policy,one_step_off_policy,...}` |
| `SDPO/verl/workers/` (fsdp, vllm, reward) | `verl/workers/megatron_workers.py` |
| `SDPO/verl/trainer/{main_ppo,ppo/,config/}` | `verl/trainer/{sft_trainer*.py,main_eval.py,main_generation*.py}` |
| `SDPO/verl/utils/` (most) | `verl/utils/{sglang/,megatron/,megatron_utils.py,npu_flash_attn_utils.py}` |

The Apptainer container is **read-only**, so `pip install -e .` cannot be used inside it.
verl is loaded via `PYTHONPATH="${ONESHOT_DIR}/SDPO"` — no install step needed.

**On HPC**: `SDPO/` is pulled as part of the normal `git pull` — no separate clone needed.
The `.gitignore` previously excluded it but the trimmed version is now tracked.

**`verl/utils/reward_score/__init__.py` — why we bypass it**:
`default_compute_score` dispatches by `data_source` and raises `NotImplementedError`
for unknown sources. Our `data_source` is `"deepscaler"` (from One-Shot-RLVR parquet).
We bypass this entirely by using `custom_reward_function.path` in the Hydra config,
which loads our `math_reward.py` directly into `NaiveRewardManager`.

---

## Reward Function Interface

`reward/math_reward.py` implements `compute_score` matching SDPO's
`NaiveRewardManager` (`verl/workers/reward_manager/naive.py`):

```python
def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth,          # from parquet reward_model.ground_truth
    extra_info=None,       # NaiveRewardManager adds: num_turns, rollout_reward_scores, truncated
) -> dict:
    # Training (data_source=="deepscaler"), wrong answer:
    #   {"score": 0.0, "extracted_answer": str, "feedback": str}
    # Training, correct answer OR validation (any source):
    #   {"score": 1.0|0.0, "extracted_answer": str}   ← no "feedback" key
```

`NaiveRewardManager` populates `extra_info` before calling us:
- `extra_info["num_turns"]` — number of conversation turns (None for single-turn)
- `extra_info["rollout_reward_scores"]` — dict of scores from rollout (may be empty)
- `extra_info["truncated"]` — True if response hit max length without EOS

Our `compute_score` ignores these extra fields (they're not needed for binary math reward).

**Important**: The original One-Shot-RLVR `compute_score` returns a **float** (`0.` or `1.`).
Our version returns a **dict** — this is intentional. SDPO's `NaiveRewardManager` accepts
both; when a dict is returned it populates `reward_extra_infos` with the extra keys,
which then appear in the validation JSONL for debugging.

```python
# One-Shot-RLVR original (float return):
return 1.  # or 0.

# Our SDPO version (dict return — no is_correct!, feedback only for π₁ wrong answers):
return {"score": 1.0, "extracted_answer": model_answer}               # correct — any source
return {"score": 0.0, "extracted_answer": model_answer}               # wrong — val/non-deepscaler
return {"score": 0.0, "extracted_answer": model_answer, "feedback": "<π₁ hint>"}  # wrong — π₁ training
```

**Why no `is_correct` key**: SDPO's reward aggregator stacks `reward_extra_infos`
values into numpy arrays across the batch. Python `bool` becomes `numpy.bool_`,
which `json.dumps` rejects with `TypeError`. `score=1.0/0.0` encodes correctness.

**Grading chain** (three fallbacks, all must fail before score=0):
1. `grade_answer_mathd` — fast string normalisation (from One-Shot-RLVR utils)
2. `grade_answer_sympy` — sympy symbolic equality
3. `grade_answer_grader` — `math_equal()` from `reward/grader.py` (numeric tolerance, latex2sympy2, matrix equality); requires `latex2sympy2` + `regex` installed to `pkgs/`

**`feedback` key** (present only for π₁ training wrong answers):
The `feedback` key is only included in the return dict when `data_source == "deepscaler"`
AND `score == 0.0`. Validation (MATH-500) and correct answers never carry this key —
omitting it keeps the validation JSONL clean.
`compute_score` feeds `feedback` to SDPO's reprompt template (used when
`include_environment_feedback=true`; ignored silently for condition A).
Four-layer verifier in `_make_feedback()`:
- **Layer 0**: no `\boxed{}` → format nudge
- **Layer 1**: generic wrong answer → `"Your answer {X} is incorrect."`
- **Layer 2a** (π₁ only, response contains "2048"): cube-root error → `"V³ = 2048 is correct, but ∛2048 ≠ {X}. Re-check cube root."`
- **Layer 2b** (π₁ only, unit ratio detectable): unit/ratio error hint
- **Layer 2c** (π₁ only, arithmetic inconsistency): computation check hint
- **Layer 2d** (π₁ only, wrong quantity): quantity context hint
- Correct → `""` (empty; teacher forward skipped for this sample)

**Three training scripts — ablation conditions:**

| Script | Condition | `include_env_feedback` | Teacher context | Notes |
|---|---|---|---|---|
| `train_oneshot_sdpo.slurm` | **D** | `true` | question + sibling solution + verifier text + original failed response | alpha=1.0, topk=20, batch=128, temp=0.6 |
| `train_oneshot_sdpo_nofeedback.slurm` | **A** | `false` | question + sibling solution + original failed response | alpha=0.5, topk=100, batch=128, temp=0.6, no entropy |
| `train_oneshot_sdpo_paper_sec3.slurm` | **Paper §3** | `false` | question + sibling solution + original failed response | alpha=0.5, topk=100, batch=32, temp=1.0, **no entropy**, lr=1e-5, norm_adv=True |

**Important — teacher re-evaluates, does not generate**:
The teacher does NOT sample a new response. It takes the student's original (failed)
response tokens and recomputes `p_teacher(token_t | augmented_context + tokens_{1..t-1})`
for every token. This is the credit-assignment step: the same tokens, rescored under a
context that includes a correct solution, producing different per-token weights.
(Verified at `ray_trainer.py:762`: `teacher_input_ids = torch.cat([teacher_prompt_ids, responses], dim=1)`)

Grading fallback chain: `grade_answer_mathd` → `grade_answer_sympy` → `grade_answer_grader`.

---

## Parquet Schema

Both `pi1_r128.parquet` and `math500.parquet` must have columns:

```
data_source  prompt  ability  reward_model  extra_info
```

- `prompt`: `[{"role": "user", "content": "..."}]` (list of chat messages)
- `reward_model`: `{"ground_truth": "12.8"}` (dict with ground_truth key)
- `extra_info`: `{}` (empty dict is fine)

`pi1_r128.parquet` = 128 identical copies of the π₁ problem.
`math500.parquet` = 500 MATH problems for online validation.

---

## Environment — Apptainer (confirmed working baseline)

**HPC**: `/home/woody/iwi7/iwi7107h/`, A100-SXM4-40GB (Ampere, sm_80)
**Runtime**: Apptainer 1.4.5 — Docker is NOT available on this HPC
**Image**: `docker://verlai/verl:vllm017.latest`
**Image path**: `/home/woody/iwi7/iwi7107h/images/verl_vllm017_latest.sif`

Versions **inside the container** (Step 4 smoke test passes):

| Package | Version | Notes |
|---|---|---|
| Python | system `/usr/bin/python` | container-supplied |
| torch | 2.10.0+cu129 | |
| vllm | 0.17.0 | |
| flash-attn | 2.8.3 | |
| ray | 2.54.0 | |
| numpy | 1.26.4 | |
| sympy | 1.14.0 | |
| verl | SDPO source via PYTHONPATH | container is read-only — `pip install -e .` not possible |

**Key runtime decisions**:
- `pip install -e .` fails inside the container (read-only filesystem). Do not attempt it.
- verl loads from: `oneshotrlvrSDPO/SDPO/verl/__init__.py` via `PYTHONPATH`.
- `rollout.mode=sync` **does not exist** in this VERL version — do not set it.
- `model.dtype=bfloat16` is required; Flash Attention 2 expects bf16, not float32.
- No conda env needed. The Apptainer image contains all dependencies.

**Pull the image (one-time, run on login node)**:
```bash
apptainer pull /home/woody/iwi7/iwi7107h/images/verl_vllm017_latest.sif \
    docker://verlai/verl:vllm017.latest
```

**Verify env inside container**:
```bash
apptainer exec --nv \
  --bind /home/woody/iwi7/iwi7107h/oneshotrlvrSDPO:/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO \
  /home/woody/iwi7/iwi7107h/images/verl_vllm017_latest.sif \
  python - <<'PY'
import sys
sys.path.insert(0, "/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/SDPO")
sys.path.insert(0, "/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO")
import torch, vllm, ray, numpy, flash_attn, sympy, verl
from verl.trainer.ppo.core_algos import compute_self_distillation_loss
print("torch      :", torch.__version__)   # 2.10.0+cu129
print("vllm       :", vllm.__version__)    # 0.17.0
print("ray        :", ray.__version__)     # 2.54.0
print("numpy      :", numpy.__version__)   # 1.26.4
print("flash_attn :", flash_attn.__version__)
print("verl       :", verl.__file__)       # .../SDPO/verl/__init__.py
print("compute_self_distillation_loss : ok")
PY
```

**Known benign warnings** (do not affect training correctness):

| Warning | Cause | Fix applied |
|---------|-------|-------------|
| `Unsupported processor type: Qwen2TokenizerFast` | verl multimodal processor path; unused for text-only math | `PYTHONWARNINGS="ignore:Unsupported processor type"` |
| `OSError: [Errno 16] Device or resource busy: pymp-*` | Python multiprocessing temp cleanup on NFS home dir | `TMPDIR=/tmp` (node-local) |
| `No CUDA runtime is found, using CUDA_HOME='/usr/local/cuda'` | Subprocess inherits host env without module-loaded CUDA | `CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"` |
| Flash Attention dtype (float32) | verl loads model in float32 by default; FA2 prefers bf16 but still works | benign — no fix needed; training proceeds correctly |

**Known errors and fixes:**

| Error | Cause | Fix |
|-------|-------|-----|
| `ImportError: undefined symbol: cuptiActivityEnableDriverApi` | `libcupti.so.12` not found; Apptainer `--nv` rewrites `LD_LIBRARY_PATH` | `APPTAINERENV_LD_PRELOAD` with container libcupti path (already in slurm) |
| `TypeError: Object of type bool_ is not JSON serializable` | SDPO aggregates `reward_extra_infos` into numpy arrays; Python `bool` → `numpy.bool_` | Remove `is_correct` from `compute_score` return dict; use `score=1.0/0.0` instead |
| `fatal: not a git repository` on login node | `/home/woody` is a separate filesystem mount | `GIT_DISCOVERY_ACROSS_FILESYSTEM=1 git pull ...` or add to `~/.bashrc` |
| Training stops at ~30 steps despite `total_training_steps=1200` | `ppo_trainer.yaml` default `total_epochs=30`; with 1-batch dataset the epoch loop exhausts after 30 steps, `total_training_steps` never reached | `trainer.total_epochs=9999` — makes epoch loop infinite; `total_training_steps` controls via `is_last_step` |
| Crash when reprompt > `max_reprompt_len` | `reprompt_truncation` defaults to `"error"` in `ray_trainer.py:333`; tokenizer raises on truncation | `self_distillation.reprompt_truncation=left` — silently truncates from left |
| `ValueError: SDPO cannot share the reference policy with KL regularization.` | `main_ppo.py:133` explicitly rejects `use_kl_loss=True` because `teacher_module = ref_module_fsdp` — the ref IS the teacher; KL against it is incoherent | Do not set `use_kl_loss=True`; SDPO uses JSD distillation instead |

---

## Repo Structure

```
oneshotrlvrSDPO/
├── CLAUDE.md                              ← this file
├── CHANGES.md                             ← session log
├── README.md
├── requirements.txt
├── data/
│   ├── pi1_r128.parquet                   ← 128 copies of π₁ (One-Shot-RLVR training set)
│   ├── math500.parquet                    ← MATH-500 (validation)
│   └── pi1_example.json                   ← π₁ raw problem for reference
├── reward/
│   ├── math_reward.py                     ← compute_score() for NaiveRewardManager
│   └── grader.py                          ← math_equal() — third grading fallback
│                                             requires latex2sympy2 + regex (in pkgs/)
├── pkgs/                                  ← packages not in container (pip --target)
│   └── (latex2sympy2, regex, antlr4, ...) ← install: pip install --target=pkgs latex2sympy2 regex antlr4-python3-runtime==4.9.3
├── eval/
│   ├── eval_math500.py                    ← standalone MATH-500 eval
│   └── eval_math500.slurm
└── scripts/
    ├── train_oneshot_sdpo.slurm           ← condition D: rich feedback (2× A100, 90 steps)
    ├── train_oneshot_sdpo_nofeedback.slurm← condition A: scalar-only  (2× A100, 120 steps)
    ├── train_oneshot_sdpo_paper_sec3.slurm← paper §3: exact settings  (2× A100, 200 steps)
    ├── run_local_test.sh                  ← smoke test (Steps 1-4)
    └── setup_hpc.sh                       ← one-time HPC env setup
```

HPC layout:
```
/home/woody/iwi7/iwi7107h/
├── oneshotrlvrSDPO/        ← this repo (git pull brings everything including SDPO/)
│   ├── SDPO/               ← trimmed SDPO verl source (committed to git)
│   │                          pulled via normal git pull — no separate clone needed
│   │                          verl loaded from here via PYTHONPATH at runtime
│   │   ├── verl/           ← verl fork with SDPO modifications (main source)
│   │   ├── pyproject.toml
│   │   └── requirements.txt
│   ├── data/
│   ├── reward/
│   ├── scripts/
│   ├── eval/
│   ├── output/             ← checkpoints, logs (runtime, not committed)
│   └── outputs/            ← may also exist; runtime only
├── images/
│   └── verl_vllm017_latest.sif  ← Apptainer image (confirmed working)
└── models/
    └── Qwen2.5-Math-1.5B/  ← base model
```

**`SDPO/` is now committed in this repo** (trimmed: verl/ source only, no examples/scripts).
All verl APIs are accessible via `PYTHONPATH="${ONESHOT_DIR}/SDPO"` — no install step needed.
This is the only approach that works in the read-only Apptainer container.

To get SDPO on HPC — just pull the repo:
```bash
cd /home/woody/iwi7/iwi7107h/oneshotrlvrSDPO
GIT_DISCOVERY_ACROSS_FILESYSTEM=1 git pull origin claude/integrate-rlvr-sdpo-dlMU5
# SDPO/ will be present under oneshotrlvrSDPO/SDPO/
```

---

## Running

```bash
# Pull latest on HPC (filesystem boundary fix required)
cd /home/woody/iwi7/iwi7107h/oneshotrlvrSDPO
GIT_DISCOVERY_ACROSS_FILESYSTEM=1 git pull origin claude/integrate-rlvr-sdpo-dlMU5

# Add permanently to avoid typing it every time:
echo 'export GIT_DISCOVERY_ACROSS_FILESYSTEM=1' >> ~/.bashrc

# One-time: install packages not in container
pip install --target=${PROJECT_ROOT}/pkgs latex2sympy2 regex antlr4-python3-runtime==4.9.3

# Production training — condition D (rich feedback), resumes from global_step_20
sbatch scripts/train_oneshot_sdpo.slurm

# Production training — condition A (no feedback), fresh run
sbatch scripts/train_oneshot_sdpo_nofeedback.slurm

# Paper Section 3 exact settings (batch=32, temp=1.0, no entropy, lr=1e-5)
sbatch scripts/train_oneshot_sdpo_paper_sec3.slurm
```

**Per-step timing (measured on 2× A100, job from 2026-04-10):**
- `timing_s/gen`: ~420s (vLLM rollout — 72% of step time)
- `timing_s/update_actor`: ~139s
- `perf/time_per_step`: ~579s (~9.7 min/step)
- At 90 steps: ~14.6h — fits within 16h SLURM allocation

**First run observations (steps 1–30, 2026-04-10):**
- `critic/score/mean` at step 30: **0.68** (68% rollouts correct)
- `self_distillation/success_group_fraction`: 1.0 (every group has ≥1 success)
- `num_turns/mean`: 2.0 (reprompting active for all sequences)
- `response_length/clip_ratio`: 0.18 (18% of responses hit 3072-token max)
- `perf/max_memory_allocated_gb`: 46.6GB total across 2 GPUs (~23.3GB/GPU)

---

## Cross-Check: SDPO Rich Feedback Experiment Confirms Our Settings

Our hyperparameters are taken directly from
`lasgroup/SDPO/experiments/rich_feedback/run_sdpo.sh` — the SDPO team's own
rich-feedback experiment (the closest analogue to our setup: EMA teacher + feedback):

```bash
# From experiments/rich_feedback/run_sdpo.sh (CSCS cluster, Qwen3-8B, lcb_v6 dataset)
ROLLOUT_BATCH_SIZE=8                                           # = rollout.n; matches our n=8
LR=1e-6                                                        # matches ours
alpha=1.0                                                      # reverse KL → our alpha=1.0
teacher_update_rate=0.01                                       # slow EMA → our 0.01
distillation_topk=20                                           # → our topk=20
dont_reprompt_on_self_success=True                             # matches ours
algorithm.rollout_correction.rollout_is=token                  # matches ours
val_kwargs.n=4                                                 # they use 4; we use 1 (500 problems × 4 = too slow)
```

Note: their script targets CSCS cluster (different env/slurm syntax) and uses
`lcb_v6` coding data on Qwen3-8B. Our task is math (π₁, Qwen2.5-Math-1.5B) but
the SDPO hyperparameters transfer directly.

---

## Weight and Memory Layout (2× A100-40GB)

Three model slots — not four. The EMA teacher is NOT a separate copy:

```python
# fsdp_workers.py:905
self.actor.teacher_module = self.ref_module_fsdp   # teacher IS the ref model
```

| Slot | Config | Lives on | Notes |
|---|---|---|---|
| **Actor** (FSDP sharded ×2) | `param_offload=False`, `optimizer_offload=False` | GPU | Params + Adam states on GPU at all times |
| **Ref = EMA Teacher** (FSDP) | `ref.param_offload=True` | **CPU** | Loaded to GPU shard-by-shard during teacher forward only, then offloaded back |
| **vLLM rollout engine** | `gpu_memory_utilization=0.4` | GPU (1 GPU, TP=1) | Reserves ~16GB permanently; KV cache stays allocated |

**Rollout flow** (`async_rollout_mode=True` hardcoded; uses `AgentLoopManager`):
```
generate_sequences()
  wake_up()  → rollout_mode():
                 FSDP.state_dict()            # all-gather shards across 2 GPUs
                 rollout.update_weights()     # overwrite vLLM weights in-place
  AgentLoopWorkers run (one per GPU, async multi-turn)
  sleep()    → trainer_mode()
               # KV cache stays allocated (free_cache_engine not set)
```

**Update flow:**
```
ulysses_sharding_manager.__enter__()    # sequence parallel context (no-op, no Ulysses SP)
  actor FSDP forward → student log probs
  teacher forward (no_grad):
    FSDP loads ref shard to GPU → forward → offload back to CPU  (×28 layers)
  actor backward → grad → optimizer step
  EMA update:  teacher_param.data = 0.99×teacher + 0.01×student_data.to(CPU)
ulysses_sharding_manager.__exit__()
```

**"Async" clarification**: `async_rollout_mode=True` means agent loop workers handle
prompts concurrently. It does NOT overlap rollout N+1 with training step N — the
training loop still waits for `generate_sequences()` to finish before updating.

---

## Active Branch

`claude/integrate-rlvr-sdpo-dlMU5`
