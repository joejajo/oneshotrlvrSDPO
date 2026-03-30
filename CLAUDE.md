# CLAUDE.md — One-Shot-RLVR + SDPO

## What We Are Building

**Goal**: Reproduce the One-Shot-RLVR result (training Qwen2.5-Math-1.5B on a single
example to improve math reasoning) but using **SDPO** as the training algorithm instead
of GRPO.

| Component | Source | Role |
|---|---|---|
| **Data** (π₁, 128 copies) | https://github.com/ypwang61/One-Shot-RLVR | Single training example duplicated 128× |
| **Reward** (binary boxed-answer check) | One-Shot-RLVR `deepscaler.py` | 1.0 if `\boxed{}` matches ground truth, else 0.0 |
| **Training algorithm** | https://github.com/lasgroup/SDPO | SDPO: EMA teacher + JSD distillation loss |
| **Base model** | Qwen2.5-Math-1.5B | Same as One-Shot-RLVR paper |
| **Eval** | MATH-500 | pass@1, greedy (temp=0) |

The SDPO verl fork is the **single source of truth** for all config keys, APIs, and
training logic. Check it first before answering any question.

---

## ALWAYS Cross-Check Against SDPO First

Before answering questions about config keys, hyperparameters, reward function signature,
data format, or script args — verify against **https://github.com/lasgroup/SDPO**.

Canonical files in the SDPO repo:

| File | What it defines |
|---|---|
| `verl/trainer/config/ppo_trainer.yaml` | Base trainer config (all defaults) |
| `verl/trainer/config/actor/actor.yaml` | Actor + SDPO self-distillation defaults |
| `verl/trainer/config/rollout/rollout.yaml` | vLLM rollout defaults |
| `verl/trainer/config/sdpo.yaml` | SDPO overrides (inherits ppo_trainer + user) |
| `verl/trainer/ppo/core_algos.py` | `compute_self_distillation_loss` — JSD loss |
| `verl/workers/reward_manager/naive.py` | `NaiveRewardManager` — calls `compute_score` |
| `verl/workers/actor/dp_actor.py` | `SDPOActor` — EMA teacher update |
| `verl/trainer/main_ppo.py` | Entry point: `python -m verl.trainer.main_ppo` |
| `training/verl_training.sh` | SDPO launch wrapper |

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
| Response length | 3072 | 4096 |
| GPUs | 8 | 4 |
| max_model_len | not set (old verl) | 8192 |
| val_kwargs.n | not set | 16 |
| Reprompting | N/A | EMA teacher reprompts failed rollouts |
| Teacher | N/A (ref model) | EMA of student (update_rate=0.05) |

---

## What SDPO Does (Algorithm)

SDPO (Self-Distilled Policy Optimization) extends GRPO with:

1. **EMA teacher**: A slow-moving exponential moving average of the student weights
   (`teacher = (1-τ) * teacher + τ * student`, τ=0.05). Acts as a stable target
   distribution instead of the frozen reference model.

2. **Reprompting**: When the student fails (reward=0), the EMA teacher's successful
   rollout (if any) is appended to the prompt as a demonstration for the next attempt.
   (`include_environment_feedback=false` → only successful solutions, no feedback text).

3. **JSD distillation loss**: Instead of pure policy gradient, the loss is a JSD
   (Jensen-Shannon divergence) between student and teacher token distributions on the
   reprompted trajectories. `alpha=0.5` gives symmetric JSD.

4. **Importance sampling correction** (`rollout_is=token`, threshold=2.0): Corrects
   for distribution shift between rollout-time and update-time policies.

The `compute_self_distillation_loss` in `verl/trainer/ppo/core_algos.py` implements:
- Full-logit KL with top-k truncation (top-100 tokens)
- JSD = alpha * forward_KL + (1-alpha) * reverse_KL
- IS weighting clipped at `is_clip=2.0`

---

## SDPO Config Defaults vs Our Overrides

**Source**: verified from `lasgroup/SDPO` config YAML files.

### ppo_trainer.yaml defaults → our overrides

| Key | SDPO default | Our override | Reason |
|---|---|---|---|
| `algorithm.adv_estimator` | `gae` | `grpo` | SDPO requires grpo (no critic) |
| `algorithm.norm_adv_by_std_in_grpo` | `True` | `False` | SDPO paper; JSD dominates gradient |
| `trainer.logger` | `["console","wandb"]` | `["console","tensorboard"]` | No W&B on HPC |
| `trainer.n_gpus_per_node` | `8` | `4` | Our A100 allocation |
| `trainer.total_epochs` | `30` | — (use `total_training_steps=500`) | Step-based for one-shot data |
| `trainer.val_before_train` | `True` | `True` | Keep baseline measurement |

### actor/actor.yaml defaults → our overrides

| Key | SDPO default | Our override | Reason |
|---|---|---|---|
| `policy_loss.loss_mode` | `"vanilla"` | `sdpo` | **Required** for SDPO loss |
| `self_distillation.success_reward_threshold` | `0.5` | `1.0` | Binary reward; only perfect = teacher |
| `self_distillation.include_environment_feedback` | `True` | `false` | Math: no text feedback exists |
| `self_distillation.remove_thinking_from_demonstration` | `True` | `false` | Qwen2.5-Math-1.5B has no `<think>` tags |
| `self_distillation.max_reprompt_len` | `10240` | `4096` | No feedback text; just solution |
| `self_distillation.full_logit_distillation` | `True` | `true` | Same as default |
| `self_distillation.distillation_topk` | `100` | `100` | Same as default |
| `self_distillation.alpha` | `0.5` | `0.5` | JSD (symmetric) |
| `self_distillation.teacher_regularization` | `ema` | `ema` | Standard SDPO |
| `self_distillation.teacher_update_rate` | `0.05` | `0.05` | Same as default |
| `self_distillation.is_clip` | `2` | `2.0` | Same as default |
| `self_distillation.dont_reprompt_on_self_success` | `True` | `true` | Same as default |
| `use_kl_loss` | `false` | not overridden | No KL penalty; SDPO uses JSD |
| `ppo_mini_batch_size` | `256` | `128` | 4 GPUs, match per-GPU load |
| `optim.lr` | `1e-6` | `1e-6` | Same as default |
| `optim.lr_warmup_steps` | `-1` | `10` | Brief warmup for one-shot |

### rollout/rollout.yaml defaults → our overrides

| Key | SDPO default | Our override | Reason |
|---|---|---|---|
| `calculate_log_probs` | `False` | `True` | **Required** for SDPO IS correction |
| `max_model_len` | `null` | `8192` | prompt(1024)+response(4096)+reprompt(4096) |
| `max_num_batched_tokens` | `8192` | `16384` | 2× for concurrent sequences |
| `n` | `1` | `8` | 8 rollouts per prompt (same as SDPO) |
| `val_kwargs.n` | `1` | `16` | Stable pass@1 estimate (SDPO experiments) |
| `temperature` | `1.0` | `0.6` | Qwen2.5-Math-1.5B recommended |
| `gpu_memory_utilization` | `0.5` | `0.7` | A100 80GB |
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

## Reward Function Interface

`oneshot_sdpo/reward/math_reward.py` implements `compute_score` matching SDPO's
`NaiveRewardManager` (`verl/workers/reward_manager/naive.py`):

```python
def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth,          # from parquet reward_model.ground_truth
    extra_info=None,
) -> dict:
    # Returns: {"score": 0.0|1.0, "extracted_answer": str|None, "is_correct": bool}
```

Returns a **dict** (not a float) so that `extracted_answer` and `is_correct` are
collected into `reward_extra_infos` and written to the SDPO validation JSONL.

Grading: extract `\boxed{}` → `grade_answer_mathd` OR `grade_answer_sympy`.
Both taken verbatim from One-Shot-RLVR `verl/utils/reward_score/utils/utils.py`.

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

## Environment (sdpo2 conda on HPC)

**HPC**: `/home/woody/iwi7/iwi7107h/`, A100 (Ampere, sm_80)

| Package | Version | Notes |
|---|---|---|
| Python | 3.12 | |
| torch | 2.6.0+cu124 | vllm 0.8.5 requires exactly 2.6.0 |
| vllm | 0.8.5 | |
| flash-attn | 2.8.3 | cxx11abiFALSE wheel (pip-installed torch) |
| ray | 2.53.0 | WITHOUT `[default]` — avoids opentelemetry conflict with vllm |
| numpy | 1.26.4 | verl requires <2.0.0 |
| verl | SDPO editable | installed from `/home/woody/iwi7/iwi7107h/SDPO` |

**Why these exact versions**:
- `vllm==0.8.5` forces `torch==2.6.0`
- `flash_attn-2.8.3+cu12torch2.6cxx11abiFALSE` matches torch 2.6.0 pip ABI
- `ray` without `[default]` avoids `opentelemetry-sdk` version conflict with vllm
- numpy<2.0.0 required by verl's array compatibility

**Install flash-attn (exact wheel)**:
```bash
pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
```

**Install SDPO verl fork**:
```bash
cd /home/woody/iwi7/iwi7107h/SDPO
pip install -e . --no-deps
```

**Verify env**:
```python
import torch; print(torch.__version__)          # 2.6.0+cu124
import vllm; print(vllm.__version__)             # 0.8.5
import flash_attn; print(flash_attn.__version__) # 2.8.3
import verl; print(verl.__file__)                # .../SDPO/verl/__init__.py
from verl.trainer.ppo.core_algos import compute_self_distillation_loss  # must not error
```

---

## Repo Structure

```
oneshotrlvrSDPO/
├── CLAUDE.md                         ← this file
└── oneshot_sdpo/
    ├── data/
    │   ├── pi1_r128.parquet          ← 128 copies of π₁ (One-Shot-RLVR training set)
    │   ├── math500.parquet           ← MATH-500 (validation)
    │   └── pi1_example.json          ← π₁ raw problem for reference
    ├── reward/
    │   └── math_reward.py            ← compute_score() for NaiveRewardManager
    ├── eval/
    │   ├── eval_math500.py           ← standalone MATH-500 eval
    │   └── eval_math500.slurm
    └── scripts/
        ├── train_oneshot_sdpo.slurm  ← MAIN training job (4× A100)
        ├── run_local_test.sh         ← smoke test (Steps 1-4)
        └── setup_hpc.sh              ← one-time HPC env setup
```

HPC layout:
```
/home/woody/iwi7/iwi7107h/
├── oneshotrlvrSDPO/        ← this repo
├── SDPO/                   ← lasgroup/SDPO editable install
├── models/
│   └── Qwen2.5-Math-1.5B/  ← base model
└── output/                 ← checkpoints, logs, rollouts (runtime, not committed)
```

---

## Running

```bash
# Pull latest on HPC
cd /home/woody/iwi7/iwi7107h/oneshotrlvrSDPO
git pull origin claude/integrate-rlvr-sdpo-dlMU5

# Smoke test (needs 1 GPU)
salloc --partition=a100 --gres=gpu:a100:1 --ntasks=1 --cpus-per-task=8 --mem=80GB --time=00:30:00
conda activate sdpo2
cd oneshot_sdpo
bash scripts/run_local_test.sh

# Production training (4× A100)
sbatch oneshot_sdpo/scripts/train_oneshot_sdpo.slurm
```

---

## Active Branch

`claude/integrate-rlvr-sdpo-dlMU5`
