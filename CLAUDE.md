# CLAUDE.md — One-Shot-RLVR + SDPO

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
| `self_distillation.success_reward_threshold` | `0.5` (YAML) / `1.0` (docs — docs are wrong, YAML is authoritative) | `1.0` | Binary reward; only perfect = teacher |
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

## What Files We Need From SDPO (Answer: None Beyond the Editable Install)

**DO NOT copy files from `lasgroup/SDPO/verl/` into this repo.**
The `pip install -e .` from `/home/woody/iwi7/iwi7107h/SDPO` makes all verl APIs
available as a Python package. Copying files would create stale duplicates.

| SDPO directory | Do we copy it? | Why |
|---|---|---|
| `verl/` (all subdirs) | **NO** | Covered by editable install |
| `training/verl_training.sh` | **NO** | We have our own slurm + run scripts |
| `run_local_sdpo.sh` | **NO** | Reference only; we have run_local_test.sh |
| `examples/` | **NO** | Reference only |
| `docs/` | **NO** | Documentation only |
| `requirements.txt` | **NO** | Installed directly from SDPO dir |
| `requirements-cuda.txt` | **NO** | Just `flash-attn` — covered by install steps |

**`verl/utils/reward_score/__init__.py` — why we bypass it**:
`default_compute_score` dispatches by `data_source` and raises `NotImplementedError`
for unknown sources. Our `data_source` is `"deepscaler"` (from One-Shot-RLVR parquet).
We bypass this entirely by using `custom_reward_function.path` in the Hydra config,
which loads our `math_reward.py` directly into `NaiveRewardManager`.

---

## Reward Function Interface

`oneshot_sdpo/reward/math_reward.py` implements `compute_score` matching SDPO's
`NaiveRewardManager` (`verl/workers/reward_manager/naive.py`):

```python
def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth,          # from parquet reward_model.ground_truth
    extra_info=None,       # NaiveRewardManager adds: num_turns, rollout_reward_scores, truncated
) -> dict:
    # Returns: {"score": 0.0|1.0, "extracted_answer": str|None, "is_correct": bool}
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

# Our SDPO version (dict return for logging):
return {"score": 1.0, "extracted_answer": model_answer, "is_correct": True}
```

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

## Environment (sdpo_vllm conda on HPC)

**HPC**: `/home/woody/iwi7/iwi7107h/`, A100-SXM4-40GB (Ampere, sm_80)
**Conda env path**: `/home/woody/iwi7/iwi7107h/conda_envs/sdpo_vllm`

| Package | Version | Notes |
|---|---|---|
| Python | 3.12 | |
| torch | 2.6.0+cu124 | confirmed working (`pip check` clean) |
| vllm | 0.8.5 | confirmed working |
| flash-attn | latest | built from source |
| ray | 2.53.0 **with** `[default]` | SDPO `requirements.txt` pins `ray[default]==2.53.0` |
| numpy | 2.1.0 | SDPO `requirements.txt` pins `numpy==2.1.0` (NOT <2.0.0) |
| transformers | 4.57.1 | SDPO `requirements.txt` |
| verl | SDPO editable | `pip install -e .` from `oneshotrlvrSDPO/SDPO/` |

**Full install sequence — confirmed working on HPC (run on login node)**:

```bash
module load python/3.12-conda
module load cuda/12.4.1

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export CONDA_PKGS_DIRS=/home/woody/iwi7/iwi7107h/conda_pkgs
export PIP_CACHE_DIR=/home/woody/iwi7/iwi7107h/.cache/pip
export XDG_CACHE_HOME=/home/woody/iwi7/iwi7107h/.cache
export TMPDIR=/home/woody/iwi7/iwi7107h/.tmp

mkdir -p /home/woody/iwi7/iwi7107h/.cache/pip
mkdir -p /home/woody/iwi7/iwi7107h/.tmp

conda deactivate || true
conda env remove -p /home/woody/iwi7/iwi7107h/conda_envs/sdpo_vllm -y 2>/dev/null || true

conda create -y -p /home/woody/iwi7/iwi7107h/conda_envs/sdpo_vllm python=3.12
conda activate /home/woody/iwi7/iwi7107h/conda_envs/sdpo_vllm

python -m pip install --upgrade pip setuptools wheel

# 1. torch
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 2. Core SDPO install: requirements.txt + editable verl fork
cd /home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/SDPO
pip install -r requirements.txt
pip install -e .

# 3. vLLM — install explicitly after SDPO deps to pin version
pip install "vllm==0.8.5"

# 4. Flash-attn — build from source AFTER vllm so it matches final torch ABI
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
pip install packaging psutil ninja
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

**Verify env**:
```python
import torch; print(torch.__version__)          # 2.6.0+cu124
import vllm; print(vllm.__version__)             # 0.8.5
import flash_attn; print(flash_attn.__version__) # compiled version
import numpy; print(numpy.__version__)           # 2.1.0
import ray; print(ray.__version__)               # 2.53.0
import verl; print(verl.__file__)                # .../oneshotrlvrSDPO/SDPO/verl/__init__.py
from verl.trainer.ppo.core_algos import compute_self_distillation_loss  # must not error
```

---

## Repo Structure

```
oneshotrlvrSDPO/
├── CLAUDE.md                         ← this file
├── README.md
├── requirements.txt
├── data/
│   ├── pi1_r128.parquet              ← 128 copies of π₁ (One-Shot-RLVR training set)
│   ├── math500.parquet               ← MATH-500 (validation)
│   └── pi1_example.json              ← π₁ raw problem for reference
├── reward/
│   └── math_reward.py                ← compute_score() for NaiveRewardManager
├── eval/
│   ├── eval_math500.py               ← standalone MATH-500 eval
│   └── eval_math500.slurm
└── scripts/
    ├── train_oneshot_sdpo.slurm      ← MAIN training job (4× A100)
    ├── run_local_test.sh             ← smoke test (Steps 1-4)
    └── setup_hpc.sh                  ← one-time HPC env setup
```

HPC layout:
```
/home/woody/iwi7/iwi7107h/
├── oneshotrlvrSDPO/        ← this repo
│   ├── SDPO/               ← lasgroup/SDPO editable install (inside repo, NOT committed)
│   ├── data/
│   ├── reward/
│   ├── scripts/
│   ├── eval/
│   └── output/             ← checkpoints, logs, rollouts (runtime, not committed)
├── conda_envs/
│   └── sdpo_vllm/          ← confirmed working conda env
└── models/
    └── Qwen2.5-Math-1.5B/  ← base model
```

**Note**: `SDPO/` lives inside the repo directory on HPC but is NOT committed to git
(it is in `.gitignore`). It is the `lasgroup/SDPO` clone used for `pip install -e .`.

---

## Running

```bash
# Pull latest on HPC
cd /home/woody/iwi7/iwi7107h/oneshotrlvrSDPO
git pull origin claude/integrate-rlvr-sdpo-dlMU5

# Smoke test (needs 1 GPU)
salloc --partition=a100 --gres=gpu:a100:1 --ntasks=1 --cpus-per-task=8 --mem=80GB --time=00:30:00
conda activate /home/woody/iwi7/iwi7107h/conda_envs/sdpo_vllm
cd /home/woody/iwi7/iwi7107h/oneshotrlvrSDPO
bash scripts/run_local_test.sh

# Production training (4× A100)
sbatch scripts/train_oneshot_sdpo.slurm
```

---

## Cross-Check: SDPO `run_local_sdpo.sh` Confirms Our Settings

From `lasgroup/SDPO/run_local_sdpo.sh` (the canonical local SDPO run script):

```bash
TRAIN_BATCH_SIZE=32       # 1 GPU; we use 128 for 4 GPUs
ROLLOUT_BATCH_SIZE=8      # = rollout.n; matches our n=8
LR=1e-5                   # their default; we use 1e-6 (more conservative)
ALPHA=0.5                 # JSD; matches ours
actor_rollout_ref.actor.optim.lr_warmup_steps=10       # matches ours
actor_rollout_ref.actor.self_distillation.distillation_topk=100  # matches ours
actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True  # matches ours
algorithm.rollout_correction.rollout_is=token          # matches ours
actor_rollout_ref.rollout.val_kwargs.n=16              # matches ours
```

Note: `run_local_sdpo.sh` uses `CONFIG_NAME="sdpo"` (which inherits `user.yaml` and
requires `TASK` + `EXPERIMENT` env vars set for CSCS cluster). We use `ppo_trainer`
instead and set all SDPO keys explicitly — same effect, no cluster dependency.

---

## Active Branch

`claude/integrate-rlvr-sdpo-dlMU5`
