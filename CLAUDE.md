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

3. **Distillation loss**: Instead of pure policy gradient, the loss is a weighted KL
   between student and teacher token distributions on the reprompted trajectories.
   `alpha=1.0` gives pure reverse KL (mode-seeking: student imitates teacher's best
   solution exactly). `alpha=0.5` gives symmetric JSD. We use `alpha=1.0`.

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
| `self_distillation.include_environment_feedback` | `True` | `true` | Rich π₁ feedback enabled |
| `self_distillation.environment_feedback_only_without_solution` | `False` | `false` | Paper-faithful: use feedback alongside solution (both in reprompt) |
| `self_distillation.remove_thinking_from_demonstration` | `True` | `false` | Qwen2.5-Math-1.5B has no `<think>` tags |
| `self_distillation.max_reprompt_len` | `10240` | `4096` | Caps reprompt + feedback length |
| `self_distillation.full_logit_distillation` | `True` | `true` | Same as default |
| `self_distillation.distillation_topk` | `100` | `20` | Matches SDPO rich_feedback experiments; less compute |
| `self_distillation.alpha` | `0.5` | `1.0` | Pure reverse KL (mode-seeking); matches SDPO rich_feedback experiments |
| `self_distillation.teacher_regularization` | `ema` | `ema` | Standard SDPO |
| `self_distillation.teacher_update_rate` | `0.05` | `0.01` | Slower EMA = more stable teacher; matches SDPO rich_feedback experiments |
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
| `max_model_len` | `null` | `8192` | prompt(1024)+response(3072)+reprompt(4096) |
| `max_num_batched_tokens` | `8192` | `16384` | 2× for concurrent sequences |
| `n` | `1` | `8` | 8 rollouts per prompt |
| `val_kwargs.n` | `1` | `1` | Greedy pass@1 (n=16 too expensive on 500 problems) |
| `temperature` | `1.0` | `0.6` | Qwen2.5-Math-1.5B recommended |
| `gpu_memory_utilization` | `0.5` | `0.6` | Safe for hybrid actor+vLLM on A100-40GB |
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
    # Returns: {"score": 0.0|1.0, "extracted_answer": str|None, "feedback": str}
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

# Our SDPO version (dict return — no is_correct!):
return {"score": 1.0, "extracted_answer": model_answer, "feedback": ""}
return {"score": 0.0, "extracted_answer": model_answer, "feedback": "<π₁ hint>"}
```

**Why no `is_correct` key**: SDPO's reward aggregator stacks `reward_extra_infos`
values into numpy arrays across the batch. Python `bool` becomes `numpy.bool_`,
which `json.dumps` rejects with `TypeError`. `score=1.0/0.0` encodes correctness.

**`feedback` key**: Paper-faithful environment feedback (SDPO Figure 4 / Table 2):
- No `\boxed{}` found → `"Your previous response did not include a final answer in \boxed{} format. Please state your answer as \boxed{your answer}."` — format correction
- Wrong answer → `"Your answer {X} is incorrect."` — names the wrong value; math analog of a coding test's `"AssertionError: expected 12.8, got 10"`
- Correct → `"feedback": ""` (empty)

With `environment_feedback_only_without_solution=false`, this feedback appears **alongside**
the teacher's successful demonstration in the reprompt (paper Table 2 full template):

```
Correct solution: [teacher's correct derivation → \boxed{12.8}]
The following is feedback from your unsuccessful earlier attempt:
Your answer 10 is incorrect.
Correctly solve the original question.
```

The self-teacher (EMA of student, conditioned on feedback) re-evaluates the original response
tokens and identifies precisely where the derivation went wrong — per-token credit assignment
as shown in Figure 4 of the paper.

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
| `ImportError: undefined symbol: cuptiActivityEnableDriverApi` | `libcupti.so.12` not found; Apptainer `--nv` rewrites `LD_LIBRARY_PATH`, dropping any host-side additions (`APPTAINERENV_LD_LIBRARY_PATH` does NOT survive `--nv` on this HPC) | Wrap the python call in `bash -c 'export LD_LIBRARY_PATH="/cupti/lib:${LD_LIBRARY_PATH}"; exec python "$@"' --` so it is set **inside** the container after `--nv` injects its paths |
| `TypeError: Object of type bool_ is not JSON serializable` | SDPO aggregates `reward_extra_infos` into numpy arrays; Python `bool` becomes `numpy.bool_` | Remove `is_correct` from `compute_score` return dict; use `score=1.0/0.0` instead |
| `fatal: not a git repository` on login node | `/home/woody` is a separate filesystem mount | `GIT_DISCOVERY_ACROSS_FILESYSTEM=1 git pull ...` or add to `~/.bashrc` |

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

# Smoke test via Apptainer (1× A100, 1 step, production-identical parameters)
salloc --partition=a100 --gres=gpu:a100:1 --ntasks=1 --cpus-per-task=8 --mem=80GB --time=00:30:00

apptainer exec --nv \
  --bind /home/woody/iwi7/iwi7107h/oneshotrlvrSDPO:/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO \
  /home/woody/iwi7/iwi7107h/images/verl_vllm017_latest.sif \
  bash /home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/scripts/run_local_test.sh

# After smoke test: inspect rollout JSONLs to verify feedback field
# (SMOKE_OUT path is printed at end of script)
python3 - <<'PY'
import json
path = "/tmp/tmp.XXXXX/train_rollouts/step_1.jsonl"  # use printed path
with open(path) as f:
    for line in f:
        d = json.loads(line)
        if d.get("score", 1) == 0.0:
            print("feedback:", d.get("feedback", "MISSING")[:100])
            break
PY

# Production training (4× A100, 500 steps)
sbatch scripts/train_oneshot_sdpo.slurm
```

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

## Active Branch

`claude/integrate-rlvr-sdpo-dlMU5`
