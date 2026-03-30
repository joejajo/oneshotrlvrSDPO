# CLAUDE.md — One-Shot-RLVR + SDPO

## Primary Goal

Implement One-Shot RLVR (Reinforcement Learning from Verifiable Rewards) using SDPO
(Self-Distilled Policy Optimization) as the training algorithm.

**Core idea**: Train Qwen2.5-Math-1.5B on a single example (π₁, 128 copies) using SDPO's
EMA teacher + JSD distillation loss, reproducing the One-Shot-RLVR result but with SDPO
instead of GRPO.

**Reference repos**:
- SDPO (verl fork, training algo): https://github.com/lasgroup/SDPO
- One-Shot-RLVR (data + concept): https://github.com/ypwang61/One-Shot-RLVR

---

## IMPORTANT: Always Cross-Check Against SDPO Repo

When answering any question about config keys, training APIs, hyperparameters,
data formats, reward functions, or script changes — **always verify against
https://github.com/lasgroup/SDPO** first. The canonical source of truth for:

- Config schema: `verl/trainer/config/ppo_trainer.yaml`, `actor/actor.yaml`,
  `rollout/rollout.yaml`, `sdpo.yaml`
- Loss implementation: `verl/trainer/ppo/core_algos.py`
  (`compute_self_distillation_loss`)
- Reward manager interface: `verl/workers/reward_manager/naive.py`
  (NaiveRewardManager — calls `compute_score(data_source, solution_str, ground_truth, extra_info)`)
- EMA teacher: `verl/workers/actor/dp_actor.py` (`SDPOActor`)
- Entry point: `verl/trainer/main_ppo.py` (launched as `python -m verl.trainer.main_ppo`)
- Example scripts: `training/verl_training.sh`, `training/experiments/`

---

## Environment

**Conda env**: `sdpo2` (on HPC at `/home/woody/iwi7/iwi7107h/`)

**Verified stack (A100, Ampere sm_80)**:
| Package | Version |
|---|---|
| Python | 3.12 |
| torch | 2.6.0+cu124 |
| vllm | 0.8.5 |
| flash-attn | 2.8.3 (cu12torch2.6cxx11abiFALSE wheel) |
| ray | 2.53.0 (without [default] extra — avoids opentelemetry conflict) |
| numpy | <2.0.0 (1.26.4) |
| verl | SDPO editable install from `/home/woody/iwi7/iwi7107h/SDPO` |

**Install flash-attn (exact wheel for torch 2.6.0)**:
```bash
pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
```

**SDPO fork install**:
```bash
cd /home/woody/iwi7/iwi7107h/SDPO
pip install -e . --no-deps
```

**Verify env is correct**:
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
├── CLAUDE.md                   ← this file
├── oneshot_sdpo/
│   ├── data/
│   │   ├── pi1_r128.parquet    ← 128-copy one-shot training set (π₁)
│   │   ├── math500.parquet     ← MATH-500 eval set
│   │   └── pi1_example.json    ← raw π₁ problem for reference
│   ├── reward/
│   │   └── math_reward.py      ← compute_score() for NaiveRewardManager
│   ├── eval/
│   │   ├── eval_math500.py     ← standalone MATH-500 eval
│   │   └── eval_math500.slurm
│   └── scripts/
│       ├── train_oneshot_sdpo.slurm  ← MAIN training job (4× A100)
│       ├── run_local_test.sh         ← smoke test (Steps 1-4)
│       └── setup_hpc.sh              ← one-time HPC env setup
└── output/                     ← created at runtime, not committed
    ├── checkpoints/
    ├── logs/
    ├── tensorboard/
    ├── rollouts/               ← val rollout JSONLs
    └── train_rollouts/         ← training rollout JSONLs
```

---

## Key Design Decisions

### Why `--config-name ppo_trainer` (not `sdpo`)?
`sdpo.yaml` inherits `user.yaml` which requires `TASK` and `EXPERIMENT` env vars
specific to the CSCS cluster. `ppo_trainer.yaml` is cluster-neutral. All SDPO-specific
keys are set explicitly via Hydra CLI overrides.

### Why `include_environment_feedback=false`?
Math reward is binary (0/1). There is no textual feedback to inject. The SDPO
`max_reprompt_len` is set to 4096 (not 10240) since we only reprompt with the
previous solution, not environment feedback.

### Why `success_reward_threshold=1.0`?
Binary reward — only reward=1.0 qualifies as a "successful" teacher demonstration.
Default in SDPO is 0.5 (partial credit use cases).

### Why `norm_adv_by_std_in_grpo=False`?
SDPO paper and `sdpo.yaml` both set this to False. Advantage normalisation by std
is off because SDPO's JSD distillation loss dominates the gradient signal.

### Why `max_model_len=8192`?
prompt(1024) + response(4096) + reprompt(4096) = 9216 worst case, but reprompt
replaces the original response so the actual max is max(prompt+response, prompt+reprompt)
= 1024+4096 = 5120 for single-turn, but vllm needs to fit the full reprompt context.
Set to 8192 to be safe. `max_num_batched_tokens=16384` allows 2 concurrent sequences.

### Why `val_kwargs.n=16`?
SDPO experiment scripts use n=16 validation samples per question for stable pass@1
estimation. Default in rollout.yaml is n=1.

### Why `data.train_batch_size=128` not 32?
SDPO's `sdpo.yaml` uses 32 (single GPU experiments). We have 4 GPUs, so 128 gives
equivalent per-GPU batch size of 32.

---

## SDPO Config Defaults (from lasgroup/SDPO)

Critical defaults to know when setting Hydra overrides:

**ppo_trainer.yaml**:
- `adv_estimator: gae` → we override to `grpo`
- `norm_adv_by_std_in_grpo: True` → we override to `False`
- `logger: ["console", "wandb"]` → we override to `["console","tensorboard"]`

**actor/actor.yaml**:
- `policy_loss.loss_mode: "vanilla"` → we override to `sdpo`
- `self_distillation.success_reward_threshold: 0.5` → we override to `1.0`
- `self_distillation.include_environment_feedback: True` → we override to `false`
- `self_distillation.remove_thinking_from_demonstration: True` → we override to `false`
- `self_distillation.max_reprompt_len: 10240` → we override to `4096`
- `use_kl_loss: false` → keep default (no KL penalty)
- `ppo_mini_batch_size: 256` → we use `128`

**rollout/rollout.yaml**:
- `max_model_len: null` → we set `8192`
- `max_num_batched_tokens: 8192` → we set `16384`
- `calculate_log_probs: False` → we override to `True` (required for SDPO loss)
- `n: 1` → we set `8`
- `val_kwargs.n: 1` → we set `16`
- `tensor_model_parallel_size: 2` → we set `1`

---

## Reward Function Interface

`reward/math_reward.py` exposes `compute_score` matching SDPO's NaiveRewardManager:

```python
def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth,
    extra_info=None,
) -> dict:
    # Returns: {"score": 0.0|1.0, "extracted_answer": str|None, "is_correct": bool}
```

Returns a dict so that `extracted_answer` and `is_correct` are collected into
`reward_extra_infos` and written to the validation JSONL.

---

## Parquet Schema

Both `pi1_r128.parquet` and `math500.parquet` must have columns:
```
data_source, prompt, ability, reward_model, extra_info
```
- `prompt`: list of chat messages `[{"role": "user", "content": "..."}]`
- `reward_model`: dict with `ground_truth` key
- `extra_info`: dict (may be empty)

---

## HPC Paths

```
/home/woody/iwi7/iwi7107h/
├── oneshotrlvrSDPO/        ← this repo
├── SDPO/                   ← lasgroup/SDPO editable install
├── models/
│   └── Qwen2.5-Math-1.5B/  ← base model
└── output/                 ← checkpoints, logs, rollouts
```

---

## Running

```bash
# On HPC — pull latest
cd /home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/oneshot_sdpo
git pull origin claude/integrate-rlvr-sdpo-dlMU5

# Smoke test (interactive GPU node)
salloc --partition=a100 --gres=gpu:a100:1 --ntasks=1 --cpus-per-task=8 --mem=80GB --time=00:30:00
conda activate sdpo2
bash scripts/run_local_test.sh

# Production training
sbatch scripts/train_oneshot_sdpo.slurm
```

---

## Branch

Active development branch: `claude/integrate-rlvr-sdpo-dlMU5`
