#!/bin/bash
# Pre-flight smoke test for One-Shot-RLVR + SDPO.
#
# Steps 1-3 run on the LOGIN NODE (no GPU needed).
# Step 4 (one training iteration) requires a GPU — run inside an interactive
# GPU allocation or on a compute node:
#
#   salloc --partition=a100 --gres=gpu:a100:4 --ntasks=1 --cpus-per-task=16 \
#          --mem=200GB --time=00:30:00
#
# Usage:
#   conda activate /home/woody/iwi7/iwi7107h/conda_envs/sdpo_vllm
#   cd /home/woody/iwi7/iwi7107h/oneshotrlvrSDPO
#   bash scripts/run_local_test.sh
#
# Override the model path if needed:
#   MODEL_PATH=/path/to/model bash scripts/run_local_test.sh

MODEL_PATH="${MODEL_PATH:-/home/woody/iwi7/iwi7107h/models/Qwen2.5-Math-1.5B}"

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONESHOT_DIR="$(dirname "${SCRIPT_DIR}")"

echo "================================================================"
echo "  One-Shot-RLVR + SDPO — pre-flight smoke test"
echo "  Working directory: ${ONESHOT_DIR}"
echo "================================================================"

cd "${ONESHOT_DIR}"

# ── Step 1: Validate committed parquet files ──────────────────────────────────
echo ""
echo ">>> Step 1: parquet file validation"

# Check files exist and start with PAR1 magic bytes
for f in data/pi1_r128.parquet data/math500.parquet; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: ${f} not found" >&2
        exit 1
    fi
    magic=$(head -c 4 "${f}" | cat -v)
    if [[ "${magic}" != "PAR1" ]]; then
        echo "ERROR: ${f} does not look like a parquet file (bad magic bytes)" >&2
        exit 1
    fi
    echo "  ${f} : exists, PAR1 magic bytes OK"
done

python - <<'EOF'
import pandas as pd

train = pd.read_parquet("data/pi1_r128.parquet")
assert len(train) == 128, f"Expected 128 rows in pi1_r128.parquet, got {len(train)}"
expected_cols = {"data_source", "prompt", "ability", "reward_model", "extra_info"}
assert expected_cols.issubset(set(train.columns)), (
    f"Missing columns in pi1_r128.parquet: {expected_cols - set(train.columns)}"
)
print(f"  pi1_r128.parquet  : {len(train)} rows  columns={train.columns.tolist()}  OK")

math500 = pd.read_parquet("data/math500.parquet")
assert len(math500) == 500, f"Expected 500 rows in math500.parquet, got {len(math500)}"
assert expected_cols.issubset(set(math500.columns)), (
    f"Missing columns in math500.parquet: {expected_cols - set(math500.columns)}"
)
print(f"  math500.parquet   : {len(math500)} rows  columns={math500.columns.tolist()}  OK")
EOF
echo ">>> Step 1 passed."

# ── Step 2: Reward self-test ──────────────────────────────────────────────────
echo ""
echo ">>> Step 2: reward/math_reward.py self-test"
python reward/math_reward.py
echo ">>> Step 2 passed."

# ── Step 3: Key imports ───────────────────────────────────────────────────────
echo ""
echo ">>> Step 3: import verification"
python - <<'EOF'
import verl
import vllm
import ray
import torch
import numpy
import flash_attn
from torch.utils.tensorboard import SummaryWriter
import sympy
from verl.trainer.ppo.core_algos import compute_self_distillation_loss

print("  verl          :", verl.__file__)
print("  torch         :", torch.__version__)    # expect 2.6.0+cu124
print("  vllm          :", vllm.__version__)     # expect 0.8.5
print("  ray           :", ray.__version__)       # expect 2.53.0
print("  numpy         :", numpy.__version__)     # expect 2.1.0
print("  flash_attn    :", flash_attn.__version__)
print("  tensorboard   : ok")
print("  sympy         :", sympy.__version__)
print("  compute_self_distillation_loss : ok")

# Verify versions match expected sdpo_vllm env
assert torch.__version__.startswith("2.6.0"), f"Expected torch 2.6.0, got {torch.__version__}"
assert vllm.__version__ == "0.8.5", f"Expected vllm 0.8.5, got {vllm.__version__}"
assert numpy.__version__ == "2.1.0", f"Expected numpy 2.1.0, got {numpy.__version__}"
print("  Version checks: all OK")
EOF
echo ">>> Step 3 passed."

# ── Step 4: One training iteration (requires GPU) ─────────────────────────────
echo ""
echo ">>> Step 4: one SDPO training iteration (end-to-end)"

if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "  WARNING: no GPU detected — skipping Step 4."
    echo "  Re-run inside an interactive GPU allocation to test training."
    echo ""
    echo "================================================================"
    echo "  === Steps 1-3 passed — request a GPU node for Step 4 ==="
    echo "================================================================"
    exit 0
fi

module load cuda/12.4.1
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
unset TRANSFORMERS_CACHE

N_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "  Detected ${N_GPUS} GPU(s)"

SMOKE_OUT=$(mktemp -d)
trap 'rm -rf "${SMOKE_OUT}"' EXIT

# Kill any stale Ray instance from a previous failed run.
ray stop --force 2>/dev/null || true
sleep 2

# AF_UNIX socket paths are capped at 107 bytes on Linux.
# Ray appends ~63 chars, so RAY_TMPDIR must be ≤ 44 chars.
export RAY_TMPDIR=/tmp/rs$$
mkdir -p "${RAY_TMPDIR}"
trap 'rm -rf "${SMOKE_OUT}" "${RAY_TMPDIR}"' EXIT

# Raise open-file-descriptor limit — Ray + vLLM open many sockets.
ulimit -n 65536 2>/dev/null || true

# Allow plasma store to fall back to /tmp if /dev/shm is too small.
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1

python -m verl.trainer.main_ppo \
    --config-name ppo_trainer \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.max_model_len=2048 \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.policy_loss.loss_mode=sdpo \
    actor_rollout_ref.actor.self_distillation.include_environment_feedback=false \
    actor_rollout_ref.actor.self_distillation.success_reward_threshold=1.0 \
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=true \
    actor_rollout_ref.actor.self_distillation.full_logit_distillation=true \
    actor_rollout_ref.actor.self_distillation.alpha=0.5 \
    actor_rollout_ref.actor.self_distillation.teacher_regularization=ema \
    actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.05 \
    actor_rollout_ref.actor.self_distillation.remove_thinking_from_demonstration=false \
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=1024 \
    actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
    actor_rollout_ref.actor.self_distillation.is_clip=2.0 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    data.train_files="[${ONESHOT_DIR}/data/pi1_r128.parquet]" \
    data.val_files="[${ONESHOT_DIR}/data/math500.parquet]" \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.trust_remote_code=True \
    custom_reward_function.path="${ONESHOT_DIR}/reward/math_reward.py" \
    custom_reward_function.name=compute_score \
    'trainer.logger=["console"]' \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.nnodes=1 \
    trainer.total_training_steps=1 \
    trainer.save_freq=0 \
    trainer.test_freq=0 \
    trainer.val_before_train=False \
    trainer.log_val_generations=0 \
    trainer.experiment_name=smoke_test \
    trainer.project_name=oneshot_sdpo \
    trainer.default_local_dir="${SMOKE_OUT}/checkpoints" \
    trainer.default_hdfs_dir=null \
    +ray_kwargs.ray_init.object_store_memory=1073741824

echo ">>> Step 4 passed."

echo ""
echo "================================================================"
echo "  === All checks passed — safe to sbatch ==="
echo "================================================================"
