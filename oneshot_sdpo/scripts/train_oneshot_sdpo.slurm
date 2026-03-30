#!/bin/bash
#SBATCH --job-name=oneshot_sdpo
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200GB
#SBATCH --time=16:00:00
#SBATCH --output=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/output/logs/train_%j.out
#SBATCH --error=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/output/logs/train_%j.err

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO
ONESHOT_DIR=${PROJECT_ROOT}
OUTPUT_ROOT=${PROJECT_ROOT}/output

MODEL_PATH=/home/woody/iwi7/iwi7107h/models/Qwen2.5-Math-1.5B

# Parquet files committed directly to the repo (from One-Shot-RLVR)
# pi1_r128.parquet  : 128 copies of π₁ — the one-shot training set
# math500.parquet   : MATH-500 — used for both online validation and standalone eval
TRAIN_PARQUET=${ONESHOT_DIR}/data/pi1_r128.parquet
EVAL_PARQUET=${ONESHOT_DIR}/data/math500.parquet

REWARD_PATH=${ONESHOT_DIR}/reward/math_reward.py
CKPT_DIR=${OUTPUT_ROOT}/checkpoints
ROLLOUT_DIR=${OUTPUT_ROOT}/rollouts
TRAIN_ROLLOUT_DIR=${OUTPUT_ROOT}/train_rollouts

# ── Environment ───────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sdpo2

export WANDB_MODE=disabled
export WANDB_DISABLED=true

# TensorBoard: SDPO's _TensorboardAdapter reads TENSORBOARD_DIR at init.
# Logs land at: $TENSORBOARD_DIR/<project_name>/<experiment_name>/
export TENSORBOARD_DIR=${OUTPUT_ROOT}/tensorboard

# Make reward module importable (verl also loads it via path directly)
# Also include SDPO repo root for editable install consistency
SDPO_DIR=/home/woody/iwi7/iwi7107h/SDPO
export PYTHONPATH=${ONESHOT_DIR}:${SDPO_DIR}:${PYTHONPATH}

# vLLM settings (from SDPO verl_training.sh)
# V1 engine uses FlashAttention by default — do NOT set VLLM_ATTENTION_BACKEND
unset VLLM_ATTENTION_BACKEND
unset ROCR_VISIBLE_DEVICES
export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1

# AF_UNIX socket paths are capped at 107 bytes on Linux.
# Ray appends ~63 chars, so RAY_TMPDIR must be ≤ 44 chars. Use short /tmp path.
export RAY_TMPDIR=/tmp/r${SLURM_JOB_ID}

# Raise open-file-descriptor limit — Ray + vLLM open many sockets.
ulimit -n 65536 2>/dev/null || true

# Ray plasma store uses /dev/shm by default.
# SLURM often caps /dev/shm at 64–512 MB, causing the Raylet to crash.
# Allow plasma to fall back to /tmp if /dev/shm is too small.
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1

mkdir -p "${OUTPUT_ROOT}/logs" "${CKPT_DIR}" "${ROLLOUT_DIR}" "${TRAIN_ROLLOUT_DIR}" "${OUTPUT_ROOT}/tensorboard" "${RAY_TMPDIR:-/tmp}"

echo "================================================================"
echo "  One-Shot-RLVR + SDPO training"
echo "  Job ID      : ${SLURM_JOB_ID}"
echo "  Model       : ${MODEL_PATH}"
echo "  Train data  : ${TRAIN_PARQUET}"
echo "  Eval data   : ${EVAL_PARQUET}"
echo "  Checkpoints : ${CKPT_DIR}"
echo "  TensorBoard : ${TENSORBOARD_DIR}"
echo "  Val rollouts: ${ROLLOUT_DIR}"
echo "  Trn rollouts: ${TRAIN_ROLLOUT_DIR}"
echo "================================================================"

# ── Launch ────────────────────────────────────────────────────────────────────
# Config base: ppo_trainer
#   Using --config-name ppo_trainer (NOT sdpo).
#   sdpo.yaml inherits user.yaml which requires TASK and EXPERIMENT env vars
#   specific to the CSCS cluster. ppo_trainer.yaml is cluster-neutral.
#   All SDPO-specific keys are set explicitly below.
#
# SDPO keys (absent from ppo_trainer.yaml, must be set explicitly):
#   actor_rollout_ref.actor.policy_loss.loss_mode=sdpo
#   algorithm.adv_estimator=grpo
#   algorithm.norm_adv_by_std_in_grpo=False
#   algorithm.rollout_correction.rollout_is=token
#   algorithm.rollout_correction.rollout_is_threshold=2.0
#   actor_rollout_ref.rollout.calculate_log_probs=True
#   actor_rollout_ref.actor.self_distillation.*
#
# val_kwargs: pre-declared in rollout.yaml with correct defaults
#   (temperature=0, n=1, do_sample=False) — no override needed.

python -m verl.trainer.main_ppo \
    --config-name ppo_trainer \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.trust_remote_code=True \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20000 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.actor.policy_loss.loss_mode=sdpo \
    actor_rollout_ref.actor.self_distillation.include_environment_feedback=false \
    actor_rollout_ref.actor.self_distillation.success_reward_threshold=1.0 \
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=true \
    actor_rollout_ref.actor.self_distillation.full_logit_distillation=true \
    actor_rollout_ref.actor.self_distillation.alpha=0.5 \
    actor_rollout_ref.actor.self_distillation.teacher_regularization=ema \
    actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.05 \
    actor_rollout_ref.actor.self_distillation.remove_thinking_from_demonstration=false \
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=4096 \
    actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
    actor_rollout_ref.actor.self_distillation.is_clip=2.0 \
    \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    \
    data.train_files="[${TRAIN_PARQUET}]" \
    data.val_files="[${EVAL_PARQUET}]" \
    data.train_batch_size=128 \
    data.val_batch_size=530 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.trust_remote_code=True \
    \
    custom_reward_function.path="${REWARD_PATH}" \
    custom_reward_function.name=compute_score \
    \
    'trainer.logger=["console","tensorboard"]' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.total_training_steps=500 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.val_before_train=True \
    trainer.log_val_generations=0 \
    trainer.experiment_name=oneshot_sdpo_qwen25math_1.5b_pi1 \
    trainer.project_name=oneshot_sdpo \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.resume_mode=auto \
    trainer.validation_data_dir="${ROLLOUT_DIR}" \
    trainer.rollout_data_dir="${TRAIN_ROLLOUT_DIR}" \
    trainer.default_hdfs_dir=null \
    +ray_kwargs.ray_init.object_store_memory=2147483648 \
    2>&1 | tee "${OUTPUT_ROOT}/logs/train_${SLURM_JOB_ID}.log"
