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
ONESHOT_DIR=${PROJECT_ROOT}/oneshot_sdpo
OUTPUT_ROOT=${PROJECT_ROOT}/output

MODEL_PATH=/home/woody/iwi7/iwi7107h/models/Qwen2.5-Math-1.5B
TRAIN_PARQUET=${ONESHOT_DIR}/data/datasets/train/train.parquet
VAL_PARQUET=${ONESHOT_DIR}/data/datasets/train/val.parquet
REWARD_PATH=${ONESHOT_DIR}/reward/math_reward.py
CKPT_DIR=${OUTPUT_ROOT}/checkpoints
ROLLOUT_DIR=${OUTPUT_ROOT}/rollouts

# ── Environment ───────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sdpo

# Disable WandB entirely — we use TensorBoard only
export WANDB_MODE=disabled

# TensorBoard log directory.
# SDPO's _TensorboardAdapter reads this env var at initialisation.
# Logs land at: $TENSORBOARD_DIR/<project_name>/<experiment_name>/
export TENSORBOARD_DIR=${OUTPUT_ROOT}/tensorboard

# Make reward module importable (fallback; verl loads it via path directly)
export PYTHONPATH=${ONESHOT_DIR}:${PYTHONPATH}

# Avoid vLLM attention backend issues (used by One-Shot-RLVR's training script)
export VLLM_ATTENTION_BACKEND=XFORMERS

# Use vLLM v1 engine (used by SDPO's training/verl_training.sh)
export VLLM_USE_V1=1

mkdir -p "${OUTPUT_ROOT}/logs"
mkdir -p "${CKPT_DIR}"
mkdir -p "${ROLLOUT_DIR}"
mkdir -p "${OUTPUT_ROOT}/tensorboard"

echo "================================================================"
echo "  One-Shot-RLVR + SDPO training"
echo "  Job ID    : ${SLURM_JOB_ID}"
echo "  Model     : ${MODEL_PATH}"
echo "  Train data: ${TRAIN_PARQUET}"
echo "  Checkpoints: ${CKPT_DIR}"
echo "  TensorBoard: ${TENSORBOARD_DIR}"
echo "  Rollouts  : ${ROLLOUT_DIR}"
echo "================================================================"

# ── Launch ────────────────────────────────────────────────────────────────────
# Config base: ppo_trainer (cluster-neutral; sdpo.yaml is avoided because
# it pulls in user.yaml which has CSCS-cluster-specific defaults).
# All SDPO algorithm keys are set explicitly below.
#
# SDPO-critical keys (would otherwise default to ppo_trainer.yaml values):
#   actor_rollout_ref.actor.policy_loss.loss_mode=sdpo
#   algorithm.adv_estimator=grpo
#   algorithm.norm_adv_by_std_in_grpo=False
#   algorithm.rollout_correction.rollout_is=token
#   algorithm.rollout_correction.rollout_is_threshold=2.0
#   actor_rollout_ref.rollout.calculate_log_probs=True

python -m verl.trainer.main_ppo \
    --config-name sdpo \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.max_model_len=10240 \
    actor_rollout_ref.rollout.max_num_batched_tokens=10240 \
    +actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    +actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    +actor_rollout_ref.rollout.val_kwargs.n=1 \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
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
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=6144 \
    actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
    actor_rollout_ref.actor.self_distillation.is_clip=2.0 \
    \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    \
    data.train_files="[${TRAIN_PARQUET}]" \
    data.val_files="[${VAL_PARQUET}]" \
    data.train_batch_size=128 \
    data.val_batch_size=530 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    \
    custom_reward_function.path="${REWARD_PATH}" \
    custom_reward_function.name=compute_score \
    \
    'trainer.logger=["console","tensorboard"]' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.total_training_steps=500 \
    trainer.save_freq=50 \
    trainer.test_freq=25 \
    trainer.val_before_train=True \
    trainer.log_val_generations=0 \
    trainer.experiment_name=oneshot_sdpo_qwen25math_1.5b_pi1 \
    trainer.project_name=oneshot_sdpo \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.validation_data_dir="${ROLLOUT_DIR}" \
    trainer.default_hdfs_dir=null \
    2>&1 | tee "${OUTPUT_ROOT}/logs/train_${SLURM_JOB_ID}.log"
