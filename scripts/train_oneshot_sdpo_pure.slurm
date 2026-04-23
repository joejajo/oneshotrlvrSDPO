#!/bin/bash
#SBATCH --job-name=oneshot_sdpo_pure
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=1
#SBATCH --time=16:00:00
#SBATCH --output=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/output_sdpo_pure/logs/train_%j.out
#SBATCH --error=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/output_sdpo_pure/logs/train_%j.err

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO
ONESHOT_DIR=${PROJECT_ROOT}
OUTPUT_ROOT=${PROJECT_ROOT}/output_sdpo_pure
SIF=/home/woody/iwi7/iwi7107h/images/verl_vllm017_latest.sif

MODEL_PATH=/home/woody/iwi7/iwi7107h/models/Qwen2.5-Math-1.5B

TRAIN_PARQUET=${ONESHOT_DIR}/data/pi1_r128.parquet
EVAL_PARQUET=${ONESHOT_DIR}/data/math500.parquet

REWARD_PATH=${ONESHOT_DIR}/reward/math_reward.py
CKPT_DIR=${OUTPUT_ROOT}/checkpoints
ROLLOUT_DIR=${OUTPUT_ROOT}/rollouts
TRAIN_ROLLOUT_DIR=${OUTPUT_ROOT}/train_rollouts
LOG=${OUTPUT_ROOT}/logs/train_${SLURM_JOB_ID}.log

# ── Environment ───────────────────────────────────────────────────────────────
export WANDB_MODE=disabled
export WANDB_DISABLED=true

export TENSORBOARD_DIR=${OUTPUT_ROOT}/tensorboard

SDPO_DIR=${PROJECT_ROOT}/SDPO
PKGS_DIR=${PROJECT_ROOT}/pkgs
export PYTHONPATH="${ONESHOT_DIR}:${SDPO_DIR}:${PKGS_DIR}:${PYTHONPATH:-}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"

_NV="/usr/local/lib/python3.12/dist-packages/nvidia"
export APPTAINERENV_LD_PRELOAD="\
${_NV}/cuda_cupti/lib/libcupti.so.12:\
${_NV}/nccl/lib/libnccl.so.2:\
${_NV}/cublas/lib/libcublas.so.12:\
${_NV}/cublas/lib/libcublasLt.so.12"

unset VLLM_ATTENTION_BACKEND
unset ROCR_VISIBLE_DEVICES
export VLLM_DISABLE_COMPILE_CACHE=1
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export APPTAINERENV_UV_THREADPOOL_SIZE=4
export PYTHONWARNINGS="ignore:Unsupported processor type"
export TMPDIR=/tmp
export RAY_TMPDIR=/tmp/r${SLURM_JOB_ID}

ulimit -n 65536 2>/dev/null || true
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1

mkdir -p "${OUTPUT_ROOT}/logs" "${CKPT_DIR}" "${ROLLOUT_DIR}" "${TRAIN_ROLLOUT_DIR}" "${OUTPUT_ROOT}/tensorboard" "${RAY_TMPDIR}"

# ── Pure SDPO — 100% paper match (Condition A / §3 scalar reward) ────────────
#
# Paper source: lasgroup/SDPO, experiments/.../run_sdpo.sh + actor.yaml defaults
# Condition A = no environment feedback (scalar reward only)
#
# SDPO mechanism:
#   1. EMA teacher: teacher = (1-τ)*teacher + τ*student, τ=0.05
#   2. Reprompt failed rollouts with a sibling's successful solution
#   3. Distillation loss = JSD between student and teacher token distributions
#      on reprompted trajectories (α=0.5 = mode-covering + mode-seeking balance)
#   4. IS correction with rollout_is=token, clip at 2.0
#   5. No use_kl_loss, no KL anchor to frozen reference (SDPO design)
#   6. Pure distillation: GRPO advantages are computed (adv_estimator=grpo is
#      required to avoid critic init) but NOT used in the loss
#
# Hardware adaptation (2× A100-40GB, paper used 8× H100):
#   - n_gpus_per_node=2 (vs paper's 8)
#   - gpu_memory_utilization=0.35 (vs paper's 0.5; reduced from 0.4 to give
#     headroom for teacher forward — first run OOMed trying to alloc 9 GiB
#     for logits in torch.logsumexp during actor update)
#   - tensor_model_parallel_size=1 (1.5B fits on one GPU)
#   - max_model_len=8192 (vs paper's 18944; fits our 40GB VRAM)
#   - max_reprompt_len=4608 (vs paper's 10240; paper used 8×H100-80GB;
#     on 2×A100-40GB the teacher forward over ppo_mini_batch_size=32 sequences
#     of up to max_model_len=8192 tokens creates a logits tensor of ~9 GiB
#     that exceeds available GPU memory; 4608 keeps teacher seqs ≤7680 tokens)
#   - ppo_max_token_len_per_gpu=4096 (dynamic batching)
#   - ref.fsdp_config.param_offload=true (CPU-offload teacher to save VRAM)
#
# Paper hyperparameters (NOT touched — 100% match):
#   alpha=0.5                           (JSD — Condition A default)
#   teacher_update_rate=0.05            (paper default — NOT our prior 0.001)
#   distillation_topk=100               (Condition A default)
#   success_reward_threshold=0.5        (paper default)
#   max_reprompt_len=4608               (hardware-adapted from paper's 10240)
#   reprompt_truncation=right           (paper uses right)
#   dont_reprompt_on_self_success=true  (paper default)
#   include_environment_feedback=false  (Condition A = no feedback)
#   full_logit_distillation=true
#   distillation_add_tail=true
#   is_clip=2.0
#   reprompt_template / solution_template / feedback_template = actor.yaml defaults
#     → reprompt="{prompt}{solution}{feedback}\n\nCorrectly solve the original question."
#     → solution="\n\nCorrect solution:\n\n{successful_previous_attempt}"
#     → feedback="" (include_environment_feedback=false drops the feedback_template)
#   lr=1e-5, lr_warmup_steps=10         (paper §3)
#   entropy_coeff=0                     (paper §3 — NOT our prior 0.001)
#   ppo_mini_batch_size=32              (paper §3)
#   norm_adv_by_std_in_grpo=false       (paper §3)
#
# One-Shot-RLVR adaptation (same training data as π₁ reproduction):
#   train_batch_size=128 (One-Shot-RLVR paper — 32 in SDPO §3 is too small
#                          for the 1-shot-128-copies setup to be comparable)
#   rollout.n=8, temperature=0.6
#   max_prompt=1024, max_response=3072
#
# What this run tests:
#   Does pure SDPO with paper-exact hyperparameters, no KL anchor, and
#   no post-hoc safety measures reproduce MATH-500 gains on π₁?
#   Expected trajectory:
#     Step 0 val ~0.30-0.36 (base Qwen2.5-Math-1.5B)
#     Steps 5-10 phase transition (success_fraction 35%→99% as model learns π₁)
#     If paper is right: val improves → 0.50+ by step 100
#     If 1-shot + SDPO is unstable: val collapses to <0.20 during transition
#       (we've seen this in earlier runs; this is the control experiment to
#        establish the baseline before trying trust-region or anchor variants)
#
# Step time: ~250s (measured). 300 steps × 250s ≈ 20.8h — doesn't fit 16h.
# Setting total_training_steps=220 so we fit: 220 × 250s ≈ 15.3h.
# val_before_train + 44 val passes × ~82s ≈ 62 min ≈ total ≈ 16h max.
# test_freq=5 for early collapse detection; save_freq=20 to keep disk manageable.
# ─────────────────────────────────────────────────────────────────────────────

echo "================================================================" | tee -a "${LOG}"
echo "  Pure SDPO — 100% paper match (Condition A / §3 scalar reward)" | tee -a "${LOG}"
echo "  alpha=0.5 JSD, τ=0.05 EMA, topk=100, lr=1e-5, NO feedback, NO KL" | tee -a "${LOG}"
echo "  Job ID : ${SLURM_JOB_ID}" | tee -a "${LOG}"
echo "  Model  : ${MODEL_PATH}" | tee -a "${LOG}"
echo "  Output : ${OUTPUT_ROOT}" | tee -a "${LOG}"
echo "================================================================" | tee -a "${LOG}"

command -v apptainer >/dev/null 2>&1 || { echo "apptainer not found on host"; exit 1; }

apptainer exec --nv \
    --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    --bind "/home/woody/iwi7/iwi7107h/models:/home/woody/iwi7/iwi7107h/models" \
    "${SIF}" \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.max_num_seqs=32 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.actor.policy_loss.loss_mode=sdpo \
    actor_rollout_ref.actor.self_distillation.alpha=0.5 \
    actor_rollout_ref.actor.self_distillation.full_logit_distillation=true \
    actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
    actor_rollout_ref.actor.self_distillation.distillation_add_tail=true \
    actor_rollout_ref.actor.self_distillation.teacher_regularization=ema \
    actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.05 \
    actor_rollout_ref.actor.self_distillation.success_reward_threshold=0.5 \
    actor_rollout_ref.actor.self_distillation.is_clip=2.0 \
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=4608 \
    actor_rollout_ref.actor.self_distillation.reprompt_truncation=right \
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=true \
    actor_rollout_ref.actor.self_distillation.include_environment_feedback=false \
    actor_rollout_ref.actor.self_distillation.environment_feedback_only_without_solution=false \
    actor_rollout_ref.actor.self_distillation.remove_thinking_from_demonstration=false \
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
    data.val_batch_size=500 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.trust_remote_code=True \
    \
    custom_reward_function.path="${REWARD_PATH}" \
    custom_reward_function.name=compute_score \
    \
    'trainer.logger=["console","tensorboard"]' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.total_epochs=9999 \
    trainer.total_training_steps=220 \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.val_before_train=True \
    trainer.log_val_generations=0 \
    trainer.experiment_name=oneshot_sdpo_pure_paper \
    trainer.project_name=oneshot_sdpo \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.resume_mode=disable \
    trainer.validation_data_dir="${ROLLOUT_DIR}" \
    trainer.rollout_data_dir="${TRAIN_ROLLOUT_DIR}" \
    trainer.max_actor_ckpt_to_keep=5 \
    trainer.default_hdfs_dir=null \
    +ray_kwargs.ray_init.object_store_memory=2147483648 \
    2>&1 | tee -a "${LOG}"
