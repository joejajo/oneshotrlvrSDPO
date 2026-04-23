#!/bin/bash
#SBATCH --job-name=oneshot_sdpo_p2fr
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=1
#SBATCH --time=16:00:00
#SBATCH --output=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/output_sdpo_p2fresh/logs/train_%j.out
#SBATCH --error=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/output_sdpo_p2fresh/logs/train_%j.err

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO
ONESHOT_DIR=${PROJECT_ROOT}
OUTPUT_ROOT=${PROJECT_ROOT}/output_sdpo_p2fresh
SIF=/home/woody/iwi7/iwi7107h/images/verl_vllm017_latest.sif

# Phase 1 weights from the completed twophase v1 run (steps 0-25, alpha=0).
# These are the HuggingFace-format weights saved at the end of phase 1.
# Using model.path here loads weights ONLY — no optimizer state restored,
# so lr=1e-6 below is the actual learning rate (not overridden by checkpoint).
P1_HF_CKPT=${PROJECT_ROOT}/output_sdpo_twophase/checkpoints/global_step_25/actor/huggingface

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

# ── Phase 2 fresh: alpha=0.5 JSD, lr=1e-6, 225 steps ─────────────────────────
#
# Resumes from the phase 1 HuggingFace checkpoint (weights only, no optimizer
# state). This is the corrected version of twophase v1 phase 2, which was
# running at lr=1e-5 due to resume_mode=auto restoring optimizer state.
#
# Why weights-only loading via model.path works:
#   main_ppo.py passes model.path to the actor FSDP initializer, which calls
#   AutoModelForCausalLM.from_pretrained(model.path). This loads weights and
#   initializes a fresh AdamW optimizer at the configured lr=1e-6. No optimizer
#   state dict is loaded, so lr is guaranteed to be 1e-6.
#
# Why 225 steps (not 275):
#   Measured step time from v1: ~250s/step
#   225 steps × 250s = 56,250s ≈ 15.6h < 16h allocation (90min headroom).
#   val_before_train=True adds one val pass (~82s) at step 0.
#
# What to expect:
#   Step 0 val (val_before_train): should be ~0.25 (same as phase 1 endpoint).
#   Steps 1-50: JSD mode-seeking pulls student toward teacher's confident tokens
#     in sibling-solution context. Entropy ~0.05-0.06 (stable, forward KL keeps it).
#   Steps 50-225: val should trend above 0.25 if mode-seeking signal is genuine.
#     Best case: val approaches or exceeds the ~0.368 alpha=0 ceiling.
#   Warning sign: if val drops below 0.20 before step 50, JSD is still too
#     aggressive at lr=1e-6 — cancel and try lr=5e-7.
# ─────────────────────────────────────────────────────────────────────────────

echo "================================================================" | tee -a "${LOG}"
echo "  SDPO Phase 2 (fresh optimizer)  [alpha=0.5 JSD, lr=1e-6, 225 steps]" | tee -a "${LOG}"
echo "  Loading weights from: ${P1_HF_CKPT}" | tee -a "${LOG}"
echo "  Job ID : ${SLURM_JOB_ID}" | tee -a "${LOG}"
echo "  Output : ${OUTPUT_ROOT}" | tee -a "${LOG}"
echo "================================================================" | tee -a "${LOG}"

if [ ! -d "${P1_HF_CKPT}" ]; then
    echo "ERROR: phase 1 HF checkpoint not found: ${P1_HF_CKPT}" | tee -a "${LOG}"
    exit 1
fi

command -v apptainer >/dev/null 2>&1 || { echo "apptainer not found on host"; exit 1; }

apptainer exec --nv \
    --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    --bind "/home/woody/iwi7/iwi7107h/models:/home/woody/iwi7/iwi7107h/models" \
    "${SIF}" \
    python -m verl.trainer.main_ppo \
    --config-name ppo_trainer \
    \
    actor_rollout_ref.model.path="${P1_HF_CKPT}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.trust_remote_code=True \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
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
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
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
    actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.001 \
    actor_rollout_ref.actor.self_distillation.remove_thinking_from_demonstration=false \
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=1024 \
    actor_rollout_ref.actor.self_distillation.reprompt_truncation=right \
    actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
    actor_rollout_ref.actor.self_distillation.distillation_add_tail=true \
    actor_rollout_ref.actor.self_distillation.is_clip=2.0 \
    'actor_rollout_ref.actor.self_distillation.reprompt_template="{prompt}\n\nCorrect solution: {solution}\n\nCorrectly solve the original question."' \
    'actor_rollout_ref.actor.self_distillation.solution_template="{successful_previous_attempt}"' \
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
    trainer.total_training_steps=225 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.val_before_train=True \
    trainer.log_val_generations=0 \
    trainer.experiment_name=oneshot_sdpo_p2fresh_alpha05_lr1e6 \
    trainer.project_name=oneshot_sdpo \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.resume_mode=disable \
    trainer.validation_data_dir="${ROLLOUT_DIR}" \
    trainer.rollout_data_dir="${TRAIN_ROLLOUT_DIR}" \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.default_hdfs_dir=null \
    +ray_kwargs.ray_init.object_store_memory=2147483648 \
    2>&1 | tee -a "${LOG}"
