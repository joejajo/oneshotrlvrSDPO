#!/bin/bash
#SBATCH --job-name=oneshot_sdpo_condD_a1
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=1
#SBATCH --time=16:00:00
#SBATCH --output=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/output_sdpo_condD_alpha0/logs/train_%j.out
#SBATCH --error=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/output_sdpo_condD_alpha0/logs/train_%j.err

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO
ONESHOT_DIR=${PROJECT_ROOT}
OUTPUT_ROOT=${PROJECT_ROOT}/output_sdpo_condD_alpha0
SIF=/home/woody/iwi7/iwi7107h/images/verl_vllm017_latest.sif

MODEL_PATH=/home/woody/iwi7/iwi7107h/models/Qwen2.5-Math-1.5B

TRAIN_PARQUET=${ONESHOT_DIR}/data/pi1_r128.parquet
EVAL_PARQUET=${ONESHOT_DIR}/data/math500.parquet

REWARD_PATH=${ONESHOT_DIR}/reward/math_reward.py
CKPT_DIR=${OUTPUT_ROOT}/checkpoints
ROLLOUT_DIR=${OUTPUT_ROOT}/rollouts
TRAIN_ROLLOUT_DIR=${OUTPUT_ROOT}/train_rollouts

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

echo "================================================================"
echo "  One-Shot-RLVR SDPO  [cond-D rich feedback, alpha=1.0, lr=1e-6]"
echo "  Job ID      : ${SLURM_JOB_ID}"
echo "  Model       : ${MODEL_PATH}"
echo "  Train data  : ${TRAIN_PARQUET}"
echo "  Eval data   : ${EVAL_PARQUET}"
echo "  Checkpoints : ${CKPT_DIR}"
echo "  TensorBoard : ${TENSORBOARD_DIR}"
echo "  Container   : ${SIF}"
echo "================================================================"

# ── Condition D + alpha=0.0 (rich feedback + forward KL) ─────────────────────
#
# What failed before and why:
#   alpha0 run (lr=1e-5, τ=0.001, no feedback): peaked at val=0.368 at step 40.
#   Root cause: π₁ success_sample_fraction hit 1.0 at step 20 → distillation
#   signal collapsed to ~0 → no more learning → post-peak regression to 0.334.
#   The useful learning window was only 20 steps.
#
# Why adding rich feedback helps break the ceiling:
#   With include_environment_feedback=true, the reprompt for every failed rollout
#   includes the verifier's error message: "Your answer X is incorrect. The
#   correct answer is 12.8." This makes the distillation target richer — the
#   teacher distribution is conditioned on BOTH the correct solution AND the
#   explicit error signal, giving the student stronger credit-assignment signal
#   on which tokens led to the wrong answer.
#
# Three fixes from the alpha0 run:
#   1. lr=1e-6 (was 1e-5): slows π₁ saturation from step 20 to ~step 200,
#      extending the useful distillation window 10×.
#   2. τ=0.01 (was 0.001): teacher absorbs 63% of student improvements by step
#      100, providing progressively richer distillation targets as training
#      progresses (compounding self-improvement loop).
#   3. Rich feedback (new): error-specific context in every reprompt gives
#      the student explicit error-correction signal even when the group has a
#      sibling success available.
#
# Critical alignment fixes for feedback to actually reach the reprompt:
#   - include_environment_feedback=true: enables feedback collection
#   - environment_feedback_only_without_solution=false: WITHOUT this, the
#     actor.yaml default suppresses feedback whenever a sibling solution
#     exists (which is almost always with n=8). This flag MUST be false.
#   - No reprompt_template/solution_template overrides: the alpha0 script
#     used custom templates that omitted {feedback}. Here we use actor.yaml
#     defaults which include {feedback} in the reprompt.
#   - reprompt_truncation=left (was right): preserves the feedback and
#     "Correctly solve" instruction at the tail of the reprompt when
#     truncation occurs; right-truncation would silently cut the feedback.
#   - max_reprompt_len=2048 (was 1024): solution + feedback together can
#     exceed 1024 tokens; 2048 accommodates π₁ solution (~500 tok) +
#     feedback (~30 tok) + original prompt (~200 tok) comfortably.
#
# alpha=1.0 (reverse KL) — paper-correct condition D:
#   Reverse KL is mode-seeking: chases the teacher's most confident tokens
#   directly. With rich feedback the teacher is MORE concentrated (it knows
#   both the correct solution AND the specific error), making the signal
#   sharper and more directed than alpha=0.0. This is the strongest
#   distillation signal achievable and the best shot at val > 0.60.
#   Collapse risk managed by lr=1e-6: previous collapses at lr=1e-5 had
#   phase-transition update ~3e-5; at lr=1e-6 the same spike = ~3e-6
#   (10× smaller). Kill signal: actor/entropy < 0.02 by step 10 = collapse.
#
# Expected behaviour:
#   - No entropy collapse (proven with alpha=0.0 in prior run)
#   - Saturation at ~step 200 instead of step 20 (10× longer learning window)
#   - val-core/simplerl/math500/reward/mean@1 peak > 0.50 (target: > 0.60)
#
# Timing: 200 steps × ~242s/step ≈ 13.5h — fits within 16h allocation.
# ─────────────────────────────────────────────────────────────────────────────

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
    actor_rollout_ref.actor.optim.lr_warmup_steps=5 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.actor.policy_loss.loss_mode=sdpo \
    actor_rollout_ref.actor.self_distillation.include_environment_feedback=true \
    actor_rollout_ref.actor.self_distillation.environment_feedback_only_without_solution=false \
    actor_rollout_ref.actor.self_distillation.success_reward_threshold=1.0 \
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=true \
    actor_rollout_ref.actor.self_distillation.full_logit_distillation=true \
    actor_rollout_ref.actor.self_distillation.alpha=1.0 \
    actor_rollout_ref.actor.self_distillation.teacher_regularization=ema \
    actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.01 \
    actor_rollout_ref.actor.self_distillation.remove_thinking_from_demonstration=false \
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=2048 \
    actor_rollout_ref.actor.self_distillation.reprompt_truncation=left \
    actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
    actor_rollout_ref.actor.self_distillation.distillation_add_tail=true \
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
    trainer.total_training_steps=200 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.val_before_train=True \
    trainer.log_val_generations=0 \
    trainer.experiment_name=oneshot_sdpo_qwen25math_1.5b_pi1_condD_alpha1_lr1e6 \
    trainer.project_name=oneshot_sdpo \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.resume_mode=disable \
    trainer.validation_data_dir="${ROLLOUT_DIR}" \
    trainer.rollout_data_dir="${TRAIN_ROLLOUT_DIR}" \
    trainer.max_actor_ckpt_to_keep=5 \
    trainer.default_hdfs_dir=null \
    +ray_kwargs.ray_init.object_store_memory=2147483648 \
    2>&1 | tee "${OUTPUT_ROOT}/logs/train_${SLURM_JOB_ID}.log"
