#!/bin/bash
#SBATCH --job-name=oneshot_sdpo_ema_rkl_rich_wide
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=1
#SBATCH --time=16:00:00
#SBATCH --output=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/output_ema_rkl_rich_wide/logs/train_%j.out
#SBATCH --error=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/output_ema_rkl_rich_wide/logs/train_%j.err

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO
ONESHOT_DIR=${PROJECT_ROOT}
OUTPUT_ROOT=${PROJECT_ROOT}/output_ema_rkl_rich_wide
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
# One-time pkgs install (login node, not here):
#   pip install --target=${PROJECT_ROOT}/pkgs latex2sympy2 regex antlr4-python3-runtime==4.9.3
PKGS_DIR=${PROJECT_ROOT}/pkgs
export PYTHONPATH="${ONESHOT_DIR}:${SDPO_DIR}:${PKGS_DIR}:${PYTHONPATH:-}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"

# Force container CUDA libs; host versions are older (missing symbols for cupti/nccl/cublas)
_NV="/usr/local/lib/python3.12/dist-packages/nvidia"
export APPTAINERENV_LD_PRELOAD="\
${_NV}/cuda_cupti/lib/libcupti.so.12:\
${_NV}/nccl/lib/libnccl.so.2:\
${_NV}/cublas/lib/libcublas.so.12:\
${_NV}/cublas/lib/libcublasLt.so.12"

# VLLM_USE_V1 is unrecognised in vLLM 0.17.0; VLLM_DISABLE_COMPILE_CACHE avoids torch.compile crash
unset VLLM_ATTENTION_BACKEND
unset ROCR_VISIBLE_DEVICES
export VLLM_DISABLE_COMPILE_CACHE=1
export PYTHONUNBUFFERED=1

# Incompatible Triton at ~/.local overrides container's — block it
export PYTHONNOUSERSITE=1

# Concurrent AgentLoopWorker spawns race in glibc getenv without this → SIGSEGV
export APPTAINERENV_UV_THREADPOOL_SIZE=4

export PYTHONWARNINGS="ignore:Unsupported processor type"
export TMPDIR=/tmp

# AF_UNIX socket limit: Ray appends ~63 chars; keep RAY_TMPDIR ≤ 44 chars
export RAY_TMPDIR=/tmp/r${SLURM_JOB_ID}

ulimit -n 65536 2>/dev/null || true
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1

mkdir -p "${OUTPUT_ROOT}/logs" "${CKPT_DIR}" "${ROLLOUT_DIR}" "${TRAIN_ROLLOUT_DIR}" \
         "${OUTPUT_ROOT}/tensorboard" "${RAY_TMPDIR}"

echo "================================================================"
echo "  One-Shot-RLVR + SDPO  [EMA teacher + rKL, rich feedback, wide reprompt]"
echo "  Job        : ${SLURM_JOB_ID}"
echo "  Model      : ${MODEL_PATH}"
echo "  Train      : ${TRAIN_PARQUET}"
echo "  Eval       : ${EVAL_PARQUET}"
echo "  Output     : ${OUTPUT_ROOT}"
echo "  Container  : ${SIF}"
echo "================================================================"

# ── EMA teacher + rKL + rich feedback + wide reprompt (best settings) ─────────
#
# Improves on ema_rkl_rich_paper with two additional changes:
#
# 1. Wide reprompt (50/50 context split):
#      max_reprompt_len=2048   (was 1024)
#      max_response_length=2048  (was 3072)
#      teacher_input = reprompt(≤2048) + response(≤2048) = ≤4096 = max_position_embeddings
#
#    Why: with max_reprompt_len=1024, sibling solution budget is only ~718 tokens
#    (after question ~256 + feedback ~30 + template ~20). π₁ correct solutions
#    are 200–800 tokens so ~1/3 were being left-truncated — the beginning of the
#    sibling solution was silently dropped, giving the teacher an incomplete example.
#    At 2048 reprompt, solution budget grows to ~1742 tokens — nearly all π₁
#    solutions fit completely, so teacher sees full question + full solution + full
#    feedback without any truncation.
#
#    Trade-off: student responses capped at 2048 instead of 3072. π₁ solutions
#    are typically 200–800 tokens; the extra 1024 tokens in the original budget
#    were needed only for edge cases. Watch clip_ratio in TensorBoard — if it
#    stays near zero the cap is fine.
#
# 2. Slower EMA (teacher_update_rate=0.05, was 0.01):
#    Paper's own rich feedback experiment uses τ=0.01. With rich feedback the
#    reprompt already provides a strong learning signal — teacher does not need
#    to track student as fast. Slower EMA → more stable target distribution →
#    less risk of teacher chasing student into bad regions.
#    Has zero effect on wall-clock speed (EMA update is a CPU scalar op, ~2s/step).
#
# Full settings:
#   alpha=1.0 (reverse KL, mode-seeking)
#   distillation_topk=20   (paper rich feedback value)
#   norm_adv_by_std_in_grpo=False  (sdpo.yaml default; unnormalized advantages)
#   teacher_update_rate=0.05
#   max_reprompt_len=2048, max_response_length=2048
#   lr=1e-5, warmup=10, batch=32, mini_batch=32, temp=1.0
#   include_environment_feedback=true, environment_feedback_only_without_solution=false
#
# Template (SDPO default actor.yaml — NOT overridden):
#   {prompt}
#   Correct solution:
#   {successful_previous_attempt}
#   The following is feedback from your unsuccessful earlier attempt:
#   {feedback_raw}
#   Correctly solve the original question.
# ─────────────────────────────────────────────────────────────────────────────

command -v apptainer >/dev/null 2>&1 || { echo "apptainer not found"; exit 1; }

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
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.max_num_seqs=128 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
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
    actor_rollout_ref.actor.self_distillation.include_environment_feedback=true \
    actor_rollout_ref.actor.self_distillation.environment_feedback_only_without_solution=false \
    actor_rollout_ref.actor.self_distillation.success_reward_threshold=1.0 \
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=true \
    actor_rollout_ref.actor.self_distillation.full_logit_distillation=true \
    actor_rollout_ref.actor.self_distillation.alpha=1.0 \
    actor_rollout_ref.actor.self_distillation.teacher_regularization=ema \
    actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.05 \
    actor_rollout_ref.actor.self_distillation.remove_thinking_from_demonstration=false \
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=2048 \
    actor_rollout_ref.actor.self_distillation.reprompt_truncation=left \
    actor_rollout_ref.actor.self_distillation.distillation_topk=20 \
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
    data.train_batch_size=32 \
    data.val_batch_size=500 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
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
    trainer.total_training_steps=300 \
    trainer.save_freq=10 \
    trainer.test_freq=20 \
    trainer.val_before_train=True \
    trainer.log_val_generations=0 \
    trainer.experiment_name=oneshot_sdpo_qwen25math_1.5b_pi1_ema_rkl_rich_wide \
    trainer.project_name=oneshot_sdpo \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.resume_mode=disable \
    trainer.validation_data_dir="${ROLLOUT_DIR}" \
    trainer.rollout_data_dir="${TRAIN_ROLLOUT_DIR}" \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.default_hdfs_dir=null \
    +ray_kwargs.ray_init.num_cpus=${SLURM_CPUS_ON_NODE} \
    +ray_kwargs.ray_init.object_store_memory=2147483648 \
    2>&1 | tee "${OUTPUT_ROOT}/logs/train_${SLURM_JOB_ID}.log"
