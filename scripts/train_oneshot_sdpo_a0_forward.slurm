#!/bin/bash
#SBATCH --job-name=oneshot_sdpo_a0fw
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=1
#SBATCH --time=16:00:00
#SBATCH --output=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/output_sdpo_a0_forward/logs/train_%j.out
#SBATCH --error=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/output_sdpo_a0_forward/logs/train_%j.err

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO
ONESHOT_DIR=${PROJECT_ROOT}
OUTPUT_ROOT=${PROJECT_ROOT}/output_sdpo_a0_forward
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

# ── Custom SDPO — alpha=0 forward KL, tau=0.1, topk=500 ───────────────────────
#
# User-designed config (NOT paper-match). Characteristics:
#   - alpha=0.0 (pure forward KL — no mode-seeking, collapse-proof)
#   - teacher_update_rate=0.1 (faster EMA drift than paper's 0.05)
#   - distillation_topk=500 (richer distillation signal than paper's 100)
#   - max_reprompt_len=5120 (sized exactly to fit max_model_len=8192:
#                            prompt(1024) + reprompt(5120) + response(3072))
#   - remove_thinking_from_demonstration=true (paper default; no-op for
#                                              Qwen2.5-Math-1.5B which has
#                                              no <think> tags anyway)
#   - success_reward_threshold=0.5 (paper default — equivalent to 1.0 for
#                                   our binary 0/1 reward)
#   - dont_reprompt_on_self_success=true (paper default)
#   - include_environment_feedback=false (no π₁ answer contamination)
#   - NO use_kl_loss (SDPO design — EMA teacher is the anchor)
#
# Token budget math for 2× A100-40GB:
#   Per-sequence: prompt(1024) + reprompt(5120) + response(3072) = 9216 MAX
#     BUT max_model_len=8192 caps total — reprompt gets truncated right
#     when prompt+reprompt+response would exceed 8192
#   Effective per-sequence length: ≤ 8192 tokens
#
#   ppo_max_token_len_per_gpu=32768 (= 4 × 8192-token sequences per micro-batch)
#   Full-vocab logits at peak (bf16):
#     student (grad): 32768 × 151936 × 2B = 9.9 GB
#     teacher (no_grad): 9.9 GB (released after top-500 extraction)
#   Per-GPU memory budget:
#     FSDP model shard (1.5B/2): 1.5 GB
#     AdamW optim states (1.5B×12B/2): 9 GB
#     Gradients: 1.5 GB
#     vLLM reserved (0.35×40): 14 GB
#     Logits+activations peak: ~13 GB
#     Total: ~39 GB — TIGHT. gpu_memory_utilization lowered from 0.4 → 0.35
#     to give +2 GB safety margin. Increase to 0.4 if stable.
#
# Training recipe choices (not specified in user's config, filled in):
#   lr=1e-5, lr_warmup_steps=10 (paper §3 + safe because alpha=0 is non-collapsing)
#   entropy_coeff=0 (paper default)
#   ppo_mini_batch_size=32 (paper §3)
#   use_dynamic_bsz=True (uses ppo_max_token_len_per_gpu for micro-batching)
#   norm_adv_by_std_in_grpo=False (paper §3)
#
# Expected behavior:
#   Step 0 val ~0.30-0.36 (base Qwen2.5-Math-1.5B)
#   Steps 5-10 phase transition; alpha=0 should survive without collapse
#     (forward KL cannot cause gradient concentration on π₁ tokens)
#   Post-transition: teacher catches up to student faster (τ=0.1) →
#     distillation target keeps evolving → sustained learning signal
#   Target: val stays ≥ 0.30 throughout, trends toward 0.40+ by step 100
#
# Step time est: ~300s (larger micro-batches, more compute). 220 × 300s ≈ 18h,
# which exceeds 16h. Setting total_training_steps=180 to fit with val passes.
# ─────────────────────────────────────────────────────────────────────────────

echo "================================================================" | tee -a "${LOG}"
echo "  Custom SDPO  [alpha=0 forward KL | tau=0.1 | topk=500 | max_reprompt=5120]" | tee -a "${LOG}"
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
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.actor.policy_loss.loss_mode=sdpo \
    actor_rollout_ref.actor.self_distillation.alpha=0.0 \
    actor_rollout_ref.actor.self_distillation.full_logit_distillation=true \
    actor_rollout_ref.actor.self_distillation.distillation_topk=500 \
    actor_rollout_ref.actor.self_distillation.distillation_add_tail=true \
    actor_rollout_ref.actor.self_distillation.teacher_regularization=ema \
    actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.1 \
    actor_rollout_ref.actor.self_distillation.success_reward_threshold=0.5 \
    actor_rollout_ref.actor.self_distillation.is_clip=2.0 \
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=5120 \
    actor_rollout_ref.actor.self_distillation.reprompt_truncation=right \
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=true \
    actor_rollout_ref.actor.self_distillation.remove_thinking_from_demonstration=true \
    actor_rollout_ref.actor.self_distillation.include_environment_feedback=false \
    actor_rollout_ref.actor.self_distillation.environment_feedback_only_without_solution=false \
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
    trainer.total_training_steps=180 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.val_before_train=True \
    trainer.log_val_generations=0 \
    trainer.experiment_name=oneshot_sdpo_a0_forward_tau01_topk500 \
    trainer.project_name=oneshot_sdpo \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.resume_mode=disable \
    trainer.validation_data_dir="${ROLLOUT_DIR}" \
    trainer.rollout_data_dir="${TRAIN_ROLLOUT_DIR}" \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.default_hdfs_dir=null \
    +ray_kwargs.ray_init.object_store_memory=2147483648 \
    2>&1 | tee -a "${LOG}"
