#!/bin/bash
#SBATCH --job-name=oneshot_sdpo_7b
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=1
#SBATCH --time=16:00:00
#SBATCH --output=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/output_sdpo_7b/logs/train_%j.out
#SBATCH --error=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/output_sdpo_7b/logs/train_%j.err

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO
ONESHOT_DIR=${PROJECT_ROOT}
OUTPUT_ROOT=${PROJECT_ROOT}/output_sdpo_7b
SIF=/home/woody/iwi7/iwi7107h/images/verl_vllm017_latest.sif

MODEL_PATH=/home/woody/iwi7/iwi7107h/models/Qwen2.5-Math-7B

TRAIN_PARQUET=${ONESHOT_DIR}/data/pi1_r128.parquet
EVAL_PARQUET=${ONESHOT_DIR}/data/math500.parquet

REWARD_PATH=${ONESHOT_DIR}/reward/math_reward.py
CKPT_DIR=${OUTPUT_ROOT}/checkpoints
ROLLOUT_DIR=${OUTPUT_ROOT}/rollouts
TRAIN_ROLLOUT_DIR=${OUTPUT_ROOT}/train_rollouts

# ── Environment ────────────────────────────────────────────────────────────────
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

mkdir -p "${OUTPUT_ROOT}/logs" "${CKPT_DIR}" "${ROLLOUT_DIR}" \
         "${TRAIN_ROLLOUT_DIR}" "${OUTPUT_ROOT}/tensorboard" "${RAY_TMPDIR}"

# ── Qwen2.5-Math-7B SDPO on 2× A100-40GB ─────────────────────────────────────
#
# Memory budget (7B bf16, FSDP×2, optimizer_offload=True):
#
#   vLLM (TP=2, gpu_memory_utilization=0.28):
#     Each GPU reserves 0.28 × 40 = 11.2 GB for vLLM.
#     7B weights in bf16 = 14 GB; with TP=2 each GPU holds 7 GB of weights.
#     KV cache per GPU: 11.2 - 7 = 4.2 GB.
#     Qwen2.5-7B: 32 layers, 8 KV heads, head_dim=128, bf16.
#     With TP=2: 4 KV heads/GPU → 64 KB/token.
#     4.2 GB / 64 KB ≈ 65,000 token slots (more than enough for 8 seqs × 4096).
#     tensor_model_parallel_size MUST be 2 — 7B (14 GB) > 11.2 GB on 1 GPU.
#
#   FSDP actor (optimizer_offload=True):
#     Sharded params: 14 GB / 2 = 7 GB per GPU.
#     Adam fp32 states: 56 GB total → CPU (optimizer_offload=True).
#     Peak during forward all-gather: 14 GB params briefly on each GPU.
#     Available per GPU after vLLM: 40 - 11.2 = 28.8 GB.
#     Peak actor usage: 14 GB (full params) + activations (small at batch=8)
#                       < 28.8 GB available. ✓
#
#   Ref/teacher (param_offload=True):
#     Kept on CPU, loaded shard-by-shard during teacher forward. ✓
#
#   max_reprompt_len = max_model_len - max_response_length = 4096 - 2048 = 2048.
#   This ensures teacher input (reprompt + response) always ≤ max_model_len.
#
#   ppo_max_token_len_per_gpu=1024: conservative for 7B. With dynamic batching
#   and ppo_mini_batch_size=8 = train_batch_size, there is exactly 1 gradient
#   update per step (no inner-loop overfitting on 1-shot data).
#
# Training design:
#   alpha=0.5 (JSD): paper Condition A default. SDPO paper shows JSD is more
#   stable than pure forward/reverse KL for on-policy training.
#   success_reward_threshold=1.0: binary reward, only perfect = teacher.
#   include_environment_feedback=false: Condition A (scalar reward only).
#   teacher_update_rate=0.05: paper default (active EMA).
#   Condition D (rich feedback) can be enabled later once Condition A is stable.
# ─────────────────────────────────────────────────────────────────────────────

echo "================================================================"
echo "  One-Shot-RLVR SDPO  [Qwen2.5-Math-7B, alpha=0.5 JSD, 2×A100]"
echo "  Job ID      : ${SLURM_JOB_ID}"
echo "  Model       : ${MODEL_PATH}"
echo "  Train data  : ${TRAIN_PARQUET}"
echo "  Eval data   : ${EVAL_PARQUET}"
echo "  Checkpoints : ${CKPT_DIR}"
echo "  TensorBoard : ${TENSORBOARD_DIR}"
echo "  Container   : ${SIF}"
echo "================================================================"

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
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.28 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.max_num_seqs=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=1024 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
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
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=2048 \
    actor_rollout_ref.actor.self_distillation.reprompt_truncation=right \
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
    data.train_batch_size=8 \
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
    trainer.total_training_steps=200 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.val_before_train=True \
    trainer.log_val_generations=0 \
    trainer.experiment_name=oneshot_sdpo_qwen25math_7b_pi1 \
    trainer.project_name=oneshot_sdpo \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.resume_mode=disable \
    trainer.validation_data_dir="${ROLLOUT_DIR}" \
    trainer.rollout_data_dir="${TRAIN_ROLLOUT_DIR}" \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.default_hdfs_dir=null \
    +ray_kwargs.ray_init.object_store_memory=2147483648 \
    2>&1 | tee "${OUTPUT_ROOT}/logs/train_${SLURM_JOB_ID}.log"
