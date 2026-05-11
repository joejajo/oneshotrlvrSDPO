#!/bin/bash
#SBATCH --job-name=thesis_2gpu_nofb
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=1
#SBATCH --time=18:00:00
#SBATCH --output=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/output_thesis_2gpu/logs/train_%j.out
#SBATCH --error=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/output_thesis_2gpu/logs/train_%j.err

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO
ONESHOT_DIR=${PROJECT_ROOT}
OUTPUT_ROOT=${PROJECT_ROOT}/output_thesis_2gpu
SIF=/home/woody/iwi7/iwi7107h/images/verl_vllm017_latest.sif

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
export PYTHONWARNINGS="ignore:Unsupported processor type,ignore:ANTLR runtime and generated code versions disagree"
export TMPDIR=/tmp
export RAY_TMPDIR=/tmp/r${SLURM_JOB_ID}
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1

ulimit -n 65536 2>/dev/null || true

mkdir -p "${OUTPUT_ROOT}/logs" "${CKPT_DIR}" "${ROLLOUT_DIR}" "${TRAIN_ROLLOUT_DIR}" "${OUTPUT_ROOT}/tensorboard" "${RAY_TMPDIR}"

echo "================================================================"
echo "  SDPO training  [Thesis — 2×A100, no rich feedback, 1000 steps]"
echo "  Job ID      : ${SLURM_JOB_ID}"
echo "  Model       : ${MODEL_PATH}"
echo "  Train data  : ${TRAIN_PARQUET}"
echo "  Eval data   : ${EVAL_PARQUET}"
echo "  Checkpoints : ${CKPT_DIR}"
echo "  TensorBoard : ${TENSORBOARD_DIR}"
echo "  Val rollouts: ${ROLLOUT_DIR}"
echo "  Trn rollouts: ${TRAIN_ROLLOUT_DIR}"
echo "  Container   : ${SIF}"
echo "================================================================"

# ── Thesis — 2×A100, no rich feedback, Paper §3 hyperparameters, 1000 steps ──
#
# Identical to thesis_sdpo.slurm (4-GPU) but without TP=2:
#   tensor_model_parallel_size=1  → single-GPU vLLM (1.5B fits on 1 GPU)
#   n_gpus_per_node=2             → actor FSDP across 2 GPUs
#   max_num_batched_tokens=16384  → 16384 (single GPU KV budget)
#   max_num_seqs=32               → half of 4-GPU version
#   object_store_memory=2GB       → 2 GPUs, half the 4-GPU budget
#
# Timing estimate: ~140s/step × 440 steps ≈ 17.1h; wall=18h (partition cap)
#
# Memory per GPU (2×A100-40GB):
#   vLLM: 0.4 × 40GB = 16GB reserved (TP=1, single GPU)
#   Teacher logits peak: 7168 tok × 151936 × 4B = 4.05 GB (1 seq/micro-batch)
#   Actor FSDP (2-way shard): ~1.5 GB params + ~3 GB Adam states/GPU
#   Total: ~24.5 GB/GPU < 40 GB ✓
#
# Hyperparameters (Paper §3 exact, no rich feedback):
#   batch=32, temp=1.0, lr=1e-5, norm_adv=True, alpha=0.5 (JSD), topk=100
#   include_environment_feedback=false → scalar reward only, no verifier text
#   ppo_mini_batch_size=256 → 256/256 = 1 optimizer step per training step
#   max_reprompt_len=3072 = max_response_length → full sibling solution, no truncation
#   val_kwargs.temperature=0.6, do_sample=True → stochastic validation
#   save_freq=20, max_actor_ckpt_to_keep=1 → keep only latest checkpoint
#   test_freq=20 → 50 validation points over 1000 steps
#   rollout_data_freq=20 → training JSONL with teacher_input + distilled flag every 20 steps
# ─────────────────────────────────────────────────────────────────────────────

# ── Launch ────────────────────────────────────────────────────────────────────
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
    actor_rollout_ref.model.dtype=bfloat16 \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.max_num_seqs=32 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
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
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=3072 \
    actor_rollout_ref.actor.self_distillation.reprompt_truncation=left \
    actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
    actor_rollout_ref.actor.self_distillation.distillation_add_tail=true \
    actor_rollout_ref.actor.self_distillation.is_clip=2.0 \
    'actor_rollout_ref.actor.self_distillation.reprompt_template="{prompt}\n\nCorrect solution: {solution}\n\nCorrectly solve the original question."' \
    'actor_rollout_ref.actor.self_distillation.solution_template="{successful_previous_attempt}"' \
    \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    \
    data.train_files="[${TRAIN_PARQUET}]" \
    data.val_files="[${EVAL_PARQUET}]" \
    data.train_batch_size=32 \
    data.val_batch_size=64 \
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
    trainer.total_training_steps=440 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.val_before_train=True \
    trainer.log_val_generations=0 \
    trainer.experiment_name=thesis_2gpu_nofb \
    trainer.project_name=oneshot_sdpo \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.resume_mode=disable \
    trainer.validation_data_dir="${ROLLOUT_DIR}" \
    trainer.rollout_data_dir="${TRAIN_ROLLOUT_DIR}" \
    trainer.rollout_data_freq=20 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.default_hdfs_dir=null \
    +ray_kwargs.ray_init.object_store_memory=2147483648 \
    2>&1 | tee "${OUTPUT_ROOT}/logs/train_${SLURM_JOB_ID}.log"
