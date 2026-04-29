#!/bin/bash
#SBATCH --job-name=oneshot_sdpo_7b_paper_rope
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks=1
#SBATCH --time=16:00:00
#SBATCH --output=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/output_sdpo_7b_paper_rope/logs/train_%j.out
#SBATCH --error=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/output_sdpo_7b_paper_rope/logs/train_%j.err

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO
ONESHOT_DIR=${PROJECT_ROOT}
OUTPUT_ROOT=${PROJECT_ROOT}/output_sdpo_7b_paper_rope
SIF=/home/woody/iwi7/iwi7107h/images/verl_vllm017_latest.sif

MODEL_PATH=/home/woody/iwi7/iwi7107h/models/Qwen2.5-Math-7B-RoPE-300k

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

# ── Original SDPO parameters (no rich feedback) ───────────────────────────────
#
# Hyperparameters taken verbatim from lasgroup/SDPO:
#   experiments/rich_feedback/run_sdpo.sh  — override values
#   verl/trainer/config/sdpo.yaml          — base config values
#
# Parameter source table:
#   ppo_mini_batch_size=1      run_sdpo.sh  (sdpo.yaml has 32; sh overrides to 1)
#   lr=1e-6                    run_sdpo.sh  (sdpo.yaml has 1e-5; sh overrides)
#   lr_warmup_steps=0          run_sdpo.sh
#   alpha=1.0                  run_sdpo.sh  (reverse KL; original uses 1.0)
#   distillation_topk=20       run_sdpo.sh
#   teacher_update_rate=0.01   run_sdpo.sh
#   dont_reprompt_on_self_success=True  run_sdpo.sh
#   val_kwargs.n=4             run_sdpo.sh
#   n=8                        sdpo.yaml + run_sdpo.sh
#   train_batch_size=32        sdpo.yaml + run_sdpo.sh
#   calculate_log_probs=True   sdpo.yaml
#   rollout_is=token           sdpo.yaml + run_sdpo.sh
#   rollout_is_threshold=2.0   sdpo.yaml
#   adv_estimator=grpo         sdpo.yaml
#   norm_adv_by_std_in_grpo=False  sdpo.yaml
#   is_clip=2.0                sdpo.yaml
#   loss_mode=sdpo             sdpo.yaml
#
# The only departure from run_sdpo.sh:
#   include_environment_feedback=false  (run_sdpo.sh targets rich feedback;
#                                        here we test scalar reward only)
#
# Memory budget (7B bf16, FSDP×4, optimizer_offload=True):
#   vLLM TP=2, gpu_memory_utilization=0.5 → 20 GB per TP GPU.
#   7B weights / TP=2 = 7 GB per GPU. KV cache = 13 GB per TP GPU.
#   FSDP shard: 14 GB / 4 = 3.5 GB per GPU on CPU (optimizer) + GPU (params).
#
#   Model: open-r1/Qwen2.5-Math-7B-RoPE-300k
#     rope_theta=300000, max_position_embeddings=32768 (extended from base 4096).
#     Same weights as Qwen2.5-Math-7B; only RoPE config changed.
#
#   max_model_len=6144: reprompt(2400)+response(2048)=4448 ≤ 6144 ✓
#   max_reprompt_len=2400: π₁ reprompt worst-case = 2206 tok < 2400 → no truncation ✓
#   reprompt_truncation=right: original SDPO default; irrelevant since 2400>2206.
#
#   ppo_max_token_len_per_gpu=4096: teacher logit matrix ≤ 4096×152064×2 ≈ 1.25 GB;
#   logsumexp peak ≈ 2.5 GB total. Fits. ✓
#   (ppo_mini_batch_size=1 means single sequences anyway; 4096 is safe upper bound)
# ─────────────────────────────────────────────────────────────────────────────

echo "================================================================"
echo "  One-Shot-RLVR SDPO  [Qwen2.5-Math-7B, original SDPO params]"
echo "  Condition    : scalar reward (no rich feedback)"
echo "  Job ID       : ${SLURM_JOB_ID}"
echo "  Model        : ${MODEL_PATH}"
echo "  Train data   : ${TRAIN_PARQUET}"
echo "  Eval data    : ${EVAL_PARQUET}"
echo "  Checkpoints  : ${CKPT_DIR}"
echo "  TensorBoard  : ${TENSORBOARD_DIR}"
echo "  Container    : ${SIF}"
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
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.max_model_len=6144 \
    actor_rollout_ref.rollout.max_num_seqs=32 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    \
    actor_rollout_ref.actor.policy_loss.loss_mode=sdpo \
    actor_rollout_ref.actor.self_distillation.include_environment_feedback=false \
    actor_rollout_ref.actor.self_distillation.success_reward_threshold=1.0 \
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=true \
    actor_rollout_ref.actor.self_distillation.full_logit_distillation=true \
    actor_rollout_ref.actor.self_distillation.alpha=1.0 \
    actor_rollout_ref.actor.self_distillation.teacher_regularization=ema \
    actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.01 \
    actor_rollout_ref.actor.self_distillation.remove_thinking_from_demonstration=false \
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=2400 \
    actor_rollout_ref.actor.self_distillation.reprompt_truncation=right \
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
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.total_epochs=9999 \
    trainer.total_training_steps=100 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.val_before_train=True \
    trainer.log_val_generations=0 \
    trainer.experiment_name=oneshot_sdpo_qwen25math_7b_rope300k_paper \
    trainer.project_name=oneshot_sdpo \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.resume_mode=disable \
    trainer.validation_data_dir="${ROLLOUT_DIR}" \
    trainer.rollout_data_dir="${TRAIN_ROLLOUT_DIR}" \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.default_hdfs_dir=null \
    +ray_kwargs.ray_init.object_store_memory=2147483648 \
    2>&1 | tee "${OUTPUT_ROOT}/logs/train_${SLURM_JOB_ID}.log"
