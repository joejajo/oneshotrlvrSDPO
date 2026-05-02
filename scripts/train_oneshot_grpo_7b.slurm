#!/bin/bash
#SBATCH --job-name=oneshot_grpo_7b_rope
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks=1
#SBATCH --time=16:00:00
#SBATCH --output=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/output_grpo_7b_rope/logs/train_%j.out
#SBATCH --error=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/output_grpo_7b_rope/logs/train_%j.err
#SBATCH --requeue

set -euo pipefail

# ── GPU health check ───────────────────────────────────────────────────────────
gpu_health_check() {
    local node="$(hostname)"
    local nv_out
    if ! nv_out=$(nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader 2>&1); then
        echo "ERROR: nvidia-smi failed on ${node}: ${nv_out}"
        return 1
    fi
    echo "----- GPU enumeration on ${node} -----"
    echo "${nv_out}"
    echo "--------------------------------------"
    local bad=0
    while IFS=, read -r idx name util mem; do
        idx=$(echo "${idx}" | xargs)
        util=$(echo "${util}" | xargs)
        if [[ "${util}" == *"N/A"* ]] || [[ "${util}" == *"ERR"* ]] || [[ -z "${util// }" ]]; then
            echo "ERROR: GPU ${idx} on ${node} unhealthy (utilization='${util}')"
            bad=$((bad + 1))
        fi
    done <<< "${nv_out}"
    return ${bad}
}

if ! gpu_health_check; then
    echo "================================================================"
    echo "  BAD NODE DETECTED on $(hostname) at $(date)"
    echo "  Requeueing job ${SLURM_JOB_ID} so SLURM allocates a different node"
    echo "================================================================"
    scontrol requeue "${SLURM_JOB_ID}" || echo "scontrol requeue failed; exiting with TEMPFAIL anyway"
    exit 75
fi
echo "GPU health check passed on $(hostname)."

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO
ONESHOT_DIR=${PROJECT_ROOT}
OUTPUT_ROOT=${PROJECT_ROOT}/output_grpo_7b_rope
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
export VLLM_SLEEP_LEVEL=2
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export APPTAINERENV_UV_THREADPOOL_SIZE=4
export PYTHONWARNINGS="ignore:Unsupported processor type"
export TMPDIR=/tmp
export RAY_TMPDIR=/tmp/r${SLURM_JOB_ID}

ulimit -n 65536 2>/dev/null || true
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p "${OUTPUT_ROOT}/logs" "${CKPT_DIR}" "${ROLLOUT_DIR}" \
         "${TRAIN_ROLLOUT_DIR}" "${OUTPUT_ROOT}/tensorboard" "${RAY_TMPDIR}"

# ── Qwen2.5-Math-7B GRPO + KL on 4 x A100-40GB ───────────────────────────────
#
# Algorithm: plain GRPO (adv_estimator=grpo, no SDPO distillation),
#   with KL regularization against the frozen reference model (kl_loss_coef=0.001).
#   This is the exact One-Shot-RLVR recipe from ypwang61/One-Shot-RLVR, ported
#   to 7B and 4 GPUs. The 1.5B SDPO condition-A run collapsed at step 80 due
#   to entropy shrinkage with no KL anchor; this run restores the anchor.
#
# Memory budget (7B bf16, FSDP1 x4, bf16 optimizer state, no offload):
#   vLLM (TP=2, gpu_memory_utilization=0.5): 20 GB/GPU for vLLM weights+KV.
#   FSDP actor (bf16, 4 GPUs): ~24 GB/GPU peak during update.
#   Ref (bf16, param_offload=True): on CPU, loaded shard-by-shard for KL.
#   No teacher forward — GRPO uses frozen ref only for KL, not distillation.
#   vLLM sleeps (VLLM_SLEEP_LEVEL=2) during actor update.
#
# use_kl_loss=True is only valid with loss_mode=vanilla (GRPO).
#   SDPO blocks it at main_ppo.py:133 because it uses the ref slot for the
#   EMA teacher and can't simultaneously use it for KL.
#
# calculate_log_probs=False: GRPO does not need stored rollout log probs
#   (no importance sampling correction needed without SDPO).
# -----------------------------------------------------------------------------

echo "================================================================"
echo "  One-Shot-RLVR — GRPO + KL (kl_coef=0.001)"
echo "  Model : Qwen2.5-Math-7B-RoPE-300k   Hardware : 4× A100-40GB"
echo "================================================================"
echo "  SLURM job     : ${SLURM_JOB_ID}  name=${SLURM_JOB_NAME}"
echo "  Node          : $(hostname)   started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "----------------------------------------------------------------"
echo "  Paths"
echo "    model       : ${MODEL_PATH}"
echo "    train       : ${TRAIN_PARQUET}"
echo "    eval        : ${EVAL_PARQUET}"
echo "    checkpoints : ${CKPT_DIR}"
echo "    tensorboard : ${TENSORBOARD_DIR}"
echo "    log         : ${OUTPUT_ROOT}/logs/train_${SLURM_JOB_ID}.log"
echo "----------------------------------------------------------------"
echo "  Rollout"
echo "    n=8  temperature=0.6  top_p=1.0  top_k=-1"
echo "    TP=2  gpu_mem_util=0.5  max_model_len=6144"
echo "    max_num_seqs=32  max_num_batched_tokens=32768"
echo "    val: n=1  do_sample=False  temperature=0  (greedy)"
echo "----------------------------------------------------------------"
echo "  Actor (GRPO)"
echo "    lr=1e-5  warmup_steps=20"
echo "    use_kl_loss=True  kl_loss_coef=0.001"
echo "    ppo_mini_batch_size=32  ppo_max_token_len_per_gpu=4096"
echo "    use_dynamic_bsz=True"
echo "    strategy=fsdp  model_dtype=bfloat16  use_torch_compile=False"
echo "----------------------------------------------------------------"
echo "  Algorithm"
echo "    adv_estimator=grpo  norm_adv_by_std=False"
echo "    (no IS correction — GRPO does not need it)"
echo "----------------------------------------------------------------"
echo "  Data"
echo "    train_batch_size=32  max_prompt_len=1024  max_resp_len=2048"
echo "    val_batch_size=500"
echo "----------------------------------------------------------------"
echo "  Training"
echo "    total_steps=300  save_freq=10  test_freq=5"
echo "    val_before_train=True  log_val_generations=16"
echo "    ref param_offload=True"
echo "================================================================"

command -v apptainer >/dev/null 2>&1 || { echo "apptainer not found on host"; exit 1; }

apptainer exec --nv \
    --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    --bind "/home/woody/iwi7/iwi7107h/models:/home/woody/iwi7/iwi7107h/models" \
    "${SIF}" \
    python -m verl.trainer.main_ppo \
    --config-name ppo_trainer \
    hydra.verbose=true \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.model.trust_remote_code=True \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.rollout.max_model_len=6144 \
    actor_rollout_ref.rollout.max_num_seqs=32 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
    actor_rollout_ref.actor.use_torch_compile=False \
    \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.use_torch_compile=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
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
    trainer.total_training_steps=300 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.val_before_train=True \
    trainer.log_val_generations=16 \
    trainer.experiment_name=oneshot_grpo_qwen25math_7b_rope300k_pi1 \
    trainer.project_name=oneshot_sdpo \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.resume_mode=disable \
    trainer.validation_data_dir="${ROLLOUT_DIR}" \
    trainer.rollout_data_dir="${TRAIN_ROLLOUT_DIR}" \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.default_hdfs_dir=null \
    +ray_kwargs.ray_init.object_store_memory=2147483648 \
    2>&1 | tee "${OUTPUT_ROOT}/logs/train_${SLURM_JOB_ID}.log"
