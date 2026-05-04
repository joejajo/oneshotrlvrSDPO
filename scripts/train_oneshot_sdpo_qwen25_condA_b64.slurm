#!/bin/bash
#SBATCH --job-name=sdpo_q25_1b_A2
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=1
#SBATCH --time=16:00:00
#SBATCH --output=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/output_sdpo_qwen25_condA_b64/logs/train_%j.out
#SBATCH --error=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/output_sdpo_qwen25_condA_b64/logs/train_%j.err
#SBATCH --requeue

set -euo pipefail

# ── GPU health check ──────────────────────────────────────────────────────────
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
OUTPUT_ROOT=${PROJECT_ROOT}/output_sdpo_qwen25_condA_b64
SIF=/home/woody/iwi7/iwi7107h/images/verl_vllm017_latest.sif

MODEL_PATH=/home/woody/iwi7/iwi7107h/models/Qwen2.5-Math-1.5B

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

# ── Qwen2.5-Math-1.5B SDPO Condition A — larger batch (b64) on 2× A100-40GB ──
#
# Identical to train_oneshot_sdpo_qwen25_condA.slurm except:
#   train_batch_size: 32 → 64
#
# Why larger batch helps SDPO on 1.5B:
#   At ~36% MATH-500 baseline, each group of 8 rollouts yields ~3 successes.
#   batch=32  → 32 groups → ~96 successful rollouts/step  → teacher solutions
#   batch=64  → 64 groups → ~192 successful rollouts/step → 2× richer teachers
#   More groups also stabilise the GRPO advantage estimate (less gradient noise).
#   batch=128 (original One-Shot-RLVR) exceeds 16h budget; batch=64 is the
#   sweet spot: ~2× slower per step, 300 steps ≈ 10h — fits within 16h.
#
# JSD loss (alpha=0.5) is bounded [0, log 2] — confirmed from core_algos.py:
#   add_tail() ensures top-k distributions sum to 1 (tail = 1 − Σp_topk).
#   M = 0.5·p_student + 0.5·p_teacher is a proper mixture distribution.
#   KL(P‖M) ≤ log(2) because M ≥ 0.5·P → p/m ≤ 2 → log(p/m) ≤ log 2.
#   JSD = 0.5·KL(student‖M) + 0.5·KL(teacher‖M) ≤ log 2. ✓
#   Also symmetric: JSD(P,Q) = JSD(Q,P). ✓
#
# Memory budget unchanged from condA — batch_size affects step time, not memory:
#   Actor update micro-batches by ppo_max_token_len_per_gpu=4096 regardless.
#   vLLM generates 64×8=512 sequences in its internal batches of max_num_seqs=32.
# ─────────────────────────────────────────────────────────────────────────────

echo "================================================================"
echo "  One-Shot-RLVR SDPO — Condition A (scalar reward, no feedback) batch=64"
echo "  Model : Qwen2.5-Math-1.5B            Hardware : 2× A100-40GB"
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
echo "    n=8  temperature=1.0  top_p=1.0  top_k=-1"
echo "    TP=1  gpu_mem_util=0.4  max_model_len=8192"
echo "    max_num_seqs=32  max_num_batched_tokens=16384"
echo "    val: n=1  do_sample=True  temperature=0.6  top_p=1.0"
echo "----------------------------------------------------------------"
echo "  Actor"
echo "    lr=1e-5  warmup_steps=10  entropy_coeff=0"
echo "    ppo_mini_batch_size=32  ppo_max_token_len_per_gpu=4096"
echo "    strategy=fsdp  model_dtype=bfloat16  use_torch_compile=False"
echo "----------------------------------------------------------------"
echo "  SDPO Condition A"
echo "    loss_mode=sdpo  alpha=0.5 (JSD)"
echo "    teacher_regularization=ema  teacher_update_rate=0.05"
echo "    distillation_topk=100  full_logit_distillation=True"
echo "    include_environment_feedback=False"
echo "    max_reprompt_len=4096  reprompt_truncation=left"
echo "    success_reward_threshold=1.0"
echo "    dont_reprompt_on_self_success=True"
echo "----------------------------------------------------------------"
echo "  Algorithm"
echo "    adv_estimator=grpo  norm_adv_by_std=False"
echo "    rollout_is=token  rollout_is_threshold=2.0"
echo "----------------------------------------------------------------"
echo "  Data"
echo "    train_batch_size=64  max_prompt_len=1024  max_resp_len=3072"
echo "    val_batch_size=500"
echo "----------------------------------------------------------------"
echo "  Training"
echo "    total_steps=300  save_freq=10  test_freq=10"
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
    actor_rollout_ref.model.trust_remote_code=True \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.max_num_seqs=32 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
    actor_rollout_ref.actor.use_torch_compile=False \
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
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=4096 \
    actor_rollout_ref.actor.self_distillation.reprompt_truncation=left \
    actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
    actor_rollout_ref.actor.self_distillation.distillation_add_tail=true \
    actor_rollout_ref.actor.self_distillation.is_clip=2.0 \
    \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.use_torch_compile=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    \
    data.train_files="[${TRAIN_PARQUET}]" \
    data.val_files="[${EVAL_PARQUET}]" \
    data.train_batch_size=64 \
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
    trainer.total_training_steps=300 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.val_before_train=True \
    trainer.log_val_generations=16 \
    trainer.experiment_name=oneshot_sdpo_qwen25_1b_pi1_condA_b64 \
    trainer.project_name=oneshot_sdpo \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.resume_mode=auto \
    trainer.validation_data_dir="${ROLLOUT_DIR}" \
    trainer.rollout_data_dir="${TRAIN_ROLLOUT_DIR}" \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.default_hdfs_dir=null \
    +ray_kwargs.ray_init.object_store_memory=5368709120 \
    2>&1 | tee "${OUTPUT_ROOT}/logs/train_${SLURM_JOB_ID}.log"
