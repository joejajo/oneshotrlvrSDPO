#!/bin/bash
#SBATCH --job-name=sdpo_q3_8b_A
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks=1
#SBATCH --time=16:00:00
#SBATCH --output=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/output_sdpo_8b_qwen3/logs/train_%j.out
#SBATCH --error=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/output_sdpo_8b_qwen3/logs/train_%j.err
#SBATCH --requeue

set -euo pipefail

# ── GPU health check (handles random bad nodes anywhere in the cluster) ────────
# Detects GPUs in ERR! state where nvidia-smi reports utilization as N/A.
# When found, requeues the job so SLURM tries a different allocation.
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
        # CUDA reports utilization as "[N/A]" or "N/A" or "ERR!" when the GPU is
        # in an error state. A healthy GPU returns a percentage like "0 %" or "97 %".
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
    scontrol requeue "${SLURM_JOB_ID}" || echo "scontrol requeue failed (may not be permitted); exiting with TEMPFAIL anyway"
    exit 75
fi
echo "GPU health check passed on $(hostname)."


# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO
ONESHOT_DIR=${PROJECT_ROOT}
OUTPUT_ROOT=${PROJECT_ROOT}/output_sdpo_8b_qwen3
SIF=/home/woody/iwi7/iwi7107h/images/verl_vllm017_latest.sif

MODEL_PATH=/home/woody/iwi7/iwi7107h/models/Qwen3-8B

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

# ── Qwen3-8B SDPO on 4 x A100-40GB ───────────────────────────────────────────
#
# Memory budget (8B bf16, FSDP1 x4, bf16 optimizer state, no offload):
#
#   vLLM (TP=2, gpu_memory_utilization=0.5):
#     Each TP GPU reserves 0.5 x 40 = 20 GB for vLLM.
#     8B weights bf16 / TP=2 = 8 GB per GPU. KV cache per GPU: 20 - 8 = 12 GB.
#
#   FSDP1 actor (strategy=fsdp, model_dtype=bfloat16, optimizer_offload=False):
#     FSDP shards bf16 params across 4 GPUs; optimizer sees bf16 and creates
#     bf16 Adam states.
#     Per GPU: params 4 GB + bf16 Adam 8 GB + fp32 grads 8 GB
#                + logit peak ~0.72 GB + scratch ~4 GB ~= 25 GB.
#     vLLM is in sleep_mode (VLLM_SLEEP_LEVEL=2) during actor step, so all
#     40 GB are free per GPU. No OOM.
#
#   Why we abandoned FSDP2 + CPUOffloadPolicy:
#     fsdp2_load_full_state_dict deadlocked at init with NCCL BROADCAST timing
#     out after 600s (rank-asymmetric op counts). torch 2.10.0 + offload policy
#     is broken for this codepath.
#
#   Why we abandoned bitsandbytes 8-bit Adam:
#     pip install --target=pkgs bitsandbytes pulled in conflicting torch 2.11.0,
#     numpy 2.4.4, and ~30 other packages that shadow the container's torch
#     2.10.0+cu129 / vllm 0.17.0 via PYTHONPATH and break vllm + flash-attn.
#
#   Ref/teacher (model_dtype=bfloat16, param_offload=True):
#     bf16 ref params on CPU, loaded shard-by-shard during teacher forward.
#     EMA update (dp_actor.py:151) sees both teacher and student bf16, no cast.
#
#   Model: Qwen3-8B
#     vocab_size=151,936, max_position_embeddings=32768 (GQA, 32 attn heads,
#     8 KV heads, 36 hidden layers).
#
#   Qwen3 thinking mode:
#     Qwen3 generates <think>...</think> traces by default. This is disabled via
#     data.apply_chat_template_kwargs={enable_thinking: false} which SDPO passes
#     to tokenizer.apply_chat_template for both training data and reprompt
#     tokenization (ray_trainer.py:749-757). remove_thinking_from_demonstration
#     stays false because thinking is disabled at generation time.
#
#   vLLM compatibility note:
#     vLLM 0.17.0 predates Qwen3 (April 2025). The container uses
#     trust_remote_code=True + transformers HF modeling code as fallback.
#     Verify transformers supports Qwen3ForCausalLM before submitting.
#
#   max_model_len note:
#     vllm_async_server.py:225 overwrites max_model_len with
#     get_max_position_embeddings(hf_config) = 32768 for Qwen3-8B.
#     Our max_model_len=6144 is IGNORED; vLLM uses 32768. Safe: our sequences
#     never exceed reprompt(2600)+response(2048)=4648 tok.
#
#   VLLM_SLEEP_LEVEL=2: enable_sleep_mode=True frees vLLM weights AND KV cache
#     during actor update.
#
#   use_torch_compile=False (actor + ref):
#     Inductor buffers consume 5-8 GB extra GPU memory; disabling runs in eager.
#
#   ppo_max_token_len_per_gpu=4096: controls STUDENT micro-batch sizing only.
#     Student logit peak = 2363*151936*2 = 718 MB. Teacher logit peak on response
#     tokens only = 2048*151936*2 = 623 MB. Both comfortably fit.
#   max_reprompt_len=2600: π₁ question (~315 tok) + sibling solution (up to 2048 tok)
#     + template overhead (~57 tok) = max 2420 tok. Buffer of 180 tok.
#     reprompt_truncation=left (safety): truncates start of question vs right
#     which drops <|im_end|> and breaks chat format.
#
#   GPU health pre-flight:
#     Some nodes have GPUs in ERR! state where nvidia-smi reports utilization
#     as N/A. We pre-check via nvidia-smi before any Python starts and
#     `scontrol requeue` to a different node if any GPU is bad.
#
# Sampling hyperparameters:
#   top_p=1.0, top_k=-1: full distribution.
#   temperature=1.0 train, 0.6 val, do_sample=True.
#
# SDPO Condition A (scalar reward, no environment feedback):
#   alpha=0.5 (JSD), distillation_topk=100, teacher_update_rate=0.05, lr=1e-5.
# -----------------------------------------------------------------------------

echo "================================================================"
echo "  One-Shot-RLVR SDPO — Condition A (scalar reward, no feedback)"
echo "  Model : Qwen3-8B                     Hardware : 4× A100-40GB"
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
echo "    TP=2  gpu_mem_util=0.5  max_model_len=6144"
echo "    max_num_seqs=32  max_num_batched_tokens=32768"
echo "    val: n=1  do_sample=True  temperature=0.6  top_p=1.0"
echo "----------------------------------------------------------------"
echo "  Actor"
echo "    lr=1e-5  warmup_steps=20  entropy_coeff=0"
echo "    ppo_mini_batch_size=32  ppo_max_token_len_per_gpu=4096"
echo "    use_dynamic_bsz=True"
echo "    strategy=fsdp  model_dtype=bfloat16  use_torch_compile=False"
echo "----------------------------------------------------------------"
echo "  SDPO"
echo "    loss_mode=sdpo  alpha=0.5 (JSD)"
echo "    teacher_regularization=ema  teacher_update_rate=0.05"
echo "    distillation_topk=100  full_logit_distillation=True"
echo "    is_clip=2.0  distillation_add_tail=True"
echo "    max_reprompt_len=2600  reprompt_truncation=left"
echo "    include_environment_feedback=False"
echo "    success_reward_threshold=1.0"
echo "    dont_reprompt_on_self_success=True"
echo "----------------------------------------------------------------"
echo "  Algorithm"
echo "    adv_estimator=grpo  norm_adv_by_std=True"
echo "    rollout_is=token  rollout_is_threshold=2.0"
echo "----------------------------------------------------------------"
echo "  Data"
echo "    train_batch_size=32  max_prompt_len=1024  max_resp_len=2048"
echo "    val_batch_size=500"
echo "----------------------------------------------------------------"
echo "  Training"
echo "    total_steps=300  save_freq=5  test_freq=20"
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
    '+data.apply_chat_template_kwargs={enable_thinking: false}' \
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
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
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
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=2600 \
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
    algorithm.norm_adv_by_std_in_grpo=True \
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
    trainer.total_training_steps=300 \
    trainer.save_freq=5 \
    trainer.test_freq=20 \
    trainer.val_before_train=True \
    trainer.log_val_generations=16 \
    trainer.experiment_name=oneshot_sdpo_qwen3_8b_pi1 \
    trainer.project_name=oneshot_sdpo \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.resume_mode=disable \
    trainer.validation_data_dir="${ROLLOUT_DIR}" \
    trainer.rollout_data_dir="${TRAIN_ROLLOUT_DIR}" \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.default_hdfs_dir=null \
    +ray_kwargs.ray_init.object_store_memory=2147483648 \
    2>&1 | tee "${OUTPUT_ROOT}/logs/train_${SLURM_JOB_ID}.log"
