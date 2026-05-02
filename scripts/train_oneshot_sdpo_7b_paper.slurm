#!/bin/bash
#SBATCH --job-name=oneshot_sdpo_7b_paper_rope
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks=1
#SBATCH --time=16:00:00
#SBATCH --output=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/output_sdpo_7b_paper_rope/logs/train_%j.out
#SBATCH --error=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/output_sdpo_7b_paper_rope/logs/train_%j.err
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
# Memory budget (7B bf16, FSDP1 x4, bf16 optimizer state, no offload):
#   vLLM TP=2, gpu_memory_utilization=0.5 -> 20 GB per TP GPU. KV cache = 13 GB.
#
#   FSDP1 actor (strategy=fsdp, model_dtype=bfloat16, optimizer_offload=False):
#     fsdp_workers.py:325-327 defaults actor torch_dtype to fp32; we override
#     via model_dtype=bfloat16. The model loads bf16; FSDP shards bf16 params
#     across 4 GPUs; optimizer sees bf16 params and creates bf16 Adam states.
#     Per GPU: params 3.81 GB + bf16 Adam 7.62 GB + fp32 grads 7.62 GB
#                + logit peak 1.25 GB + scratch 3.81 GB ~= 24 GB. Fits 40 GB.
#     vLLM in sleep_mode during the actor step frees its 20 GB; no OOM.
#
#   Why FSDP2 was abandoned:
#     fsdp2_load_full_state_dict deadlocked at init with NCCL BROADCAST timing
#     out after 600s. torch 2.10.0 + CPUOffloadPolicy is broken for this path.
#
#   Why bitsandbytes 8-bit Adam was abandoned:
#     pip --target install pulled in torch 2.11.0, numpy 2.4.4, and ~30 packages
#     that shadow the container's torch 2.10.0+cu129 / vllm 0.17.0 and break
#     vllm + flash-attn. bf16 native saves enough memory without the conflict.
#
#   Ref/teacher (model_dtype=bfloat16, param_offload=True):
#     bf16 ref params on CPU, loaded shard-by-shard during teacher forward.
#     EMA update (dp_actor.py:151) sees both teacher and student bf16, no cast.
#
#   Model: open-r1/Qwen2.5-Math-7B-RoPE-300k (rope_theta=300000).
#
#   max_model_len note:
#     vllm_async_server.py:225 overwrites max_model_len with
#     get_max_position_embeddings(hf_config) = 32768 for RoPE-300k.
#     Our 6144 is IGNORED; vLLM uses 32768. Safe.
#
#   ppo_mini_batch_size=1 (paper run): single sequences; 4096 covers one seq.
#
#   GPU health pre-flight:
#     nvidia-smi check before Python starts; auto-requeue on bad nodes where
#     transformer_engine's lazy CUDA init would crash with DeferredCudaCallError.
# -----------------------------------------------------------------------------

echo "================================================================"
echo "  One-Shot-RLVR SDPO — Paper §3 (original SDPO params, no feedback)"
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
echo "    n=8  temperature=1.0  top_p=1.0  top_k=-1"
echo "    TP=2  gpu_mem_util=0.5  max_model_len=6144"
echo "    max_num_seqs=32  max_num_batched_tokens=32768"
echo "    val: n=1  do_sample=True  temperature=0.6  top_p=1.0"
echo "----------------------------------------------------------------"
echo "  Actor  [original SDPO run_sdpo.sh params]"
echo "    lr=1e-6  warmup_steps=0  entropy_coeff=0"
echo "    ppo_mini_batch_size=1  ppo_max_token_len_per_gpu=4096"
echo "    use_dynamic_bsz=True"
echo "    strategy=fsdp  model_dtype=bfloat16  use_torch_compile=False"
echo "----------------------------------------------------------------"
echo "  SDPO"
echo "    loss_mode=sdpo  alpha=1.0 (reverse KL)"
echo "    teacher_regularization=ema  teacher_update_rate=0.01"
echo "    distillation_topk=20  full_logit_distillation=True"
echo "    is_clip=2.0  distillation_add_tail=True"
echo "    max_reprompt_len=2400  reprompt_truncation=right"
echo "    include_environment_feedback=False"
echo "    success_reward_threshold=1.0"
echo "    dont_reprompt_on_self_success=True"
echo "----------------------------------------------------------------"
echo "  Algorithm"
echo "    adv_estimator=grpo  norm_adv_by_std=False"
echo "    rollout_is=token  rollout_is_threshold=2.0"
echo "----------------------------------------------------------------"
echo "  Data"
echo "    train_batch_size=32  max_prompt_len=1024  max_resp_len=2048"
echo "    val_batch_size=500"
echo "----------------------------------------------------------------"
echo "  Training"
echo "    total_steps=300  save_freq=5  test_freq=5"
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
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
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
    trainer.test_freq=5 \
    trainer.val_before_train=True \
    trainer.log_val_generations=16 \
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
