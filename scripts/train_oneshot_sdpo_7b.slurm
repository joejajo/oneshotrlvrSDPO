#!/bin/bash
#SBATCH --job-name=oneshot_sdpo_7b_rope
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks=1
#SBATCH --time=16:00:00
#SBATCH --output=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/output_sdpo_7b_rope/logs/train_%j.out
#SBATCH --error=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/output_sdpo_7b_rope/logs/train_%j.err

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO
ONESHOT_DIR=${PROJECT_ROOT}
OUTPUT_ROOT=${PROJECT_ROOT}/output_sdpo_7b_rope
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

mkdir -p "${OUTPUT_ROOT}/logs" "${CKPT_DIR}" "${ROLLOUT_DIR}" \
         "${TRAIN_ROLLOUT_DIR}" "${OUTPUT_ROOT}/tensorboard" "${RAY_TMPDIR}"

# ── Qwen2.5-Math-7B SDPO on 4× A100-40GB ─────────────────────────────────────
#
# Memory budget (7B bf16, FSDP×4, optimizer_offload=True):
#
#   vLLM (TP=2, gpu_memory_utilization=0.5):
#     Each of the 2 TP GPUs reserves 0.5 × 40 = 20 GB for vLLM.
#     7B weights in bf16 = 14 GB; with TP=2 each holds 7 GB weights.
#     KV cache per GPU: 20 - 7 = 13 GB.
#     Ample for 32 seqs × 4096 tokens.
#
#   FSDP actor (optimizer_offload=True):
#     Sharded params across 4 GPUs: 14 GB / 4 = 3.5 GB per GPU.
#     Adam fp32 states (56 GB) → CPU (optimizer_offload=True).
#     Peak during forward all-gather: 14 GB params on active GPU.
#     During rollout vLLM holds 20 GB on its 2 GPUs; FSDP params stay
#     in their 3.5 GB shards → peak on TP GPUs: 20 + 3.5 = 23.5 GB < 40 GB ✓
#
#   Ref/teacher (param_offload=True):
#     Kept on CPU, loaded shard-by-shard during teacher forward. ✓
#
#   Model: open-r1/Qwen2.5-Math-7B-RoPE-300k
#     rope_theta=300000, max_position_embeddings=32768 (extended from base 4096).
#     Same weights as Qwen2.5-Math-7B; only RoPE config changed.
#
#   max_model_len config note:
#     vllm_async_server.py:225 unconditionally overwrites max_model_len with
#     get_max_position_embeddings(hf_config) = 32768 for RoPE-300k.
#     Our max_model_len=6144 setting is IGNORED at runtime; vLLM uses 32768.
#     Safe: our sequences never exceed reprompt(2400)+response(2048)=4448 tok.
#     max_num_batched_tokens=32768 aligns with actual vLLM max_model_len=32768. ✓
#
#   VLLM_SLEEP_LEVEL note:
#     verl/third_party/vllm/__init__.py auto-detects level=2 for vllm>=0.8.5.
#     vLLM 0.17.0 > 0.8.5 → VLLM_SLEEP_LEVEL=2 in Python without any env var.
#     Our export VLLM_SLEEP_LEVEL=2 sets an env var nothing reads; harmless.
#     enable_sleep_mode=True (default in RolloutConfig) → vLLM frees weights
#     AND KV cache during training, giving full 40 GB for actor update. ✓
#
#   max_reprompt_len=2400: π₁ reprompt worst-case = 2206 tok < 2400 → no truncation ✓
#   reprompt_truncation=right: SDPO default; irrelevant since 2400>2206.
#
#   use_torch_compile=False (actor + ref):
#     Default in actor.yaml is use_torch_compile=True. Torch.compile/inductor buffers
#     consume 5-8 GB extra GPU memory during actor update ("Online softmax is disabled
#     on the fly since Inductor decides to split the reduction"). With compile enabled,
#     35.88 GB was in use leaving only 3.57 GB — insufficient for logsumexp (needs 5 GB).
#     Disabling compile frees those buffers; actor update runs in eager mode (~1.5× slower
#     per-step but no OOM). Ref also needs False since it shares FSDP state.
#
#   ppo_max_token_len_per_gpu=8192: teacher logit matrix = 8192×152064×2 ≈ 2.5 GB;
#   logsumexp peak ≈ 5 GB. Without compile overhead (~26.5 GB base), headroom ≈ 8.5 GB ✓
#   (16384 caused OOM; 4096 was safe but created ~48 micro-batches; 8192 gives ~23)
#
# Sampling hyperparameters — matched to original SDPO (rollout.yaml defaults):
#   top_p=1.0, top_k=-1: full distribution, no nucleus / top-k truncation.
#   temperature=1.0: training rollouts (original SDPO default).
#   val_kwargs.temperature=0.6: inline eval (greedy is too noisy at 500 problems).
#
# SDPO Condition A settings (scalar reward, no environment feedback):
#   alpha=0.5 (JSD), distillation_topk=100, teacher_update_rate=0.05, lr=1e-5.
# ─────────────────────────────────────────────────────────────────────────────

echo "================================================================"
echo "  One-Shot-RLVR SDPO — Condition A (scalar reward, no feedback)"
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
echo "    val: n=1  temperature=0 (greedy)  top_p=1.0"
echo "----------------------------------------------------------------"
echo "  Actor"
echo "    lr=1e-5  warmup_steps=20  entropy_coeff=0"
echo "    ppo_mini_batch_size=32  ppo_max_token_len_per_gpu=8192"
echo "    use_dynamic_bsz=True"
echo "    param_offload=False  optimizer_offload=True  use_torch_compile=False"
echo "----------------------------------------------------------------"
echo "  SDPO"
echo "    loss_mode=sdpo  alpha=0.5 (JSD)"
echo "    teacher_regularization=ema  teacher_update_rate=0.05"
echo "    distillation_topk=100  full_logit_distillation=True"
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
echo "    total_steps=50  save_freq=5  test_freq=5"
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
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
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
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=2400 \
    actor_rollout_ref.actor.self_distillation.reprompt_truncation=right \
    actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
    actor_rollout_ref.actor.self_distillation.distillation_add_tail=true \
    actor_rollout_ref.actor.self_distillation.is_clip=2.0 \
    \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.use_torch_compile=False \
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
    trainer.total_training_steps=50 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.val_before_train=True \
    trainer.log_val_generations=16 \
    trainer.experiment_name=oneshot_sdpo_qwen25math_7b_rope300k_pi1 \
    trainer.project_name=oneshot_sdpo \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.resume_mode=disable \
    trainer.validation_data_dir="${ROLLOUT_DIR}" \
    trainer.rollout_data_dir="${TRAIN_ROLLOUT_DIR}" \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.default_hdfs_dir=null \
    +ray_kwargs.ray_init.object_store_memory=2147483648 \
    2>&1 | tee "${OUTPUT_ROOT}/logs/train_${SLURM_JOB_ID}.log"
