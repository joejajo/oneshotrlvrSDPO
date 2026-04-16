#!/bin/bash
#SBATCH --job-name=oneshot_sdpo_sec3
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=1
#SBATCH --time=16:00:00
#SBATCH --output=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/output_sec3/logs/train_%j.out
#SBATCH --error=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/output_sec3/logs/train_%j.err

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO
ONESHOT_DIR=${PROJECT_ROOT}
OUTPUT_ROOT=${PROJECT_ROOT}/output_sec3
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
# No conda or module load needed — runtime is supplied by the Apptainer container.
# All env vars set here are inherited by the container (Apptainer does not clean env
# by default).

export WANDB_MODE=disabled
export WANDB_DISABLED=true

# TensorBoard: SDPO's _TensorboardAdapter reads TENSORBOARD_DIR at init.
# Logs land at: $TENSORBOARD_DIR/<project_name>/<experiment_name>/
export TENSORBOARD_DIR=${OUTPUT_ROOT}/tensorboard

# Make reward module and verl importable inside the container.
SDPO_DIR=${PROJECT_ROOT}/SDPO
# pkgs/ holds packages not in the container (latex2sympy2, regex, antlr4).
# Install once with:
#   pip install --target=${PROJECT_ROOT}/pkgs latex2sympy2 regex antlr4-python3-runtime==4.9.3
# antlr4-python3-runtime==4.9.3 must match latex2sympy2's generated code version.
PKGS_DIR=${PROJECT_ROOT}/pkgs
export PYTHONPATH="${ONESHOT_DIR}:${SDPO_DIR}:${PKGS_DIR}:${PYTHONPATH:-}"

# Container ships its own CUDA at /usr/local/cuda — use that as fallback.
# Do NOT call "module load cuda" here; modules are unavailable inside Apptainer.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"

# --nv rewrites LD_LIBRARY_PATH but does NOT touch LD_PRELOAD.
# Force-load container versions of these libraries before --nv's host versions:
#   libcupti    : host CUPTI missing cuptiActivityEnableDriverApi
#   libnccl     : host NCCL missing ncclCommShrink (batch nodes have old host NCCL)
#   libcublas*  : ensure cuBLASLt init uses container's version, not host's
#                 (host cuBLASLt init fails in vLLM worker when NCCL is also preloaded)
_NV="/usr/local/lib/python3.12/dist-packages/nvidia"
export APPTAINERENV_LD_PRELOAD="\
${_NV}/cuda_cupti/lib/libcupti.so.12:\
${_NV}/nccl/lib/libnccl.so.2:\
${_NV}/cublas/lib/libcublas.so.12:\
${_NV}/cublas/lib/libcublasLt.so.12"

# vLLM settings
# VLLM_USE_V1 is NOT recognized by vLLM 0.17.0 — triggers "Unknown env var" warning.
# VLLM_DISABLE_COMPILE_CACHE=1 suppresses the torch.compile serialization crash.
unset VLLM_ATTENTION_BACKEND
unset ROCR_VISIBLE_DEVICES
export VLLM_DISABLE_COMPILE_CACHE=1
export PYTHONUNBUFFERED=1

# Block ~/.local/lib/python3.12/site-packages from being loaded inside the container.
# The user has an incompatible Triton installed at ~/.local that overrides the container's
# Triton, causing: triton.compiler.errors.CompilationError (uint32/int32 signedness mismatch)
# in vLLM's topk_topp_triton kernel during profile_run().
export PYTHONNOUSERSITE=1

# Prevent uvloop from calling getenv("UV_THREADPOOL_SIZE") during event loop init.
# Without this, concurrent AgentLoopWorker actor spawns race in glibc getenv → SIGSEGV.
export APPTAINERENV_UV_THREADPOOL_SIZE=4

# Suppress known-benign warning from verl's multimodal processor path:
#   "Unsupported processor type: Qwen2TokenizerFast"
# This code path is not used for text-only math training.
export PYTHONWARNINGS="ignore:Unsupported processor type"

# Use node-local /tmp for multiprocessing temp files to avoid NFS-related
# "OSError: [Errno 16] Device or resource busy: pymp-*" cleanup noise.
export TMPDIR=/tmp

# AF_UNIX socket paths are capped at 107 bytes on Linux.
# Ray appends ~63 chars, so RAY_TMPDIR must be ≤ 44 chars. Use short /tmp path.
export RAY_TMPDIR=/tmp/r${SLURM_JOB_ID}

# Raise open-file-descriptor limit — Ray + vLLM open many sockets.
ulimit -n 65536 2>/dev/null || true

# Ray plasma store uses /dev/shm by default.
# SLURM often caps /dev/shm at 64–512 MB, causing the Raylet to crash.
# Allow plasma to fall back to /tmp if /dev/shm is too small.
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1

mkdir -p "${OUTPUT_ROOT}/logs" "${CKPT_DIR}" "${ROLLOUT_DIR}" "${TRAIN_ROLLOUT_DIR}" "${OUTPUT_ROOT}/tensorboard" "${RAY_TMPDIR}"

echo "================================================================"
echo "  One-Shot-RLVR + SDPO training  [Paper Section 3 — exact settings]"
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

# ── SDPO Paper Section 3 — scalar reward, exact hyperparameters ──────────────
#
# This script reproduces the SDPO paper Section 3 (Table 4) setting as closely
# as possible on the One-Shot-RLVR π₁ task.
#
# Key differences vs our condition A (train_oneshot_sdpo_nofeedback.slurm):
#
#   rollout.temperature=1.0   (paper uses temp=1.0; condition A uses 0.6)
#   data.train_batch_size=32  (paper uses 32; condition A uses 128)
#   optim.lr=1e-5             (paper lr for batch=32; condition A uses 1e-6 scaled for batch=128)
#   entropy_coeff=0           (paper has no entropy term; all conditions now use 0)
#   norm_adv_by_std_in_grpo=True  (paper/YAML default; condition A uses False)
#
# Shared with condition A (confirmed against SDPO paper Section 3 / Table 4):
#   alpha=0.5                 (JSD = symmetric KL; Section 3 "scalar reward" setting)
#   teacher_update_rate=0.05  (EMA τ=0.05; paper/YAML default)
#   distillation_topk=100     (Section 3 default; condition D uses 20)
#   include_environment_feedback=false  (scalar-only; no text feedback)
#   is_clip=2.0               (IS weight truncation)
#   rollout.n=8               (group size G=8)
#
# SDPO loss (Section 3 scalar reward, no entropy):
#   L = α·KL(m‖π_teacher) + (1-α)·KL(m‖π_student)    [m = mixture dist]
#   where α=0.5 → symmetric JSD
#   Importance sampling: ratio = exp(log π_student − log π_old), clipped at is_clip=2.0
#   Top-k=100 tokens per position + uniform tail bucket
#   Policy gradient advantage (GRPO): normalised by std (norm_adv_by_std_in_grpo=True)
#
# Step budget:
#   batch=32, ~4 prompts/step (128 prompts / 32 batch), ~140s/step → 300 steps ≈ 11.7h
#   16h wall time gives headroom for startup. Validation is in-memory (Hybrid Engine weight sync), essentially free.
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
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
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
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=2048 \
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
    trainer.test_freq=1 \
    trainer.val_before_train=True \
    trainer.log_val_generations=0 \
    trainer.experiment_name=oneshot_sdpo_qwen25math_1.5b_pi1_sec3 \
    trainer.project_name=oneshot_sdpo \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.resume_mode=auto \
    trainer.validation_data_dir="${ROLLOUT_DIR}" \
    trainer.rollout_data_dir="${TRAIN_ROLLOUT_DIR}" \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.default_hdfs_dir=null \
    +ray_kwargs.ray_init.object_store_memory=2147483648 \
    2>&1 | tee "${OUTPUT_ROOT}/logs/train_${SLURM_JOB_ID}.log"
