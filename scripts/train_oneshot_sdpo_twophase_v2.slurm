#!/bin/bash
#SBATCH --job-name=oneshot_sdpo_2phv2
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=1
#SBATCH --time=16:00:00
#SBATCH --output=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/output_sdpo_twophase_v2/logs/train_%j.out
#SBATCH --error=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/output_sdpo_twophase_v2/logs/train_%j.err

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO
ONESHOT_DIR=${PROJECT_ROOT}
OUTPUT_ROOT=${PROJECT_ROOT}/output_sdpo_twophase_v2
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

# ── Two-Phase SDPO v2 ─────────────────────────────────────────────────────────
#
# v1 bug: trainer.resume_mode=auto in phase 2 restores the phase 1 optimizer
# state dict, which includes lr=1e-5 in param_groups. PyTorch's
# optimizer.load_state_dict() overwrites param_groups['lr'], so the new
# lr=1e-6 config value is silently ignored. Phase 2 ran at lr=1e-5,
# causing val to stall at 0.2-0.25 (JSD mode-seeking at lr=1e-5 is too
# aggressive for stable post-transition signals).
#
# v2 fix: Phase 2 loads model weights only, not optimizer state.
#   actor_rollout_ref.model.path → ${CKPT_DIR}/global_step_25/actor/huggingface
#     (the HuggingFace-format weights saved at end of phase 1)
#   trainer.resume_mode=disable  → fresh optimizer initialized at lr=1e-6
#   trainer.default_local_dir → phase 2's own ckpt dir (no phase 1 overlap)
#
# Phase 1 (steps 0-25, alpha=0, lr=1e-5):
#   Forward KL only — covers teacher's distribution without concentrating.
#   Survives the phase transition (success_fraction 35%→99% at step 5-7)
#   without catastrophic forgetting. By step 25 all 8 rollouts succeed.
#   Checkpoints saved every 5 steps. Phase 2 reads the HuggingFace export
#   at global_step_25 as its initial model.
#
# Phase 2 (steps 0-275, alpha=0.5 JSD, lr=1e-6, FRESH optimizer):
#   Real SDPO post-transition. lr=1e-6 is guaranteed because optimizer state
#   is NOT restored from phase 1. Gentle mode-seeking: JSD pushes student to
#   concentrate on teacher's high-probability tokens from sibling-solution
#   context while forward KL component prevents entropy collapse.
#   275 steps × ~250s ≈ 19h — fits inside 16h limit? No:
#     25 steps × ~250s + 275 steps × ~250s = 300 × ~250s ≈ 20.8h
#   So we cap phase 2 at 200 steps to be safe (total ≈ 56h wall-time max).
#   Adjust if phase 1 runs faster than expected.
#
# Timing note: actual step time from v1 run was ~250s (not 140s as estimated).
#   Phase 1: 25 × 250s ≈ 104min ≈ 1.7h
#   Phase 2: 200 × 250s ≈ 832min ≈ 13.9h
#   Total ≈ 15.6h < 16h limit (tight). Abort early if needed — checkpoints
#   are saved every 10 steps so at most 10 steps of work is lost.
# ─────────────────────────────────────────────────────────────────────────────

echo "================================================================" | tee -a "${LOG}"
echo "  Two-Phase SDPO v2  [Phase1: alpha=0 steps 0-25 | Phase2: JSD steps 0-200]" | tee -a "${LOG}"
echo "  v2 fix: Phase 2 loads HF weights only — fresh optimizer at lr=1e-6" | tee -a "${LOG}"
echo "  Job ID      : ${SLURM_JOB_ID}" | tee -a "${LOG}"
echo "  Model       : ${MODEL_PATH}" | tee -a "${LOG}"
echo "  Checkpoints : ${CKPT_DIR}" | tee -a "${LOG}"
echo "================================================================" | tee -a "${LOG}"

command -v apptainer >/dev/null 2>&1 || { echo "apptainer not found on host"; exit 1; }

# ── Shared args (rollout, data, reward — identical across both phases) ─────────
SHARED_ARGS=(
    --config-name ppo_trainer

    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.trust_remote_code=True

    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.n=8
    actor_rollout_ref.rollout.temperature=0.6
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.max_model_len=8192
    actor_rollout_ref.rollout.max_num_seqs=32
    actor_rollout_ref.rollout.max_num_batched_tokens=16384
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.val_kwargs.temperature=0
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0
    actor_rollout_ref.rollout.val_kwargs.do_sample=False
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.agent.num_workers=1

    actor_rollout_ref.actor.entropy_coeff=0.001
    actor_rollout_ref.actor.ppo_mini_batch_size=32
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False

    actor_rollout_ref.actor.policy_loss.loss_mode=sdpo
    actor_rollout_ref.actor.self_distillation.include_environment_feedback=false
    actor_rollout_ref.actor.self_distillation.success_reward_threshold=1.0
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=true
    actor_rollout_ref.actor.self_distillation.full_logit_distillation=true
    actor_rollout_ref.actor.self_distillation.teacher_regularization=ema
    actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.001
    actor_rollout_ref.actor.self_distillation.remove_thinking_from_demonstration=false
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=1024
    actor_rollout_ref.actor.self_distillation.reprompt_truncation=right
    actor_rollout_ref.actor.self_distillation.distillation_topk=100
    actor_rollout_ref.actor.self_distillation.distillation_add_tail=true
    actor_rollout_ref.actor.self_distillation.is_clip=2.0
    'actor_rollout_ref.actor.self_distillation.reprompt_template="{prompt}\n\nCorrect solution: {solution}\n\nCorrectly solve the original question."'
    'actor_rollout_ref.actor.self_distillation.solution_template="{successful_previous_attempt}"'

    actor_rollout_ref.ref.fsdp_config.param_offload=True

    algorithm.adv_estimator=grpo
    algorithm.norm_adv_by_std_in_grpo=False
    algorithm.rollout_correction.rollout_is=token
    algorithm.rollout_correction.rollout_is_threshold=2.0

    data.train_files="[${TRAIN_PARQUET}]"
    data.val_files="[${EVAL_PARQUET}]"
    data.train_batch_size=128
    data.val_batch_size=500
    data.max_prompt_length=1024
    data.max_response_length=3072
    data.filter_overlong_prompts=True
    data.trust_remote_code=True

    custom_reward_function.path="${REWARD_PATH}"
    custom_reward_function.name=compute_score

    'trainer.logger=["console","tensorboard"]'
    trainer.n_gpus_per_node=2
    trainer.nnodes=1
    trainer.total_epochs=9999
    trainer.val_before_train=True
    trainer.log_val_generations=0
    trainer.project_name=oneshot_sdpo
    trainer.validation_data_dir="${ROLLOUT_DIR}"
    trainer.rollout_data_dir="${TRAIN_ROLLOUT_DIR}"
    trainer.default_hdfs_dir=null
    +ray_kwargs.ray_init.object_store_memory=2147483648
)

# ══════════════════════════════════════════════════════════════════════════════
echo "" | tee -a "${LOG}"
echo "▶ PHASE 1: alpha=0.0, steps 0-25 (survive phase transition)" | tee -a "${LOG}"
echo "" | tee -a "${LOG}"

P1_CKPT_DIR="${CKPT_DIR}/phase1"
mkdir -p "${P1_CKPT_DIR}"

apptainer exec --nv \
    --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    --bind "/home/woody/iwi7/iwi7107h/models:/home/woody/iwi7/iwi7107h/models" \
    "${SIF}" \
    python -m verl.trainer.main_ppo \
    "${SHARED_ARGS[@]}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.self_distillation.alpha=0.0 \
    trainer.total_training_steps=25 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.max_actor_ckpt_to_keep=10 \
    trainer.resume_mode=disable \
    trainer.default_local_dir="${P1_CKPT_DIR}" \
    trainer.experiment_name=oneshot_sdpo_twophase_v2_p1_alpha0 \
    2>&1 | tee -a "${LOG}"

# Phase 1 checkpoint: HuggingFace weights at global_step_25
P1_HF_CKPT="${P1_CKPT_DIR}/oneshot_sdpo_twophase_v2_p1_alpha0/global_step_25/actor/huggingface"

echo "" | tee -a "${LOG}"
echo "▶ Phase 1 complete. HF checkpoint: ${P1_HF_CKPT}" | tee -a "${LOG}"
if [ ! -d "${P1_HF_CKPT}" ]; then
    echo "ERROR: phase 1 HF checkpoint not found at ${P1_HF_CKPT}" | tee -a "${LOG}"
    echo "Available checkpoints:" | tee -a "${LOG}"
    ls "${P1_CKPT_DIR}/" 2>&1 | tee -a "${LOG}"
    exit 1
fi

# ══════════════════════════════════════════════════════════════════════════════
echo "" | tee -a "${LOG}"
echo "▶ PHASE 2: alpha=0.5 JSD, 200 steps (genuine SDPO, FRESH optimizer at lr=1e-6)" | tee -a "${LOG}"
echo "  Loading weights from: ${P1_HF_CKPT}" | tee -a "${LOG}"
echo "  Optimizer state: NOT restored (fresh AdamW at lr=1e-6)" | tee -a "${LOG}"
echo "" | tee -a "${LOG}"

P2_CKPT_DIR="${CKPT_DIR}/phase2"
mkdir -p "${P2_CKPT_DIR}"

apptainer exec --nv \
    --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    --bind "/home/woody/iwi7/iwi7107h/models:/home/woody/iwi7/iwi7107h/models" \
    "${SIF}" \
    python -m verl.trainer.main_ppo \
    "${SHARED_ARGS[@]}" \
    actor_rollout_ref.model.path="${P1_HF_CKPT}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.self_distillation.alpha=0.5 \
    trainer.total_training_steps=200 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.max_actor_ckpt_to_keep=5 \
    trainer.resume_mode=disable \
    trainer.default_local_dir="${P2_CKPT_DIR}" \
    trainer.experiment_name=oneshot_sdpo_twophase_v2_p2_alpha05_lr1e6 \
    2>&1 | tee -a "${LOG}"
