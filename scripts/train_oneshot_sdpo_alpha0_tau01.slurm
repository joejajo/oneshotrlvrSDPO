#!/bin/bash
#SBATCH --job-name=oneshot_sdpo_a0_t01
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=1
#SBATCH --time=16:00:00
#SBATCH --output=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/output_sdpo_alpha0_tau01/logs/train_%j.out
#SBATCH --error=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/output_sdpo_alpha0_tau01/logs/train_%j.err

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO
ONESHOT_DIR=${PROJECT_ROOT}
OUTPUT_ROOT=${PROJECT_ROOT}/output_sdpo_alpha0_tau01
SIF=/home/woody/iwi7/iwi7107h/images/verl_vllm017_latest.sif

MODEL_PATH=/home/woody/iwi7/iwi7107h/models/Qwen2.5-Math-1.5B

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
export PYTHONWARNINGS="ignore:Unsupported processor type"
export TMPDIR=/tmp
export RAY_TMPDIR=/tmp/r${SLURM_JOB_ID}

ulimit -n 65536 2>/dev/null || true
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1

mkdir -p "${OUTPUT_ROOT}/logs" "${CKPT_DIR}" "${ROLLOUT_DIR}" "${TRAIN_ROLLOUT_DIR}" "${OUTPUT_ROOT}/tensorboard" "${RAY_TMPDIR}"

echo "================================================================"
echo "  One-Shot-RLVR Pure SDPO  [alpha=0.0 forward KL, tau=0.01]"
echo "  Job ID      : ${SLURM_JOB_ID}"
echo "  Model       : ${MODEL_PATH}"
echo "  Train data  : ${TRAIN_PARQUET}"
echo "  Eval data   : ${EVAL_PARQUET}"
echo "  Checkpoints : ${CKPT_DIR}"
echo "  TensorBoard : ${TENSORBOARD_DIR}"
echo "  Container   : ${SIF}"
echo "================================================================"

# ── alpha=0.0 forward KL + τ=0.01 compounding teacher ─────────────────────────
#
# Why alpha > 0 is blocked on a single training example:
#
#   The JSD formula (0 < alpha < 1) computes:
#     M = alpha*teacher + (1-alpha)*student
#     loss = (1-alpha)*KL(M||student) + alpha*KL(M||teacher)
#   For alpha=0.1: M is 90% student. KL(M||student) creates large gradients
#   precisely where 0.1*teacher >> 0.9*student — i.e., teacher's high-confidence
#   tokens. At the phase transition (success_fraction 35%→99% in one step),
#   99% of sequences simultaneously push these pi1-specific token gradients.
#   Result: catastrophic forgetting (val 0.310→0.074 at step 10, alpha=0.1 run).
#   The JSD formula is MORE damaging than either pure KL direction alone.
#
# Why τ=0.001 limits the alpha=0 ceiling:
#   With τ=0.001, teacher ≈ original model throughout (only 2% drift after 20 steps).
#   Post-saturation (step 20+): all 8 rollouts succeed → each reprompted with sibling
#   → forward KL pushes student toward original-model-conditional-on-pi1-solution.
#   As student drifts AWAY from original (val improves), the forward KL acts as a
#   restoration force pulling student BACK → val regression after step 40.
#   Best result: 0.368 at step 40, then regression.
#
# Why τ=0.01 creates a compounding improvement loop:
#   EMA update: teacher = (1-τ)*teacher + τ*student
#   τ=0.01: after step 20, teacher = 18% student + 82% original.
#   After step 100, teacher = 63% student + 37% original.
#
#   The teacher ABSORBS student improvements over time. Forward KL now pushes
#   student to cover teacher's distribution, which already includes the student's
#   math reasoning improvements. This creates a compounding signal:
#     improved student → teacher absorbs improvement → teacher target is now better
#     → forward KL pushes student further → better student → ...
#
#   This extends the effective learning window WELL PAST step 20:
#   - Steps 0-20: standard distillation (teacher mostly original, signal from
#     reprompted sibling context)
#   - Steps 20-100: teacher gradually = improved student; distillation target
#     drifts toward better model; KL non-zero as long as teacher ≠ student
#   - Steps 100-300: teacher mostly student; small KL → gentle continued signal
#
# Why not τ=0.01 in prior condD runs (which failed)?
#   condD_alpha0 (τ=0.01) failed due to rich feedback: the reprompt template
#   included "Your answer X is incorrect. The correct answer is 12.8." This bakes
#   pi1-specific answer into every distillation step → MATH-500 forgetting to 0.128.
#   This run uses NO feedback (include_environment_feedback=false) + custom template
#   with no {feedback} slot. τ=0.01 + no feedback has never been tried.
#
# Why τ=0.05 (paper default) doesn't work:
#   After step 20: teacher = 64% student. KL(teacher||student) ≈ very small
#   → gradient nearly zero → signal dies even faster than τ=0.001.
#   τ=0.01 is the sweet spot: teacher tracks student but stays meaningfully
#   different long enough to sustain a gradient for 100+ steps.
#
# Key parameters (all other settings identical to working alpha=0/lr=1e-5 run):
#   lr=1e-5, entropy_coeff=0.001, no feedback, custom template (no {feedback}),
#   reprompt_truncation=right, max_reprompt_len=1024, batch=128, temp=0.6.
#
# Expected: val peak > 0.368 (compounding teacher tracks improvements past step 40)
# ─────────────────────────────────────────────────────────────────────────────

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
    actor_rollout_ref.rollout.temperature=0.6 \
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
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.actor.policy_loss.loss_mode=sdpo \
    actor_rollout_ref.actor.self_distillation.include_environment_feedback=false \
    actor_rollout_ref.actor.self_distillation.success_reward_threshold=1.0 \
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=true \
    actor_rollout_ref.actor.self_distillation.full_logit_distillation=true \
    actor_rollout_ref.actor.self_distillation.alpha=0.0 \
    actor_rollout_ref.actor.self_distillation.teacher_regularization=ema \
    actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.01 \
    actor_rollout_ref.actor.self_distillation.remove_thinking_from_demonstration=false \
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=1024 \
    actor_rollout_ref.actor.self_distillation.reprompt_truncation=right \
    actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
    actor_rollout_ref.actor.self_distillation.distillation_add_tail=true \
    actor_rollout_ref.actor.self_distillation.is_clip=2.0 \
    'actor_rollout_ref.actor.self_distillation.reprompt_template="{prompt}\n\nCorrect solution: {solution}\n\nCorrectly solve the original question."' \
    'actor_rollout_ref.actor.self_distillation.solution_template="{successful_previous_attempt}"' \
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
    data.train_batch_size=128 \
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
    trainer.log_val_generations=0 \
    trainer.experiment_name=oneshot_sdpo_qwen25math_1.5b_pi1_lr1e5_alpha0_tau01 \
    trainer.project_name=oneshot_sdpo \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.resume_mode=disable \
    trainer.validation_data_dir="${ROLLOUT_DIR}" \
    trainer.rollout_data_dir="${TRAIN_ROLLOUT_DIR}" \
    trainer.max_actor_ckpt_to_keep=5 \
    trainer.default_hdfs_dir=null \
    +ray_kwargs.ray_init.object_store_memory=2147483648 \
    2>&1 | tee "${OUTPUT_ROOT}/logs/train_${SLURM_JOB_ID}.log"
