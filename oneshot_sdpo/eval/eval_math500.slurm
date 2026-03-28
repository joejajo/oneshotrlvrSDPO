#!/bin/bash
#SBATCH --job-name=eval_math500
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --time=02:00:00
#SBATCH --output=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/output/logs/eval_math500_%j.out
#SBATCH --error=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/output/logs/eval_math500_%j.err

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO
ONESHOT_DIR=${PROJECT_ROOT}
OUTPUT_ROOT=${PROJECT_ROOT}/output

EVAL_DATA=${ONESHOT_DIR}/data/math500.parquet
OUTPUT_DIR=${OUTPUT_ROOT}/eval_results

# Checkpoint to evaluate.
# Pass via --export when submitting, e.g.:
#   sbatch --export=CKPT=${OUTPUT_ROOT}/checkpoints/global_step_500 eval/eval_math500.slurm
# Falls back to the final checkpoint if CKPT is not set.
CKPT=${CKPT:-${OUTPUT_ROOT}/checkpoints/global_step_500}

# ── Environment ───────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sdpo

export WANDB_MODE=disabled
export PYTHONPATH=${ONESHOT_DIR}:${PYTHONPATH}

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_ROOT}/logs"

echo "================================================================"
echo "  MATH-500 evaluation"
echo "  Job ID      : ${SLURM_JOB_ID}"
echo "  Checkpoint  : ${CKPT}"
echo "  Eval data   : ${EVAL_DATA}"
echo "  Output dir  : ${OUTPUT_DIR}"
echo "================================================================"

# ── Run eval ──────────────────────────────────────────────────────────────────
python "${ONESHOT_DIR}/eval/eval_math500.py" \
    --checkpoint "${CKPT}" \
    --eval_data  "${EVAL_DATA}" \
    --output_dir "${OUTPUT_DIR}" \
    2>&1 | tee "${OUTPUT_ROOT}/logs/eval_math500_${SLURM_JOB_ID}.log"
