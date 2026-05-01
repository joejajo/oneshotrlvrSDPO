#!/bin/bash
# Run this on the LOGIN node before sbatch'ing the 7B training jobs.
# Compute nodes have no internet access, so pip install must happen here.
#
# Installs bitsandbytes (for AdamW8bit optimizer) into the project's pkgs/
# directory, using the container's Python so the compiled .so files are
# ABI-compatible with the runtime environment used during training.

set -euo pipefail

PROJECT_ROOT=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO
PKGS_DIR=${PROJECT_ROOT}/pkgs
SIF=/home/woody/iwi7/iwi7107h/images/verl_vllm017_latest.sif

if [ ! -f "${SIF}" ]; then
    echo "ERROR: Apptainer image not found at ${SIF}"
    exit 1
fi

mkdir -p "${PKGS_DIR}"

echo "Installing bitsandbytes into ${PKGS_DIR} ..."
apptainer exec \
    --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    "${SIF}" \
    pip install --target="${PKGS_DIR}" --upgrade bitsandbytes

echo
echo "Verifying installation (with --nv so CUDA libs are visible) ..."
apptainer exec --nv \
    --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    "${SIF}" \
    bash -c "PYTHONPATH='${PKGS_DIR}' python -c '
import bitsandbytes
print(\"bitsandbytes:\", bitsandbytes.__version__)
from bitsandbytes.optim import AdamW8bit
print(\"AdamW8bit:    importable\")
'"

echo
echo "Done. You can now sbatch the 7B training scripts."
