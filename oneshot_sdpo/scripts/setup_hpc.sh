#!/bin/bash
# One-time HPC environment setup for One-Shot-RLVR + SDPO.
#
# Run this ONCE on the HPC login node after cloning the repo:
#   bash scripts/setup_hpc.sh
#
# This is a SETUP STEP, not a compute-node runtime dependency.
# Run it from the login node where pip/internet access is available.
# The installed packages persist in the sdpo_a100 conda environment.
#
# Prerequisites (see CLAUDE.md "Full install sequence" for full env creation):
#   - conda env sdpo_a100 exists at /home/woody/iwi7/iwi7107h/conda_envs/sdpo_a100
#   - torch 2.5.1, vllm 0.8.4, flash-attn, ray[default]==2.53.0 already installed
#   - SDPO repo cloned at /home/woody/iwi7/iwi7107h/SDPO

set -e

PROJECT_ROOT=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO
OUTPUT_ROOT=${PROJECT_ROOT}/output

echo "================================================================"
echo "  One-Shot-RLVR + SDPO — HPC setup"
echo "  Project root: ${PROJECT_ROOT}"
echo "================================================================"

# Activate conda env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/woody/iwi7/iwi7107h/conda_envs/sdpo_a100

# ── Step 1: Install SDPO (verl with SDPO modifications) ──────────────────────
# Install verl as an editable package from the local SDPO clone.
# Using pip install -e . (NOT git+https) so edits to the local clone take effect.
# requirements.txt covers all deps: tensorboard, pylatexenc, hydra-core,
#   ray[default]==2.53.0, numpy==2.1.0, accelerate, peft, transformers,
#   datasets, pandas, pyarrow, sympy, word2number, math-verify, and more.
SDPO_DIR=/home/woody/iwi7/iwi7107h/SDPO
echo ""
echo ">>> Installing SDPO deps from requirements.txt …"
pip install -r "${SDPO_DIR}/requirements.txt"
echo ""
echo ">>> Installing SDPO verl fork as editable package …"
pip install -e "${SDPO_DIR}"
echo ">>> SDPO installed."

# ── Step 2: Create output directory tree ─────────────────────────────────────
echo ""
echo ">>> Creating output directory tree under ${OUTPUT_ROOT} …"
mkdir -p "${OUTPUT_ROOT}/checkpoints"
mkdir -p "${OUTPUT_ROOT}/logs"
mkdir -p "${OUTPUT_ROOT}/tensorboard"
mkdir -p "${OUTPUT_ROOT}/rollouts"
mkdir -p "${OUTPUT_ROOT}/eval_results"
echo ">>> Output directories created:"
ls -1 "${OUTPUT_ROOT}/"

# ── Step 3: Verify key imports ────────────────────────────────────────────────
echo ""
echo ">>> Verifying key imports …"
python - <<'EOF'
import verl
import torch
import numpy
import ray
from torch.utils.tensorboard import SummaryWriter
import sympy
import pylatexenc
from verl.trainer.ppo.core_algos import compute_self_distillation_loss
print("  verl          :", verl.__file__)
print("  torch         :", torch.__version__)   # expect 2.5.1+cu124
print("  numpy         :", numpy.__version__)    # expect 2.1.0
print("  ray           :", ray.__version__)      # expect 2.53.0
print("  tensorboard   : ok")
print("  sympy         :", sympy.__version__)
print("  pylatexenc    : ok")
print("  compute_self_distillation_loss : ok")
EOF
echo ">>> All imports verified."

echo ""
echo "================================================================"
echo "  Setup complete."
echo ""
echo "  Next steps:"
echo "    1. cd ${PROJECT_ROOT}/oneshot_sdpo"
echo "    2. bash scripts/run_local_test.sh"
echo "    3. sbatch scripts/train_oneshot_sdpo.slurm"
echo "================================================================"
