#!/bin/bash
# One-time HPC environment setup for One-Shot-RLVR + SDPO.
#
# Run this ONCE on the HPC login node after copying the repo:
#   bash scripts/setup_hpc.sh
#
# This is a SETUP STEP, not a compute-node runtime dependency.
# Run it from the login node where pip/internet access is available.
# The installed packages persist in the sdpo conda environment.
#
# Assumes the sdpo conda env already contains:
#   torch, vllm, flash-attn, ray, transformers, datasets, pandas, pyarrow

set -e

PROJECT_ROOT=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO
OUTPUT_ROOT=${PROJECT_ROOT}/output

echo "================================================================"
echo "  One-Shot-RLVR + SDPO — HPC setup"
echo "  Project root: ${PROJECT_ROOT}"
echo "================================================================"

# Activate conda env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sdpo

# ── Step 1: Install SDPO (verl with SDPO modifications) ──────────────────────
# This installs verl plus all its declared dependencies:
#   tensorboard, pylatexenc, hydra-core, ray[default], accelerate, peft,
#   transformers, datasets, pandas, pyarrow, and more.
# Source: https://github.com/lasgroup/SDPO setup.py / pyproject.toml
echo ""
echo ">>> Installing SDPO (verl with SDPO modifications) from GitHub …"
pip install git+https://github.com/lasgroup/SDPO.git
echo ">>> SDPO installed."

# ── Step 2: Install sympy ─────────────────────────────────────────────────────
# Required by grade_answer_sympy in reward/math_reward.py.
# Not in SDPO's core install_requires, so install explicitly.
echo ""
echo ">>> Installing sympy …"
pip install sympy
echo ">>> sympy installed."

# ── Step 3: Create output directory tree ─────────────────────────────────────
echo ""
echo ">>> Creating output directory tree under ${OUTPUT_ROOT} …"
mkdir -p "${OUTPUT_ROOT}/checkpoints"
mkdir -p "${OUTPUT_ROOT}/logs"
mkdir -p "${OUTPUT_ROOT}/tensorboard"
mkdir -p "${OUTPUT_ROOT}/rollouts"
mkdir -p "${OUTPUT_ROOT}/eval_results"
echo ">>> Output directories created:"
ls -1 "${OUTPUT_ROOT}/"

# ── Step 4: Verify key imports ────────────────────────────────────────────────
echo ""
echo ">>> Verifying key imports …"
python - <<'EOF'
import verl
import torch
from torch.utils.tensorboard import SummaryWriter
import sympy
import pylatexenc
print("  verl          : ok")
print("  torch         :", torch.__version__)
print("  tensorboard   : ok")
print("  sympy         :", sympy.__version__)
print("  pylatexenc    : ok")
EOF
echo ">>> All imports verified."

echo ""
echo "================================================================"
echo "  Setup complete."
echo ""
echo "  Next steps:"
echo "    1. cd ${PROJECT_ROOT}/oneshot_sdpo"
echo "    2. python data/prepare_train_data.py  --output_dir data/datasets/train"
echo "    3. python data/prepare_math500_data.py --output_dir data/datasets/math500"
echo "    4. bash scripts/run_local_test.sh"
echo "    5. sbatch scripts/train_oneshot_sdpo.slurm"
echo "================================================================"
