#!/bin/bash
# Pre-flight smoke test for One-Shot-RLVR + SDPO.
#
# Run this interactively on the HPC LOGIN NODE after setup_hpc.sh and before
# submitting the SLURM job.  Every step must pass before sbatch.
#
# Usage:
#   conda activate sdpo
#   cd /home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/oneshot_sdpo
#   bash scripts/run_local_test.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONESHOT_DIR="$(dirname "${SCRIPT_DIR}")"

echo "================================================================"
echo "  One-Shot-RLVR + SDPO — pre-flight smoke test"
echo "  Working directory: ${ONESHOT_DIR}"
echo "================================================================"

cd "${ONESHOT_DIR}"

# ── Step 1: Generate training parquets ───────────────────────────────────────
echo ""
echo ">>> Step 1: prepare_train_data.py"
python data/prepare_train_data.py --output_dir data/datasets/train

# Verify row counts
python - <<'EOF'
import pandas as pd
train = pd.read_parquet("data/datasets/train/train.parquet")
val   = pd.read_parquet("data/datasets/train/val.parquet")
assert len(train) == 128, f"Expected 128 train rows, got {len(train)}"
assert len(val)   == 16,  f"Expected 16 val rows, got {len(val)}"
expected_cols = {"data_source", "prompt", "ability", "reward_model", "extra_info"}
assert expected_cols.issubset(set(train.columns)), (
    f"Missing columns: {expected_cols - set(train.columns)}"
)
print(f"  train.parquet : {len(train)} rows  columns={train.columns.tolist()}  OK")
print(f"  val.parquet   : {len(val)} rows    columns={val.columns.tolist()}  OK")
EOF
echo ">>> Step 1 passed."

# ── Step 2: Generate MATH-500 eval parquet ────────────────────────────────────
echo ""
echo ">>> Step 2: prepare_math500_data.py"
python data/prepare_math500_data.py --output_dir data/datasets/math500

python - <<'EOF'
import pandas as pd
df = pd.read_parquet("data/datasets/math500/eval.parquet")
assert len(df) == 500, f"Expected 500 eval rows, got {len(df)}"
print(f"  eval.parquet  : {len(df)} rows  columns={df.columns.tolist()}  OK")
EOF
echo ">>> Step 2 passed."

# ── Step 3: Reward self-test ──────────────────────────────────────────────────
echo ""
echo ">>> Step 3: reward/math_reward.py self-test"
python reward/math_reward.py
echo ">>> Step 3 passed."

# ── Step 4: Key imports ───────────────────────────────────────────────────────
echo ""
echo ">>> Step 4: import verification"
python - <<'EOF'
import verl
import vllm
import ray
import torch
from torch.utils.tensorboard import SummaryWriter
import sympy
print("  verl          : ok")
print("  vllm          : ok")
print("  ray           : ok")
print("  torch         :", torch.__version__)
print("  tensorboard   : ok")
print("  sympy         :", sympy.__version__)
EOF
echo ">>> Step 4 passed."

echo ""
echo "================================================================"
echo "  === All checks passed — safe to sbatch ==="
echo "================================================================"
