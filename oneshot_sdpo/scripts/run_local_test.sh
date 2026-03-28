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

# ── Step 1: Validate committed parquet files ──────────────────────────────────
echo ""
echo ">>> Step 1: parquet file validation"

# Check files exist and start with PAR1 magic bytes
for f in data/pi1_r128.parquet data/math500.parquet; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: ${f} not found" >&2
        exit 1
    fi
    magic=$(head -c 4 "${f}" | cat -v)
    if [[ "${magic}" != "PAR1" ]]; then
        echo "ERROR: ${f} does not look like a parquet file (bad magic bytes)" >&2
        exit 1
    fi
    echo "  ${f} : exists, PAR1 magic bytes OK"
done

python - <<'EOF'
import pandas as pd

train = pd.read_parquet("data/pi1_r128.parquet")
assert len(train) == 128, f"Expected 128 rows in pi1_r128.parquet, got {len(train)}"
expected_cols = {"data_source", "prompt", "ability", "reward_model", "extra_info"}
assert expected_cols.issubset(set(train.columns)), (
    f"Missing columns in pi1_r128.parquet: {expected_cols - set(train.columns)}"
)
print(f"  pi1_r128.parquet  : {len(train)} rows  columns={train.columns.tolist()}  OK")

math500 = pd.read_parquet("data/math500.parquet")
assert len(math500) == 500, f"Expected 500 rows in math500.parquet, got {len(math500)}"
assert expected_cols.issubset(set(math500.columns)), (
    f"Missing columns in math500.parquet: {expected_cols - set(math500.columns)}"
)
print(f"  math500.parquet   : {len(math500)} rows  columns={math500.columns.tolist()}  OK")
EOF
echo ">>> Step 1 passed."

# ── Step 2: Reward self-test ──────────────────────────────────────────────────
echo ""
echo ">>> Step 2: reward/math_reward.py self-test"
python reward/math_reward.py
echo ">>> Step 2 passed."

# ── Step 3: Key imports ───────────────────────────────────────────────────────
echo ""
echo ">>> Step 3: import verification"
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
echo ">>> Step 3 passed."

echo ""
echo "================================================================"
echo "  === All checks passed — safe to sbatch ==="
echo "================================================================"
