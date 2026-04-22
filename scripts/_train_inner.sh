#!/bin/bash
# Inner launch script — runs INSIDE the Apptainer container.
# Called by train_oneshot_sdpo.slurm via:
#   apptainer exec --nv ... bash scripts/_train_inner.sh <hydra-args>
#
# Sets LD_LIBRARY_PATH and LD_PRELOAD here (inside the container) AFTER
# --nv has injected its CUDA driver paths.

set -euo pipefail

# ── Locate libcupti.so.12 ────────────────────────────────────────────────────
# torch/lib/libtorch_cpu.so is dynamically linked against libcupti.so.12.
# On this HPC, --nv does not inject it; we must find and expose it ourselves.

KNOWN_CUPTI_PATH="/usr/local/lib/python3.12/dist-packages/nvidia/cuda_cupti/lib"

if [ -f "${KNOWN_CUPTI_PATH}/libcupti.so.12" ]; then
    CUPTI_DIR="${KNOWN_CUPTI_PATH}"
else
    # Search the container for libcupti.so.12 (slow, only runs if not at known path)
    echo "[_train_inner] libcupti.so.12 not at expected path — searching..." >&2
    CUPTI_DIR=""
    while IFS= read -r found; do
        CUPTI_DIR=$(dirname "${found}")
        break
    done < <(find /usr/local -name "libcupti.so.12" 2>/dev/null)
fi

if [ -n "${CUPTI_DIR}" ]; then
    echo "[_train_inner] cupti dir : ${CUPTI_DIR}" >&2
    export LD_PRELOAD="${CUPTI_DIR}/libcupti.so.12${LD_PRELOAD:+:${LD_PRELOAD}}"
    export LD_LIBRARY_PATH="${CUPTI_DIR}:${CUDA_HOME:-/usr/local/cuda}/lib64:${LD_LIBRARY_PATH:-}"
else
    echo "[_train_inner] WARNING: libcupti.so.12 not found — torch may fail to import" >&2
fi

echo "[_train_inner] LD_PRELOAD    : ${LD_PRELOAD:-}" >&2
echo "[_train_inner] LD_LIBRARY_PATH (first 200): ${LD_LIBRARY_PATH:0:200}" >&2

exec python -m verl.trainer.main_ppo "$@"
