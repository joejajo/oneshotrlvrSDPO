#!/bin/bash
# Inner launch script — runs INSIDE the Apptainer container.
# Called by train_oneshot_sdpo.slurm via:
#   apptainer exec --nv ... bash scripts/_train_inner.sh
#
# LD_LIBRARY_PATH is set here (inside the container) so it is applied
# AFTER --nv injects its CUDA driver paths, not before.

set -euo pipefail

# CUPTI: libcupti.so.12 lives inside the nvidia Python package inside the container.
export LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/nvidia/cuda_cupti/lib:${CUDA_HOME:-/usr/local/cuda}/lib64:${CUDA_HOME:-/usr/local/cuda}/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}"

exec python -m verl.trainer.main_ppo "$@"
