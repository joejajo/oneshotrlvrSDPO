#!/bin/bash
# Inner launch script — runs INSIDE the Apptainer container.
# Called by train_oneshot_sdpo.slurm via:
#   apptainer exec --nv ... bash scripts/_train_inner.sh
#
# LD_LIBRARY_PATH is set here (inside the container) so it is applied
# AFTER --nv injects its CUDA driver paths, not before.

set -euo pipefail

exec python -m verl.trainer.main_ppo "$@"
