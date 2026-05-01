#!/bin/bash
# Run this on the LOGIN node ONCE to clean up the pkgs/ directory after the
# bitsandbytes pip install accidentally pulled in a conflicting torch 2.11.0,
# numpy 2.4.4, cuda-bindings 13.2.0, etc. that would shadow the container's
# torch 2.10.0+cu129 / vllm 0.17.0 environment via PYTHONPATH.
#
# Reinstalls only the packages CLAUDE.md says belong in pkgs/:
#   latex2sympy2, regex, antlr4-python3-runtime==4.7.2

set -euo pipefail

PROJECT_ROOT=/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO
PKGS_DIR=${PROJECT_ROOT}/pkgs
SIF=/home/woody/iwi7/iwi7107h/images/verl_vllm017_latest.sif

if [ ! -f "${SIF}" ]; then
    echo "ERROR: Apptainer image not found at ${SIF}"
    exit 1
fi

echo "Wiping ${PKGS_DIR} (was contaminated by bitsandbytes install) ..."
rm -rf "${PKGS_DIR}"
mkdir -p "${PKGS_DIR}"

echo "Reinstalling math grading deps (latex2sympy2, regex, antlr4) ..."
apptainer exec \
    --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    "${SIF}" \
    pip install --target="${PKGS_DIR}" --no-deps \
        latex2sympy2 \
        regex \
        antlr4-python3-runtime==4.7.2

echo
echo "Verifying ..."
apptainer exec --nv \
    --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    "${SIF}" \
    bash -c "PYTHONPATH='${PKGS_DIR}' python -c '
import latex2sympy2, regex, antlr4
print(\"latex2sympy2:\", latex2sympy2.__name__)
print(\"regex:        ok\")
print(\"antlr4:       ok\")
'"

echo
echo "Done. pkgs/ is clean and ready."
