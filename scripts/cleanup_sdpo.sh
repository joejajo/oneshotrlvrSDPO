#!/bin/bash
# cleanup_sdpo.sh — Remove files from SDPO/ that are not needed for the
# one-shot RLVR + SDPO training run. Safe to run on HPC after cloning
# https://github.com/lasgroup/SDPO into oneshotrlvrSDPO/SDPO/
#
# KEEPS: all verl/ Python source needed by python -m verl.trainer.main_ppo
#        (FSDP actor, vLLM rollout, agent_loop, reward_manager, configs)
# REMOVES: examples, experiments, scripts, SFT trainers, megatron backend,
#          sglang backend, VLA module, build artifacts, __pycache__

set -euo pipefail

SDPO_DIR="${1:-/home/woody/iwi7/iwi7107h/oneshotrlvrSDPO/SDPO}"

if [ ! -d "$SDPO_DIR" ]; then
    echo "ERROR: SDPO directory not found: $SDPO_DIR"
    echo "Usage: $0 [/path/to/SDPO]"
    exit 1
fi

echo "=== Cleaning up $SDPO_DIR ==="

# ── Directories to fully remove ──────────────────────────────────────────────

# Examples and experiment configs — not imported at runtime
rm -rf "$SDPO_DIR/examples"
echo "removed: examples/"

rm -rf "$SDPO_DIR/experiments"
echo "removed: experiments/"

# SDPO's own launch scripts (we have our own slurm script)
rm -rf "$SDPO_DIR/training"
echo "removed: training/"

# Utility scripts — model converters, install helpers (not needed at runtime)
rm -rf "$SDPO_DIR/scripts"
echo "removed: scripts/"

# Build artifact from pip install -e (not needed — we use PYTHONPATH)
rm -rf "$SDPO_DIR/verl.egg-info"
echo "removed: verl.egg-info/"

# ── Unneeded verl/ experimental modules ──────────────────────────────────────

# Vision-Language-Action (robotics) — unrelated to math training
rm -rf "$SDPO_DIR/verl/experimental/vla"
echo "removed: verl/experimental/vla/"

# Async policy variants we don't use
rm -rf "$SDPO_DIR/verl/experimental/fully_async_policy"
rm -rf "$SDPO_DIR/verl/experimental/one_step_off_policy"
rm -rf "$SDPO_DIR/verl/experimental/dynamic_dataset"
rm -rf "$SDPO_DIR/verl/experimental/transfer_queue"
echo "removed: verl/experimental/{fully_async_policy,one_step_off_policy,dynamic_dataset,transfer_queue}/"

# ── Unneeded backends ────────────────────────────────────────────────────────

# SGLang backend — we use vLLM
rm -rf "$SDPO_DIR/verl/utils/sglang"
echo "removed: verl/utils/sglang/"

# Megatron-LM backend — we use FSDP
rm -rf "$SDPO_DIR/verl/utils/megatron"
echo "removed: verl/utils/megatron/"
rm -f "$SDPO_DIR/verl/utils/megatron_utils.py"
rm -f "$SDPO_DIR/verl/utils/megatron_peft_utils.py"
echo "removed: verl/utils/megatron_utils.py, megatron_peft_utils.py"
rm -f "$SDPO_DIR/verl/workers/megatron_workers.py"
echo "removed: verl/workers/megatron_workers.py"

# NPU-specific flash attention — we run on A100 (CUDA)
rm -f "$SDPO_DIR/verl/utils/npu_flash_attn_utils.py"
echo "removed: verl/utils/npu_flash_attn_utils.py"

# ── Unneeded trainer entry points ────────────────────────────────────────────

# SFT trainers — we only do RL (no supervised fine-tuning)
rm -f "$SDPO_DIR/verl/trainer/sft_trainer.py"
rm -f "$SDPO_DIR/verl/trainer/sft_trainer_ray.py"
rm -f "$SDPO_DIR/verl/trainer/fsdp_sft_trainer.py"
echo "removed: verl/trainer/sft_trainer*.py, fsdp_sft_trainer.py"

# Standalone eval/generation entry points (eval runs through main_ppo val loop)
rm -f "$SDPO_DIR/verl/trainer/main_eval.py"
rm -f "$SDPO_DIR/verl/trainer/main_generation.py"
rm -f "$SDPO_DIR/verl/trainer/main_generation_server.py"
echo "removed: verl/trainer/main_eval.py, main_generation*.py"

# ── Root-level reference scripts ─────────────────────────────────────────────

rm -f "$SDPO_DIR/run_local_grpo.sh"
rm -f "$SDPO_DIR/run_local_sdpo.sh"
rm -f "$SDPO_DIR/run_local_test.sh"
echo "removed: run_local_{grpo,sdpo,test}.sh"

# Duplicate requirements (all pre-installed in Apptainer image)
rm -f "$SDPO_DIR/requirements-cuda.txt"
rm -f "$SDPO_DIR/requirements-full.txt"
rm -f "$SDPO_DIR/requirements-test.txt"
echo "removed: requirements-{cuda,full,test}.txt"

# ── __pycache__ / compiled bytecode ──────────────────────────────────────────

find "$SDPO_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$SDPO_DIR" -name "*.pyc" -o -name "*.pyo" | xargs rm -f 2>/dev/null || true
echo "removed: all __pycache__/ dirs and .pyc/.pyo files"

# ── Summary ──────────────────────────────────────────────────────────────────

echo ""
echo "=== Cleanup complete. Remaining SDPO/ structure ==="
find "$SDPO_DIR" -maxdepth 3 -not -path '*/__pycache__/*' \
    | sed "s|$SDPO_DIR/||" | sort | head -80
echo ""
echo "Key files present (required for training):"
for f in \
    "verl/__init__.py" \
    "verl/protocol.py" \
    "verl/trainer/main_ppo.py" \
    "verl/trainer/ppo/ray_trainer.py" \
    "verl/trainer/ppo/core_algos.py" \
    "verl/trainer/config/ppo_trainer.yaml" \
    "verl/workers/fsdp_workers.py" \
    "verl/workers/reward_manager/naive.py" \
    "verl/experimental/agent_loop/agent_loop.py" \
    "verl/workers/actor/dp_actor.py" \
; do
    if [ -f "$SDPO_DIR/$f" ]; then
        echo "  OK  $f"
    else
        echo "  MISSING  $f  <-- CHECK THIS"
    fi
done
