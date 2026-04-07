#!/bin/bash
# Standalone vLLM smoke test to isolate cuBLAS/CUDA errors from Ray/SDPO

export VLLM_DISABLE_COMPILE_CACHE=1
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export CUBLAS_WORKSPACE_CONFIG=:0:0
export CUBLASLT_WORKSPACE_SIZE=0

echo "Running standalone vLLM serve test..."

apptainer exec --nv \
  --bind /home/woody/iwi7/iwi7107h/models:/home/woody/iwi7/iwi7107h/models \
  /home/woody/iwi7/iwi7107h/images/verl_vllm017_latest.sif \
  vllm serve /home/woody/iwi7/iwi7107h/models/Qwen2.5-Math-1.5B \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.5 \
    --max-model-len 4096 \
    --max-num-seqs 64 \
    --enforce-eager
