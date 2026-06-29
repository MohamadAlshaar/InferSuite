#!/usr/bin/env bash
# Serve Qwen2.5-7B-AWQ on the A2000 for the during-inference measurement.
# flashinfer removed (no nvcc) -> native sampler; Triton attention (no nvcc JIT).
set +e
cd /home/mohamad/llm-service-kernel-latest/agentic/bigcodebench || exit 2
pkill -9 -f "Qwen2.5-7B-Instruct-AWQ" 2>/dev/null
sleep 3
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
exec .venv/bin/vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --port 8000 \
  --served-model-name qwen7b
