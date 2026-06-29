#!/usr/bin/env bash
# Serve Qwen2.5-7B-AWQ for the spin-vs-block experiment. CUDA_BLOCKING_SYNC (0=spin default, 1=block)
# is read by cudasync/sitecustomize.py in every vLLM process. Served name matches run_local.sh (qwen2.5-7b).
set +e
cd /home/mohamad/llm-service-kernel-latest/agentic/bigcodebench || exit 2
pkill -9 -f "Qwen2.5-7B-Instruct-AWQ" 2>/dev/null; sleep 3
export PYTHONPATH="/home/mohamad/llm-service-kernel-latest/agentic/inference/cudasync:${PYTHONPATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=0
if [ "${CUDA_BLOCKING_SYNC:-0}" = "1" ]; then
  # the real lever: force every CUDA event to BLOCKING_SYNC so cudaEventSynchronize sleeps (not spins).
  export LD_PRELOAD="/home/mohamad/llm-service-kernel-latest/agentic/inference/cudasync/evblock.so:${LD_PRELOAD:-}"
fi
echo "[serve] CUDA_BLOCKING_SYNC=${CUDA_BLOCKING_SYNC:-0}  (0=spin / 1=block)  LD_PRELOAD=${LD_PRELOAD:-none}"
exec .venv/bin/vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --max-model-len 32768 --gpu-memory-utilization 0.92 \
  --port 8000 --served-model-name qwen2.5-7b
