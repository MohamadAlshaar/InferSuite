#!/usr/bin/env bash
# Serve a 32B on the H100 for orchestration capture (PRODUCTION config: cudagraphs on,
# prefix caching on -- NOT the ncu enforce_eager profiling config).
# CUDA_BLOCKING_SYNC=1 -> evblock (host thread BLOCKS on sync); 0 (default) -> spin (phantom busy-wait).
set +e
MODEL="${MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct}"
NAME="${NAME:-coder-32b}"
CUDADIR="$HOME/bcb/cudasync"
# kill launcher AND the spawned EngineCore worker (the latter holds the GPU and is NOT matched by "vllm serve")
pkill -9 -f "vllm serve" 2>/dev/null; pkill -9 -f "VLLM::EngineCore" 2>/dev/null; pkill -9 -f "multiprocessing.spawn" 2>/dev/null; sleep 6
source "$HOME/vllmenv/bin/activate"
export PYTHONPATH="$CUDADIR:${PYTHONPATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=0
if [ "${CUDA_BLOCKING_SYNC:-0}" = "1" ]; then
  export LD_PRELOAD="$CUDADIR/evblock.so:${LD_PRELOAD:-}"
fi
echo "[serve] model=$MODEL name=$NAME BLOCKING_SYNC=${CUDA_BLOCKING_SYNC:-0} LD_PRELOAD=${LD_PRELOAD:-none}"
exec vllm serve "$MODEL" \
  --max-model-len 16384 --gpu-memory-utilization 0.92 \
  --max-num-seqs 16 --port 8000 --served-model-name "$NAME" \
  --disable-log-stats
