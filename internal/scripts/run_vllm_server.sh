#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_DIR="${REPO_ROOT}/Qwen2.5-0.5B-Instruct"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"

# Optional if needed during bring-up
# export VLLM_TORCH_COMPILE=0
# export VLLM_USE_V1=0

PORT="${VLLM_PORT:-8001}"
HOST="${VLLM_HOST:-127.0.0.1}"
MODEL_NAME="${SERVED_MODEL_NAME:-qwen2.5-0.5b}"
GPU_UTIL="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_LEN="${MAX_MODEL_LEN:-4096}"

echo "[vllm] repo root: ${REPO_ROOT}"
echo "[vllm] model dir: ${MODEL_DIR}"
echo "[vllm] host: ${HOST}"
echo "[vllm] port: ${PORT}"
echo "[vllm] served model name: ${MODEL_NAME}"
echo "[vllm] gpu memory utilization: ${GPU_UTIL}"
echo "[vllm] max model len: ${MAX_LEN}"

exec vllm serve "${MODEL_DIR}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name "${MODEL_NAME}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --max-model-len "${MAX_LEN}"

