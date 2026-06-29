#!/usr/bin/env bash
cd /home/mohamad/llm-service-kernel-latest/agentic/inference
for m in ngram3 ngram5 ngram8; do
  pkill -9 -f "Qwen2.5-7B" 2>/dev/null
  for p in $(ps -eo pid,comm | awk '$2 ~ /VLLM::/ || $2=="vllm" {print $1}'); do kill -9 "$p" 2>/dev/null; done
  for i in $(seq 1 20); do mm=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null); [ "${mm:-9999}" -lt 800 ] 2>/dev/null && break; sleep 2; done
  ../bigcodebench/.venv/bin/python3 measure_spec.py "$m" > "runs/sync/sb_$m.log" 2>&1
done
echo ALLDONE
