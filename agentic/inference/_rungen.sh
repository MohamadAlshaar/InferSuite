#!/usr/bin/env bash
cd /home/mohamad/llm-service-kernel-latest/agentic/inference
pkill -9 -f "Qwen2.5-7B" 2>/dev/null
for p in $(ps -eo pid,comm|awk '$2~/VLLM::/||$2=="vllm"{print $1}'); do kill -9 "$p" 2>/dev/null; done
sleep 4
../bigcodebench/.venv/bin/python3 gen_chains.py
echo "GEN_EXIT=$?"
