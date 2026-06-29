#!/usr/bin/env bash
# Validate the spin<->block lever. Serve in MODE, drive ONE sustained single-stream generation, and
# perf-stat the vLLM engine processes for 10s during it. Spin => ~1 CPU busy, high IPC (phantom spin);
# block => core near-idle. Throughput (tok/s) should be ~unchanged. Usage: validate_sync.sh {spin|block}
set -uo pipefail
MODE="${1:-spin}"
cd /home/mohamad/llm-service-kernel-latest/agentic/inference
mkdir -p runs/sync
. ../common/perf_events.sh; . ../common/lib_perf.sh
PERF=$(perf_bin) || { echo "no working perf"; exit 1; }
kill_vllm(){   # kill model proc AND the EngineCore (comm VLLM::, cmdline has no 'Qwen'), wait for GPU to free
  pkill -9 -f "Qwen2.5-7B-Instruct-AWQ" 2>/dev/null
  for p in $(ps -eo pid,comm | awk '$2 ~ /VLLM::/ || $2=="vllm" {print $1}'); do kill -9 "$p" 2>/dev/null; done
  for i in $(seq 1 25); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null); [ "${m:-9999}" -lt 800 ] 2>/dev/null && break; sleep 2; done
}
export CUDA_BLOCKING_SYNC=$([ "$MODE" = block ] && echo 1 || echo 0)
kill_vllm
nohup bash serve_sync.sh > "runs/sync/serve_$MODE.log" 2>&1 &
echo "[$MODE] waiting for vLLM /health ..."
for i in $(seq 1 160); do curl -sf localhost:8000/health >/dev/null 2>&1 && { echo "[$MODE] ready"; break; }; sleep 2; done
curl -sf localhost:8000/health >/dev/null 2>&1 || { echo "[$MODE] server never came up"; tail -5 "runs/sync/serve_$MODE.log"; exit 1; }
sleep 2
echo "[$MODE] cudasync lines:"; grep -i cudasync "runs/sync/serve_$MODE.log" | tail -4
PIDS=$(ps -eo pid,comm | awk '$2=="vllm" || $2 ~ /VLLM::/ {print $1}' | paste -sd,)
echo "[$MODE] engine pids: $PIDS"
# one long single-stream generation (ignore_eos so it runs through the window)
t0=$(date +%s.%N)
curl -s localhost:8000/v1/completions -H 'content-type: application/json' \
  -d '{"model":"qwen2.5-7b","prompt":"Write a very long, detailed technical essay about computer architecture.","max_tokens":1600,"temperature":0.7,"ignore_eos":true}' \
  -o "runs/sync/gen_$MODE.json" &
GEN=$!
sleep 3   # let decode ramp
echo "=== [$MODE] perf stat engine, 10s during single-stream decode ==="
"$PERF" stat -p "$PIDS" -e task-clock,cycles,instructions -- sleep 10 2>&1 | grep -iE "task-clock|cycles|instructions|CPUs utilized|insn per"
echo "[$MODE] GPU util during: $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader)"
wait "$GEN" 2>/dev/null
t1=$(date +%s.%N)
ntok=$(python3 -c "import json;print(json.load(open('runs/sync/gen_$MODE.json'))['usage']['completion_tokens'])" 2>/dev/null || echo "?")
echo "[$MODE] generated $ntok tokens in $(python3 -c "print(f'{$t1-$t0:.1f}')")s  (tok/s = $(python3 -c "print(f'{$ntok/($t1-$t0):.1f}')" 2>/dev/null || echo ?))"
kill_vllm
echo "[$MODE] done"
