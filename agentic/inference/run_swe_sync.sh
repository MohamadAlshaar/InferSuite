#!/usr/bin/env bash
# SWE-agent spin-vs-block A/B. Serve vLLM in MODE, run ONE SWE-agent instance (single-stream) for a
# bounded window while perf-stat'ing the vLLM ENGINE (the phantom-spin core) and sampling GPU util.
# Key comparable metric (robust to the 7B's non-deterministic trajectory): engine core-seconds PER
# inference-second (= task-clock / GPU-active time). Spin ~1.0, block should be ~0.3, agent wall ~same.
# Usage: run_swe_sync.sh {spin|block} [window_s]
set -uo pipefail
MODE="${1:-spin}"; WIN="${2:-420}"
ROOT=/home/mohamad/llm-service-kernel-latest/agentic
cd "$ROOT/inference"; mkdir -p runs/sync
. ../common/perf_events.sh; . ../common/lib_perf.sh
PERF=$(perf_bin) || { echo "no perf"; exit 1; }
kill_vllm(){ pkill -9 -f "Qwen2.5-7B-Instruct-AWQ" 2>/dev/null
  for p in $(ps -eo pid,comm | awk '$2 ~ /VLLM::/ || $2=="vllm" {print $1}'); do kill -9 "$p" 2>/dev/null; done
  for i in $(seq 1 25); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null); [ "${m:-9999}" -lt 800 ] 2>/dev/null && break; sleep 2; done; }

export CUDA_BLOCKING_SYNC=$([ "$MODE" = block ] && echo 1 || echo 0)
kill_vllm
nohup bash serve_sync.sh > "runs/sync/serve_swe_$MODE.log" 2>&1 &
echo "[$MODE] waiting for vLLM ..."; for i in $(seq 1 160); do curl -sf localhost:8000/health >/dev/null 2>&1 && break; sleep 2; done
curl -sf localhost:8000/health >/dev/null 2>&1 || { echo "server down"; tail -5 "runs/sync/serve_swe_$MODE.log"; exit 1; }
grep -iE "evblock|spin mode" "runs/sync/serve_swe_$MODE.log" | head -2
PIDS=$(ps -eo pid,comm | awk '$2=="vllm" || $2 ~ /VLLM::/ {print $1}' | paste -sd,)
echo "[$MODE] engine pids: $PIDS"

# perf over a max WIN window (SIGINT'd when agent finishes -> summary for the real elapsed)
"$PERF" stat -p "$PIDS" -e task-clock,cycles,instructions -o "runs/sync/swe_perf_$MODE.txt" -- sleep "$WIN" & PERFPID=$!
# GPU-util sampler (epoch,util) -> inference-active fraction
( while sleep 1; do printf "%s,%s\n" "$(date +%s)" "$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)"; done ) > "runs/sync/gpu_$MODE.csv" & GPUPID=$!

echo "[$MODE] running SWE-agent (bounded ${WIN}s) ..."
t0=$(date +%s)
timeout "$WIN" bash -c "cd '$ROOT/swe_agent' && bash run_local.sh 1" > "runs/sync/agent_$MODE.log" 2>&1
t1=$(date +%s); WALL=$((t1-t0))
kill -INT "$PERFPID" 2>/dev/null; sleep 2; kill "$GPUPID" 2>/dev/null; wait "$PERFPID" 2>/dev/null

# GPU-active (inference) seconds = samples with util>0
GPUACT=$(awk -F, '$2>0{n++} END{print n+0}' "runs/sync/gpu_$MODE.csv")
echo "=== [$MODE] RESULT ==="
echo "[$MODE] agent wall = ${WALL}s | GPU-active (inference) ~= ${GPUACT}s"
grep -iE "task-clock|cycles|instructions|CPUs utilized|insn per" "runs/sync/swe_perf_$MODE.txt"
TC=$(awk '/task-clock/{gsub(/,/,"",$1);print $1}' "runs/sync/swe_perf_$MODE.txt" | head -1)
python3 -c "
tc=${TC:-0}/1000.0; ga=${GPUACT:-0}
print(f'[$MODE] engine core-seconds = {tc:.1f}s ; core-seconds per inference-second = {tc/ga:.2f}' if ga>0 else '[no gpu-active]')"
echo "[$MODE] agent log tail:"; tail -4 "runs/sync/agent_$MODE.log"
kill_vllm
echo "[$MODE] done"
