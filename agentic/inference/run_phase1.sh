#!/usr/bin/env bash
# Phase 1: CPU-during-inference. Sustained agent-prompt load on local vLLM; perf scoped to the
# WHOLE engine (API server + VLLM::EngineCore). 4 stat passes (td2/cache/fp/mlp, no multiplexing)
# + 1 perf-record pass (function/flame profile). Warmup fence before any probe.
set +e
cd /home/mohamad/llm-service-kernel-latest/agentic/inference
. ../common/perf_events.sh; . ../common/lib_perf.sh
PERF=$(perf_bin) || { echo "no perf"; exit 1; }
SUDO="sudo"
OUT=runs/phase1; mkdir -p "$OUT"; rm -f "$OUT"/group_*.txt "$OUT"/perf.data
# whole engine: API server (comm=vllm) + worker (comm VLLM::EngineCor)
PIDS=$(ps -eo pid,comm | awk '$2=="vllm" || $2 ~ /VLLM::/ {print $1}' | paste -sd,)
echo "engine PIDS=$PIDS"; echo "$PIDS" > "$OUT/pids.txt"

# sustained load: concurrency 16, 256 out tokens (agentic-short), covers all passes
../bigcodebench/.venv/bin/python3 drive_load.py 480 16 256 > "$OUT/load.log" 2>&1 &
LOAD=$!
echo "$(date +%T) warmup 50s (kernels hot, CPU caches warm, turbo ramped)..."; sleep 50

run_stat(){ echo "$(date +%T) stat pass $1 (60s)"; $SUDO "$PERF" stat -e "$2" -p "$PIDS" -o "$OUT/group_$1.txt" -- sleep 60; }
run_stat TMA   "$PG_TD2"
run_stat CACHE "$PG_CACHE"
run_stat FP    "$PG_FP"
run_stat MLP   "$PG_MLP"

echo "$(date +%T) perf record (function profile, 30s)"
$SUDO "$PERF" record -g --call-graph dwarf,4096 -F 499 -p "$PIDS" -o "$OUT/perf.data" -- sleep 30 2>>"$OUT/record.log"
$SUDO chown "$(id -u):$(id -g)" "$OUT/perf.data" 2>/dev/null

kill "$LOAD" 2>/dev/null
echo "$(date +%T) ALLDONE -> $OUT"
tail -1 "$OUT/load.log"