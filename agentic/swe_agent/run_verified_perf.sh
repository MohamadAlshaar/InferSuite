#!/usr/bin/env bash
# Run N SWE-bench VERIFIED instances (parallel) under a SYSTEM-WIDE perf -> aggregate tool-exec
# CPU + GPU-generation time. Uses the FIXED thought_action config (default_backticks) so the
# agent actually reasons + runs commands. Clean output dir runs/verified (NO old broken data).
# Assumes the instance Docker images are ALREADY PULLED (pre-pull step, not measured).
# Usage: run_verified_perf.sh <N_instances> <num_workers>
set -uo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
. ../common/perf_events.sh
. ../common/lib_perf.sh
PERF="$(perf_bin)" || { echo "[verified] FATAL: no working perf"; exit 1; }
EVENTS="$(tma_group),mem_load_retired.l1_hit,mem_load_retired.l2_hit,mem_load_retired.l3_hit,mem_load_retired.l3_miss,fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.128b_packed_double,fp_arith_inst_retired.256b_packed_double,fp_arith_inst_retired.512b_packed_double"
VLLM_API_BASE="${VLLM_API_BASE:-http://localhost:8000/v1}"
MODEL="${MODEL:-qwen2.5-7b}"
export HOSTED_VLLM_API_BASE="$VLLM_API_BASE" HOSTED_VLLM_API_KEY="dummy" OPENAI_API_KEY="dummy"
N="${1:-10}"; WORKERS="${2:-4}"
# Default to the 10 LOCALLY-CACHED astropy Verified images -> ZERO docker pull. Override with FILTER.
FILTER="${FILTER:-astropy__astropy-(12907|13033|13236|13398|13453|13579|13977|14096|14182|14309)}"
OUT=runs/verified; PERFOUT="$OUT/perf"; mkdir -p "$PERFOUT"
echo "[verified] perf=$PERF | TMA=$(tma_group | cut -c1-30)... | filter=$FILTER x $WORKERS workers, model=$MODEL"

sweagent run-batch \
  --config external/SWE-agent/config/sweagent_0_7/07_thought_action_7b.yaml \
  --instances.type swe_bench \
  --instances.subset verified \
  --instances.split test \
  --instances.filter "$FILTER" \
  --agent.model.name "hosted_vllm/${MODEL}" \
  --agent.model.api_base "$VLLM_API_BASE" \
  --agent.model.api_key dummy \
  --agent.model.per_instance_cost_limit 0 \
  --agent.model.total_cost_limit 0 \
  --agent.model.per_instance_call_limit "${CALL_LIMIT:-12}" \
  --agent.model.max_input_tokens 28000 \
  --agent.model.max_output_tokens 4096 \
  --agent.model.temperature 0.5 \
  --agent.model.completion_kwargs '{"frequency_penalty":0.5,"presence_penalty":0.3}' \
  --agent.tools.execution_timeout 120 \
  --num_workers "$WORKERS" \
  --output_dir "$OUT" > "$OUT/batch.log" 2>&1 &
AG=$!
echo "$(date +%s.%N) RUN_START" > "$PERFOUT/markers.txt"
"$PERF" stat -e "$EVENTS" -a -I 1000 -x, -o "$PERFOUT/perf_timeline.csv" 2>/dev/null & PP=$!
( while kill -0 "$AG" 2>/dev/null; do
    printf "%s,%s\n" "$(date +%s.%N)" "$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits|head -1|tr -d ' ')"
    sleep 1; done ) > "$PERFOUT/gpu_timeline.csv" 2>/dev/null & GP=$!
VPIDS=$(pgrep -f "EngineCore|vllm.*serve" | paste -sd, -)
if [ -n "$VPIDS" ]; then "$PERF" stat -p "$VPIDS" -e "$(tma_group)" -I 1000 -x, -o "$PERFOUT/vllm_perf_timeline.csv" 2>/dev/null & VPP=$!; else VPP=""; fi

wait "$AG"
echo "$(date +%s.%N) RUN_END" >> "$PERFOUT/markers.txt"
kill -INT "$PP" ${VPP:+"$VPP"} 2>/dev/null; sleep 1; kill "$PP" "$GP" ${VPP:+"$VPP"} 2>/dev/null
echo "[verified] done -> $OUT"
