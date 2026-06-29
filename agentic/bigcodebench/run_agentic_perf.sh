#!/usr/bin/env bash
# Run agentic BigCodeBench under measurement -> TIME donut (GPU-gen vs CPU tool-exec).
# Markers (RUN_START/END + toolexec phases) come from agentic_bcb.py; GPU sampler gives
# inference time. (Tool-exec microarch ~= the one-shot code-exec TMA we already have, since
# it's the same test code running; a clean cgroup-scoped TMA can be added later.)
set -uo pipefail
cd "$(dirname "$0")"
. .venv/bin/activate
# perf split into GPU-generation vs CPU-tool-exec via markers from agentic_bcb.py.
. ../common/perf_events.sh; . ../common/lib_perf.sh
PERF="$(perf_bin)" || { echo "[agentic-bcb] no working perf"; exit 1; }
EVENTS="$(tma_group),mem_load_retired.l1_hit,mem_load_retired.l2_hit,mem_load_retired.l3_hit,mem_load_retired.l3_miss,fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.128b_packed_double,fp_arith_inst_retired.256b_packed_double,fp_arith_inst_retired.512b_packed_double"
N="${1:-12}"; TURNS="${2:-3}"
OUT=runs/agentic; mkdir -p "$OUT"
rm -f /tmp/bcb_agentic_markers.txt "$OUT/gpu_timeline.csv" "$OUT/perf_timeline.csv" "$OUT/vllm_perf_timeline.csv"
echo "[agentic-bcb] perf=$PERF ; model=${MODEL:-?}"
echo "[agentic-bcb] launching loop: $N tasks x $TURNS turns ..."
python3 agentic_bcb.py "$N" "$TURNS" > /tmp/bcb_agentic.log 2>&1 &
AG=$!
# system-wide interval perf -> split by toolexec markers into generation(vLLM) vs tool-exec CPU
"$PERF" stat -e "$EVENTS" -a -I 1000 -x, -o "$OUT/perf_timeline.csv" 2>/dev/null &
PP=$!
# vLLM-engine perf (the inference CPU during generation)
VPIDS=$(pgrep -f "EngineCore|vllm.*serve" | paste -sd, -)
if [ -n "$VPIDS" ]; then "$PERF" stat -p "$VPIDS" -e "$(tma_group)" -I 1000 -x, -o "$OUT/vllm_perf_timeline.csv" 2>/dev/null & VPP=$!; else VPP=""; fi
# GPU sampler (epoch,util) -> inference time
( while kill -0 "$AG" 2>/dev/null; do
    printf "%s,%s\n" "$(date +%s.%N)" "$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    sleep 1
  done ) > "$OUT/gpu_timeline.csv" 2>/dev/null &
GP=$!
wait "$AG"; kill "$GP" 2>/dev/null
kill -INT "$PP" ${VPP:+"$VPP"} 2>/dev/null; sleep 1; kill "$PP" ${VPP:+"$VPP"} 2>/dev/null
cp /tmp/bcb_agentic_markers.txt "$OUT/markers.txt" 2>/dev/null
echo "[agentic-bcb] done -> $OUT"
tail -3 /tmp/bcb_agentic.log 2>/dev/null | grep -ivE "^\s*$"
