#!/usr/bin/env bash
# Live perf harness for OpenClaw / WildClawBench (no replay -> single live run).
# Probes (mirror the SWE-agent suite, adapted to OpenClaw's single-container model):
#   1. task-container cgroup  -> OUTSIDE-inference CPU (agent + ALL tools: browser/bash/skills)
#        - fixed-counter TMA timeline (no multiplexing)
#        - deep GP aggregate (cache/fp/mlp) over the whole run (multiplexed, since no replay)
#   2. vLLM EngineCore cgroup -> DURING-inference CPU (TMA)  [sudo perf -a -G kubepods cgroup]
#   3. nvidia-smi GPU util
# markers.txt carries perf_start / agent_done epochs.
#
# Usage: run_perf.sh <task.md path, relative to external/WildClawBench>
set -uo pipefail
cd "$(dirname "$0")"
HERE="$(pwd)"
ROOT="external/WildClawBench"
# WORKING perf + correct TMA group via shared lib (verifies perf actually counts; handles the
# OEM-kernel wrapper mismatch + slots-vs-legacy auto-detect).
. "$HERE/../common/perf_events.sh"
. "$HERE/../common/lib_perf.sh"
PERF="$(perf_bin)" || { echo "[oc-perf] FATAL: no working perf binary (set PERF_HOST_BIN)"; exit 1; }
TASK="${1:?usage: run_perf.sh <task.md (relative to external/WildClawBench)>}"
OUT="$HERE/runs/perf"; mkdir -p "$OUT"

TMA="$(tma_group)"
# deep GP set (cache + fp + mlp) — multiplexed in one pass (no replay to split them).
# Full FP set incl ALL packed-double (was FP32-biased -> falsely "0% AVX") + uops_executed.thread (SMT-correct).
DEEP="cycles,instructions,mem_load_retired.l1_hit,mem_load_retired.l2_hit,mem_load_retired.l3_hit,mem_load_retired.l3_miss,fp_arith_inst_retired.scalar_single,fp_arith_inst_retired.128b_packed_single,fp_arith_inst_retired.256b_packed_single,fp_arith_inst_retired.512b_packed_single,fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.128b_packed_double,fp_arith_inst_retired.256b_packed_double,fp_arith_inst_retired.512b_packed_double,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles,uops_executed.thread,uops_retired.slots"
echo "[oc-perf] perf=$PERF ; TMA=$TMA"

echo "[oc-perf] launching agent on task: $TASK"
( cd "$ROOT" && . .venv/bin/activate && \
  python3 eval/run_batch.py --task "$TASK" --models-config my_api.json \
    --model my-openai-proxy/claude-sonnet-4-6 --parallel 1 ) > /tmp/oc_run.log 2>&1 &
AG=$!

echo "[oc-perf] waiting for task container (by image)..."
CID=""
for i in $(seq 1 150); do
  CID=$(docker ps -q --filter ancestor=wildclawbench-ubuntu:v1.3 | head -1)
  [[ -n "$CID" ]] && break
  kill -0 "$AG" 2>/dev/null || { echo "[oc-perf] agent exited before container appeared"; break; }
  sleep 2
done
[[ -z "$CID" ]] && { echo "[oc-perf] no task container; aborting"; wait "$AG"; exit 1; }
FULL=$(docker inspect -f '{{.Id}}' "$CID")
CG="system.slice/docker-${FULL}.scope"
echo "[oc-perf] container=$CID cgroup=$CG"

# WARMUP FENCE: the container exists during warmup (pip install playwright + chromium
# download = heavy CPU that is NOT agent tool-exec). Wait until the harness logs that the
# agent loop is starting, THEN begin measuring -> excludes warmup/gateway-startup.
echo "[oc-perf] waiting for warmup to finish (agent loop start)..."
for i in $(seq 1 240); do
  grep -q "Waiting for agent to finish" /tmp/oc_run.log 2>/dev/null && break
  kill -0 "$AG" 2>/dev/null || { echo "[oc-perf] agent exited during warmup"; break; }
  sleep 1
done
echo "$(date +%s.%N) perf_start cgroup=$CG cid=$CID" > "$OUT/markers.txt"
echo "[oc-perf] warmup done -> probes starting"

# probe 1: container timeline — ONE perf instance with TMA(fixed) + GP(cache/fp/mlp).
# Single instance: topdown/slots/cycles/instructions stay on FIXED counters (no mux, TMA
# clean); only the GP events multiplex among the 8 GP counters. (Two separate instances
# fought over the fixed counters and degraded TMA to ~60%.)
GP="mem_load_retired.l1_hit,mem_load_retired.l2_hit,mem_load_retired.l3_hit,mem_load_retired.l3_miss,fp_arith_inst_retired.scalar_single,fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.256b_packed_single,fp_arith_inst_retired.256b_packed_double,fp_arith_inst_retired.512b_packed_single,fp_arith_inst_retired.512b_packed_double"
"$PERF" stat -e "${TMA},${GP}" -G "$CG" -a -I 1000 -x, -o "$OUT/container_timeline.csv" & P1=$!
P2=""

# probe 2: vLLM EngineCore cgroup (during-inference). Find the kubepods cgroup of EngineCore.
P3=""
VCG=$(for d in /proc/[0-9]*; do
        cl=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null)
        [[ "$cl" == *EngineCore* ]] || continue
        cg=$(sudo cat "$d/cgroup" 2>/dev/null)
        [[ "$cg" == *kubepods* ]] && { echo "$cg" | sed 's/^0:://; s#^/##'; break; }
      done)
if [[ -n "$VCG" ]]; then
  sudo "$PERF" stat -e "$TMA" -a -G "$VCG" -o "$OUT/vllm_tma.txt" & P3=$!
  echo "[oc-perf] vLLM cgroup attached"
else
  echo "[oc-perf] WARN: vLLM EngineCore cgroup not found"
fi

# probe 3: GPU sampler (epoch,util,mem)
( while sleep 1; do
    printf "%s,%s\n" "$(date +%s.%N)" \
      "$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  done ) > "$OUT/gpu_timeline.csv" 2>/dev/null & GP=$!

# probe 4: vLLM EngineCore CPU cores over time (for the CPU core-seconds donut)
/usr/bin/python3 vllm_cpu_sampler.py "$OUT/vllm_timeline.csv" & VP=$!

echo "[oc-perf] probes running; waiting for agent to finish..."
wait "$AG"
echo "$(date +%s.%N) agent_done" >> "$OUT/markers.txt"
kill -INT "$P1" ${P2:+$P2} ${P3:+$P3} 2>/dev/null; sleep 1
kill "$P1" ${P2:+$P2} ${P3:+$P3} "$GP" ${VP:+$VP} 2>/dev/null
echo "[oc-perf] done -> $OUT"