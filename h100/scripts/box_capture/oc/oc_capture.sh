#!/usr/bin/env bash
# OpenClaw two-view capture, H100 PMC-safe (4 groups core/fp/mem/stall; this KVM VM has NO Intel TMA).
# One live agent run PER counter-group (OpenClaw is non-deterministic -> can't record-replay a browser
# agent; matches the CANONICAL live-repeated methodology), fenced past warmup (pip/chromium install),
# perf scoped to EITHER the task container cgroup (SCOPE=outside = agent+tools CPU) OR the vLLM engine
# PIDs (SCOPE=during = serving orchestration). Separate runs -> no PMU contention/multiplexing.
# Emits tool_<group>.csv in the SAME `perf stat -x,` aggregate format as the SWE tool-exec captures.
# Usage: SCOPE=outside|during TASK=<relpath> LABEL=<name> bash oc_capture.sh
set -uo pipefail
cd /home/ubuntu/oc/WildClawBench
. .venv/bin/activate
TASK="${TASK:?task relpath}"; LABEL="${LABEL:?label}"; SCOPE="${SCOPE:-outside}"
IMG=wildclawbench-ubuntu:v1.3
OUT="/home/ubuntu/oc/runs/${LABEL}_${SCOPE}"; mkdir -p "$OUT"
sudo sysctl -w kernel.perf_event_paranoid=-1 >/dev/null 2>&1
sudo sysctl -w kernel.kptr_restrict=0    >/dev/null 2>&1
PERF="perf"

core="cycles,instructions,cache-references,cache-misses,branch-instructions,branch-misses"
fp="cycles,instructions,fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.128b_packed_double,fp_arith_inst_retired.256b_packed_double,fp_arith_inst_retired.512b_packed_double"
mem="cycles,instructions,mem_load_retired.l1_hit,mem_load_retired.l2_hit,mem_load_retired.l3_hit,mem_load_retired.l3_miss"
stall="cycles,instructions,cycle_activity.stalls_total,cycle_activity.stalls_l3_miss,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles"

cleanup(){ docker ps -aq --filter ancestor=$IMG 2>/dev/null | xargs -r docker rm -f >/dev/null 2>&1; }

for g in core fp mem stall; do
  EV=$(eval echo "\$$g")
  log="$OUT/agent_${g}.log"
  echo "== $LABEL $SCOPE $g =="
  cleanup
  ( python3 eval/run_batch.py --task "$TASK" --models-config my_api.json \
      --model my-openai-proxy/instruct-32b --parallel 1 </dev/null ) > "$log" 2>&1 &
  AG=$!
  # wait for container
  CID=""; for i in $(seq 1 150); do CID=$(docker ps -q --filter ancestor=$IMG | head -1); [ -n "$CID" ] && break; kill -0 $AG 2>/dev/null || break; sleep 2; done
  if [ -z "$CID" ]; then echo "  NO_CONTAINER"; wait $AG 2>/dev/null; continue; fi
  # warmup fence: start counting only once the agent loop begins (skip pip/chromium install)
  for i in $(seq 1 300); do grep -q "Waiting for agent to finish" "$log" 2>/dev/null && break; kill -0 $AG 2>/dev/null || break; sleep 1; done
  if [ "$SCOPE" = outside ]; then
    CG="system.slice/docker-$(docker inspect -f '{{.Id}}' "$CID").scope"
    SCOPEARG=(-a -G "$CG")
  else
    # scope to the vLLM engine's login-session cgroup (system-wide -a filtered to the cgroup) so the 6
    # counters are shared ACROSS the engine's threads -> no multiplexing (perf -p multiplies counters
    # per-thread on the many-threaded engine and multiplexes; -a -G cgroup does not).
    ECPID=$(pgrep -f "VLLM::EngineCore" | head -1)
    CG=$(sed 's#^0::/##' "/proc/$ECPID/cgroup" 2>/dev/null)
    SCOPEARG=(-a -G "$CG")
  fi
  # cap each group's capture window at MAXSEC (bounds cost for long/hanging web tasks; also makes the
  # 4 group runs closer in duration -> more comparable). Kill the agent if it exceeds the cap.
  MAXSEC="${MAXSEC:-240}"
  sudo "$PERF" stat -e "$EV" "${SCOPEARG[@]}" -x, -o "$OUT/tool_${g}.csv" \
    -- bash -c "s=0; while kill -0 $AG 2>/dev/null && [ \$s -lt $MAXSEC ]; do sleep 2; s=\$((s+2)); done" >/dev/null 2>&1
  kill -9 $AG 2>/dev/null; wait $AG 2>/dev/null
  echo "  -> $OUT/tool_${g}.csv ($(wc -l < "$OUT/tool_${g}.csv" 2>/dev/null || echo 0) lines)"
  cleanup
done
echo "OC_CAPTURE_DONE $LABEL $SCOPE"
