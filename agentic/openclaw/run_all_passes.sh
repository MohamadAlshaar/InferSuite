#!/usr/bin/env bash
# OpenClaw PARITY driver: live-repeated AGGREGATE passes (one perf event-group per agent run,
# repeated for the non-determinism), cgroup-scoped on the task container -> matches the
# BigCodeBench/SWE-agent 6-pass methodology via the shared lib. Replaces the old multiplexed
# -I single-instance run_perf.sh.
#
# Usage: REPEATS=2 run_all_passes.sh <task.md (relative to external/WildClawBench)>
set -uo pipefail
cd "$(dirname "$0")"
source ../common/perf_events.sh
source ../common/lib_perf.sh
PERF=$(perf_bin) || exit 1
SUDO=$(perf_sudo); perf_enable
ROOT="external/WildClawBench"
TASK="${1:?usage: run_all_passes.sh <task.md relative to external/WildClawBench>}"
REPEATS="${REPEATS:-2}"
OUT="runs/passes"; mkdir -p "$OUT"; rm -f "$OUT"/group_*.txt "$OUT"/freq_*
ST=/tmp/oc_passes_status; : > "$ST"

launch_agent() {   # $1=logfile  -> sets global AG
  ( cd "$ROOT" && . .venv/bin/activate && \
    python3 eval/run_batch.py --task "$TASK" --models-config my_api.json \
      --model "my-openai-proxy/${MODEL:-qwen2.5-32b}" --parallel 1 </dev/null ) > "$1" 2>&1 &
  AG=$!
}
find_cgroup() {    # waits for container, fences past warmup; echoes cgroup or empty
  local log="$1" cid="" i
  for i in $(seq 1 150); do
    cid=$(docker ps -q --filter ancestor=wildclawbench-ubuntu:v1.3 | head -1)
    [ -n "$cid" ] && break; kill -0 "$AG" 2>/dev/null || return 1; sleep 2
  done
  [ -z "$cid" ] && return 1
  for i in $(seq 1 240); do            # WARMUP FENCE (exclude pip/chromium install)
    grep -q "Waiting for agent to finish" "$log" 2>/dev/null && break
    kill -0 "$AG" 2>/dev/null || return 1; sleep 1
  done
  echo "system.slice/docker-$(docker inspect -f '{{.Id}}' "$cid").scope"
}
cleanup() { docker ps -aq --filter ancestor=wildclawbench-ubuntu:v1.3 2>/dev/null | xargs -r docker rm -f >/dev/null 2>&1; }

run_pass() {       # $1=name $2=events $3=repeat ; $4=scope (cgroup|node)
  local g="$1" ev="$2" r="$3" mode="${4:-cgroup}"
  local log="/tmp/oc_${g}_r${r}.log"
  echo "$(date +%T) $g r$r START" >> "$ST"; cleanup
  launch_agent "$log"
  local CG; CG=$(find_cgroup "$log") || { echo "$(date +%T) $g r$r NO_CONTAINER/WARMUP" >> "$ST"; wait "$AG" 2>/dev/null; return; }
  local scope="-a -G $CG"; [ "$mode" = node ] && scope="-a"
  $SUDO "$PERF" stat -e "$ev" $scope -o "$OUT/group_${g}_r${r}.txt" </dev/null & local PP=$!
  ( while kill -0 "$AG" 2>/dev/null; do eff_freq_hz; sleep 2; done ) > "$OUT/freq_${g}_${r}" 2>/dev/null &
  wait "$AG"; $SUDO kill -INT "$PP" 2>/dev/null
  for _ in $(seq 1 20); do [ -s "$OUT/group_${g}_r${r}.txt" ] && break; sleep 0.5; done
  if assert_perf_ok "$OUT/group_${g}_r${r}.txt" "$g.r$r" 2>>"$ST"; then echo "$(date +%T) $g r$r OK" >> "$ST"
  else echo "$(date +%T) $g r$r ASSERT_FAIL" >> "$ST"; fi
  cleanup
}

# PASS_GROUPS selects which passes to run (default all) -> re-run only the wrong/missing ones cheaply.
# (NB: do NOT name this GROUPS — that's a bash builtin array of the user's gids.)
PASS_GROUPS="${PASS_GROUPS:-TMA TD2 CACHE FP MLP IMC}"
for r in $(seq 1 "$REPEATS"); do
  for g in $PASS_GROUPS; do
    case "$g" in
      TMA)   run_pass TMA   "$PG_TMA"   "$r" cgroup;;
      TD2)   run_pass TD2   "$PG_TD2"   "$r" cgroup;;   # clean L2 drill-down (replaces toplev)
      CACHE) run_pass CACHE "$PG_CACHE" "$r" cgroup;;
      FP)    run_pass FP    "$PG_FP"    "$r" cgroup;;
      MLP)   run_pass MLP   "$PG_MLP"   "$r" cgroup;;
      IMC)   run_pass IMC   "$PG_IMC"   "$r" node;;     # uncore node-wide (DRAM use cgroup L3-miss)
    esac
  done
done

# toplev -l2 pass — superseded by TD2; only run if explicitly requested (RUN_TOPLEV=1)
if [ "${RUN_TOPLEV:-0}" = 1 ]; then
echo "$(date +%T) toplev START" >> "$ST"; cleanup
launch_agent /tmp/oc_toplev.log
CG=$(find_cgroup /tmp/oc_toplev.log) || true
if [ -n "${CG:-}" ]; then
  $SUDO python3 ../swe_agent/external/pmu-tools/toplev.py -l2 --no-desc -a -o "$OUT/group_toplev.txt" \
    -- bash -c "while kill -0 $AG 2>/dev/null; do sleep 1; done" 2>/tmp/oc_toplev.err
fi
wait "$AG" 2>/dev/null
[ -s "$OUT/group_toplev.txt" ] && echo "$(date +%T) toplev OK" >> "$ST" || echo "$(date +%T) toplev EMPTY" >> "$ST"
cleanup
fi

FREQ=$(cat "$OUT"/freq_* 2>/dev/null | awk '{s+=$1;n++} END{if(n)printf "%d",s/n; else print 0}')
echo "FREQ=$FREQ" > "$OUT/freq.txt"
echo "$(date +%T) ALLDONE freq=$FREQ" >> "$ST"
