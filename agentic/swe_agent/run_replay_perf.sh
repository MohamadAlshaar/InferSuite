#!/usr/bin/env bash
# Deterministic multi-replay deep-CPU profiling of the tool-exec phase.
# Replays the SAME recorded trajectory once per counter group (no LLM, no GPU),
# each replay measured with aggregate perf on the sandbox cgroup -> no multiplexing,
# directly comparable across groups (agentic analog of run_benchmark.sh pass1-4).
#
# Usage: run_replay_perf.sh <trajectory.traj>
set -uo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
# WORKING perf via shared lib (verifies it counts; bare 'perf' wrapper refuses on mismatched OEM kernels).
. ../common/perf_events.sh
. ../common/lib_perf.sh
PERF="$(perf_bin)" || { echo "FATAL: no working perf binary (set PERF_HOST_BIN)"; exit 1; }
TRAJ="${1:?usage: run_replay_perf.sh <trajectory.traj>}"
OUT=runs/replay; mkdir -p "$OUT"

# counter groups (<=8 GP counters each -> no multiplexing on SPR). cycles+instructions
# in every group for IPC + cross-check. Mirrors RAG pass2a / pass4 / pass1.
group_events() {
  case "$1" in
    tma)   echo "cycles,instructions,slots,topdown-retiring,topdown-fe-bound,topdown-bad-spec,topdown-be-bound";;
    td2)   echo "cycles,instructions,slots,topdown-retiring,topdown-fe-bound,topdown-bad-spec,topdown-be-bound,topdown-fetch-lat,topdown-heavy-ops,topdown-mem-bound,topdown-br-mispredict";;
    cache) echo "cycles,instructions,mem_load_retired.l1_hit,mem_load_retired.l2_hit,mem_load_retired.l3_hit,mem_load_retired.l3_miss";;
    fp)    echo "cycles,instructions,fp_arith_inst_retired.scalar_single,fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.128b_packed_single,fp_arith_inst_retired.128b_packed_double,fp_arith_inst_retired.256b_packed_single,fp_arith_inst_retired.256b_packed_double,fp_arith_inst_retired.512b_packed_single,fp_arith_inst_retired.512b_packed_double";;
    mlp)   echo "cycles,instructions,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles,uops_executed.thread";;
  esac
}

for g in ${PGROUPS-tma cache fp mlp}; do   # set PGROUPS= to skip these and run only imc+toplev
  echo "===== replay group: $g ====="
  sweagent run-replay --traj_path "$TRAJ" > "$OUT/replay_$g.log" 2>&1 &
  AG=$!
  CID=""
  for i in $(seq 1 120); do
    CID=$(docker ps --format '{{.ID}} {{.Names}}' | grep -i sweb | awk '{print $1}' | head -1)
    [[ -n "$CID" ]] && break
    kill -0 "$AG" 2>/dev/null || { echo "  replay exited before sandbox"; break; }
    sleep 2
  done
  if [[ -z "$CID" ]]; then echo "  no sandbox for $g, skipping"; wait "$AG" 2>/dev/null; continue; fi
  FULL=$(docker inspect -f '{{.Id}}' "$CID")
  CG="system.slice/docker-${FULL}.scope"
  EV=$(group_events "$g")
  echo "  sandbox=$CID  group=[$EV]"
  # aggregate (no -I) over the whole replay -> totals on SIGINT
  "$PERF" stat -e "$EV" -G "$CG" -a -o "$OUT/group_$g.txt" &
  PP=$!
  wait "$AG"
  kill -INT "$PP" 2>/dev/null; sleep 1; kill "$PP" 2>/dev/null
  echo "  -> $OUT/group_$g.txt"
  # clean the sandbox before the next group's fresh replay
  for c in $(docker ps -aq --filter "name=sweb" 2>/dev/null); do docker rm -f "$c" >/dev/null 2>&1; done
done

wait_sandbox() {  # $1 = agent pid; returns when a sweb container is up (or agent died)
  for i in $(seq 1 120); do
    docker ps --format '{{.Names}}' | grep -qi sweb && return 0
    kill -0 "$1" 2>/dev/null || return 1
    sleep 2
  done
}

wait_marker() {  # $1 = logfile, $2 = agent pid; returns when "Running agent" logged (reset done)
  for i in $(seq 1 600); do
    grep -q "Running agent" "$1" 2>/dev/null && return 0
    kill -0 "$2" 2>/dev/null || return 1
    sleep 0.5
  done
  return 1
}

# wait_gone: blocks until no sweb sandbox container exists -> used as a measurement
# tool's "workload" so the tool exits CLEANLY (and flushes its -o file) when replay ends.
wait_gone() { while docker ps --format '{{.Names}}' | grep -qi sweb; do sleep 1; done; }
export -f wait_gone

# ---- IMC: node-wide DRAM bytes (uncore can't be cgroup-scoped; replay=dominant load) ----
if [[ "${RUN_IMC:-1}" == 1 ]]; then
echo "===== replay group: imc (node-wide DRAM) ====="
sweagent run-replay --traj_path "$TRAJ" > "$OUT/replay_imc.log" 2>&1 & AG=$!
if wait_sandbox "$AG"; then
  "$PERF" stat -e "cycles,instructions,uncore_cha/unc_cha_imc_reads_count.normal/,uncore_cha/unc_cha_imc_writes_count.full/" -a -o "$OUT/group_imc.txt" & PP=$!
  wait "$AG"; kill -INT "$PP" 2>/dev/null; sleep 1; kill "$PP" 2>/dev/null
  echo "  -> $OUT/group_imc.txt"
else wait "$AG" 2>/dev/null; fi
for c in $(docker ps -aq --filter "name=sweb" 2>/dev/null); do docker rm -f "$c" >/dev/null 2>&1; done
fi

# ---- IMC FENCED: start counters only AFTER "Running agent" (container start + git
# repo-reset excluded) -> isolates steady-state command-exec DRAM from setup churn. ----
if [[ "${RUN_IMC_FENCED:-0}" == 1 ]]; then
echo "===== replay group: imc_fenced (node-wide DRAM, post-reset only) ====="
sweagent run-replay --traj_path "$TRAJ" > "$OUT/replay_imc_fenced.log" 2>&1 & AG=$!
if wait_marker "$OUT/replay_imc_fenced.log" "$AG"; then
  echo "  marker 'Running agent' seen -> starting counters (reset done)"
  "$PERF" stat -e "cycles,instructions,uncore_cha/unc_cha_imc_reads_count.normal/,uncore_cha/unc_cha_imc_writes_count.full/" -a -o "$OUT/group_imc_fenced.txt" & PP=$!
  wait "$AG"; kill -INT "$PP" 2>/dev/null; sleep 1; kill "$PP" 2>/dev/null
  echo "  -> $OUT/group_imc_fenced.txt"
else echo "  marker never appeared / agent died"; wait "$AG" 2>/dev/null; fi
for c in $(docker ps -aq --filter "name=sweb" 2>/dev/null); do docker rm -f "$c" >/dev/null 2>&1; done
fi

# ---- toplev -l2: deep TMA tree (system-wide; replay=dominant load, vLLM off) ----
# toplev only writes -o on clean exit, so its workload = wait_gone (returns when the
# replay's sandbox disappears). No SIGINT -> file is flushed.
if [[ "${RUN_TOPLEV:-1}" == 1 ]]; then
echo "===== replay group: toplev -l2 ====="
sweagent run-replay --traj_path "$TRAJ" > "$OUT/replay_toplev.log" 2>&1 & AG=$!
if wait_sandbox "$AG"; then
  PERF="$PERF" python3 external/pmu-tools/toplev.py -l2 --no-desc -a -o "$OUT/group_toplev.txt" -- bash -c wait_gone & TP=$!
  wait "$AG"; wait "$TP" 2>/dev/null
  echo "  -> $OUT/group_toplev.txt"
else wait "$AG" 2>/dev/null; fi
for c in $(docker ps -aq --filter "name=sweb" 2>/dev/null); do docker rm -f "$c" >/dev/null 2>&1; done
fi

echo "ALL GROUPS DONE"
