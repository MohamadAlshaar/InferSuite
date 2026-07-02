#!/usr/bin/env bash
# Capture vLLM during-inference ORCHESTRATION while an agent drives it, on the H100 VM.
# PMC-SAFE: the VM's vPMU exposes ~6 GP counters and SILENTLY ZEROS events beyond that
# (no multiplex tag). So counting events are kept within budget and attribution uses
# task-clock sampling (a SOFTWARE event -> zero PMCs -> no contention with the counters).
#
#   MODE=core -> IPC/cache/branch timeline + native attribution (perf record task-clock)
#                + python attribution (py-spy) + core-seconds (task-clock)
#   MODE=fp   -> FP timeline only (separate pass; FP GP events would oversubscribe with core)
#
# Usage: capture_orchestration.sh <label> <core|fp> <agent_cmd...>
set -uo pipefail
LABEL="${1:?label}"; MODE="${2:?mode: core|fp}"; shift 2
AGENT_CMD="$*"
OUT="$HOME/bcb/runs/$LABEL"; mkdir -p "$OUT"
MARKERS=/tmp/bcb_agentic_markers.txt; rm -f "$MARKERS"
sudo sysctl -w kernel.perf_event_paranoid=-1 >/dev/null 2>&1
sudo sysctl -w kernel.kptr_restrict=0    >/dev/null 2>&1

ENG=$(pgrep -f "EngineCore" | head -1)
ALL=$(pgrep -f "vllm|EngineCore" | sort -u | paste -sd, -)
[ -z "$ENG" ] && { echo "[cap] ERROR: no EngineCore pid (is vllm serving?)"; exit 3; }
echo "EngineCore=$ENG ; ALL_vllm=$ALL" | tee "$OUT/pids.txt"

declare -a PIDS=()
if [ "$MODE" = "core" ]; then
  # 4 GP + 2 fixed + task-clock(SW). -x, keeps the %-enabled column so we can prove no multiplexing.
  perf stat -p "$ALL" \
    -e cycles,instructions,cache-references,cache-misses,branch-instructions,branch-misses,task-clock \
    -I 1000 -x, -o "$OUT/engine_timeline.csv" 2>"$OUT/perfstat.err" & PIDS+=($!)
  # system-wide task-clock (SW, PMC-free) -> total & tool-exec core-seconds via markers
  perf stat -a -e task-clock -I 1000 -x, -o "$OUT/sys_timeline.csv" 2>/dev/null & PIDS+=($!)
  # native function attribution via SOFTWARE task-clock sampling (no PMC -> no contention). No py-spy.
  perf record -e task-clock -F 99 -g --call-graph dwarf -o "$OUT/engine_perf.data" -p "$ALL" 2>"$OUT/perfrec.err" & PIDS+=($!)
elif [ "$MODE" = "fp" ]; then
  perf stat -p "$ALL" \
    -e cycles,instructions,fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.128b_packed_double,fp_arith_inst_retired.256b_packed_double,fp_arith_inst_retired.512b_packed_double \
    -I 1000 -x, -o "$OUT/engine_fp_timeline.csv" 2>"$OUT/perfstat_fp.err" & PIDS+=($!)
elif [ "$MODE" = "mem" ]; then
  # cache hierarchy (L1/L2/L3 hit) + bandwidth proxy (l3_miss x 64B): 4 GP + 2 fixed
  perf stat -p "$ALL" \
    -e cycles,instructions,mem_load_retired.l1_hit,mem_load_retired.l2_hit,mem_load_retired.l3_hit,mem_load_retired.l3_miss \
    -I 1000 -x, -o "$OUT/engine_mem_timeline.csv" 2>"$OUT/perfstat_mem.err" & PIDS+=($!)
elif [ "$MODE" = "stall" ]; then
  # memory-bound % (TMA-lite) + MLP: 4 GP + 2 fixed
  perf stat -p "$ALL" \
    -e cycles,instructions,cycle_activity.stalls_total,cycle_activity.stalls_l3_miss,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles \
    -I 1000 -x, -o "$OUT/engine_stall_timeline.csv" 2>"$OUT/perfstat_stall.err" & PIDS+=($!)
else
  echo "[cap] bad MODE=$MODE"; exit 2
fi
sleep 1
echo "[cap] captures up (${PIDS[*]}); running agent: $AGENT_CMD"
eval "$AGENT_CMD" > "$OUT/agent.log" 2>&1; AG_RC=$?
echo "[cap] agent rc=$AG_RC; stopping captures"
for p in "${PIDS[@]}"; do kill -INT "$p" 2>/dev/null; done
sleep 4
for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done
sleep 1
cp "$MARKERS" "$OUT/markers.txt" 2>/dev/null
echo "AGENT_RC=$AG_RC" > "$OUT/status.txt"
# flat native attribution (fast: -g none, no callgraph tree) for the figure
if [ "$MODE" = "core" ] && [ -s "$OUT/engine_perf.data" ]; then
  perf report -i "$OUT/engine_perf.data" --stdio -g none --no-children 2>/dev/null \
    | grep -vE "^#|^$" | head -120 > "$OUT/perf_flat.txt" || true
fi
echo "[cap] done -> $OUT"
