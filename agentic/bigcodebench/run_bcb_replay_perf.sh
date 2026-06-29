#!/usr/bin/env bash
# Phase 2: clean tool-exec microarch for the agentic BCB run. Re-runs the recorded executions
# (final-per-task) once per counter group under a separate perf pass (no multiplexing). Box must be
# idle (vLLM stopped) so system-wide -a captures only the replay. Outputs runs/replay_perf/group_*.txt
set -uo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
. ../common/perf_events.sh; . ../common/lib_perf.sh
PERF="$(perf_bin)" || { echo "FATAL: no working perf"; exit 1; }
EXEC="${1:-runs/agentic_claude/executed.jsonl}"
OUT=runs/replay_perf; rm -rf "$OUT"; mkdir -p "$OUT"
declare -A G=(
 [tma]="$(tma_group)"
 [cache]="cycles,instructions,mem_load_retired.l1_hit,mem_load_retired.l2_hit,mem_load_retired.l3_hit,mem_load_retired.l3_miss"
 [fp]="cycles,instructions,fp_arith_inst_retired.scalar_single,fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.128b_packed_single,fp_arith_inst_retired.128b_packed_double,fp_arith_inst_retired.256b_packed_single,fp_arith_inst_retired.256b_packed_double,fp_arith_inst_retired.512b_packed_single,fp_arith_inst_retired.512b_packed_double"
 [mlp]="cycles,instructions,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles,uops_executed.thread"
 [imc]="cycles,instructions,uncore_cha/unc_cha_imc_reads_count.normal/,uncore_cha/unc_cha_imc_writes_count.full/"
)
echo "[bcb-replay] perf=$PERF | exec=$EXEC | $(wc -l < "$EXEC") recorded runs"
for g in tma cache fp mlp imc; do
  echo "===== group $g ====="
  "$PERF" stat -e "${G[$g]}" -a -o "$OUT/group_$g.txt" -- python3 replay_executions.py "$EXEC" last 2>>"$OUT/replay_$g.log"
  tail -1 "$OUT/replay_$g.log"
done
echo "ALL_GROUPS_DONE -> $OUT"
