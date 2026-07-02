#!/usr/bin/env bash
# Validate a capture run BEFORE trusting it: (1) agent completed tasks/loops,
# (2) markers form phase pairs, (3) counters non-zero & NOT multiplexed, (4) attribution non-empty.
# perf stat -x, interval CSV columns: time,value,unit,event,run-time,pct-enabled,...
OUT="${1:?run dir}"
echo "========== VALIDATE $OUT =========="
echo "--- (1) AGENT COMPLETION ---"
grep -iE "solved|unsolved|tool-exec runs|===|chat-err|Traceback|Connection|refused" "$OUT/agent.log" 2>/dev/null | tail -8
echo "--- (2) MARKERS (expect RUN_START=1 RUN_END=1, toolexec_start==toolexec_end>0) ---"
if [ -f "$OUT/markers.txt" ]; then
  printf "RUN_START=%s RUN_END=%s toolexec_start=%s toolexec_end=%s\n" \
    "$(grep -c RUN_START "$OUT/markers.txt")" "$(grep -c RUN_END "$OUT/markers.txt")" \
    "$(grep -c toolexec_start "$OUT/markers.txt")" "$(grep -c toolexec_end "$OUT/markers.txt")"
else echo "  !! NO markers.txt"; fi
echo "--- (3) COUNTER INTEGRITY (zeroed cycles rows or pct<99 = BAD) ---"
for f in engine_timeline.csv engine_fp_timeline.csv sys_timeline.csv; do
  [ -f "$OUT/$f" ] || continue
  echo "  [$f] rows=$(wc -l < "$OUT/$f")"
  grep -iE "not counted|not supported|<not" "$OUT/$f" >/dev/null && echo "    !! <not counted/supported> PRESENT"
  awk -F, '
    $4=="cycles"{c++; if($2+0==0) zc++}
    $4=="instructions"{if($2+0==0) zi++}
    NF>=6 && $6 ~ /^[0-9.]+$/ && $6+0<99 {mux++}
    END{printf "    cycles rows=%d zeroed=%d ; instr zeroed=%d ; rows_with_multiplexing(pct<99)=%d\n", c, zc+0, zi+0, mux+0}' "$OUT/$f"
done
echo "--- (4) NATIVE ATTRIBUTION (perf record) top symbols ---"
if [ -s "$OUT/engine_perf.data" ]; then
  perf report -i "$OUT/engine_perf.data" --stdio 2>/dev/null | grep -vE "^#|^$" | head -12
else echo "  !! NO/empty engine_perf.data"; fi
echo "--- (5) PYTHON ATTRIBUTION (py-spy folded) top stacks ---"
if [ -s "$OUT/engine_pyspy.folded" ]; then
  echo "  folded lines=$(wc -l < "$OUT/engine_pyspy.folded")"
  awk '{n=$NF; $NF=""; print n"\t"$0}' "$OUT/engine_pyspy.folded" | sort -rn | head -4
else echo "  !! NO/empty engine_pyspy.folded (check pyspy.err)"; fi
echo "--- (6) CORE-SECONDS (vLLM task-clock total) ---"
[ -f "$OUT/engine_timeline.csv" ] && awk -F, '$4=="task-clock"{s+=$2} END{printf "  vLLM = %.1f ms = %.2f core-sec\n", s, s/1000}' "$OUT/engine_timeline.csv"
[ -f "$OUT/sys_timeline.csv" ]    && awk -F, '$4=="task-clock"{s+=$2} END{printf "  system = %.1f ms = %.2f core-sec\n", s, s/1000}' "$OUT/sys_timeline.csv"
echo "===================================="
