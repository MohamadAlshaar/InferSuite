#!/usr/bin/env bash
# Per-task SWE-agent OUTSIDE-inference (tool-exec) microarch: run the instance's pytest suite (swebench
# eval_script = FAIL_TO_PASS + PASS_TO_PASS) inside its SWE-bench Docker container under perf, scoped
# to the container cgroup, once per counter group (core/fp/mem/stall) -> PMC-safe, no multiplexing,
# directly comparable across groups. Requires sudo docker. Usage: swe_toolexec.sh <instance_id>
set -uo pipefail
INST="${1:?instance_id}"
IMGTAG=$(echo "$INST" | sed "s/__/_1776_/")
IMG="swebench/sweb.eval.x86_64.${IMGTAG}:latest"
EVAL="/home/ubuntu/swe/eval_${INST}.sh"
OUT="/home/ubuntu/swe/runs/tool_${INST}"; mkdir -p "$OUT"
sudo sysctl -w kernel.perf_event_paranoid=-1 >/dev/null 2>&1
sudo sysctl -w kernel.kptr_restrict=0    >/dev/null 2>&1
core="cycles,instructions,cache-references,cache-misses,branch-instructions,branch-misses"
fp="cycles,instructions,fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.128b_packed_double,fp_arith_inst_retired.256b_packed_double,fp_arith_inst_retired.512b_packed_double"
mem="cycles,instructions,mem_load_retired.l1_hit,mem_load_retired.l2_hit,mem_load_retired.l3_hit,mem_load_retired.l3_miss"
stall="cycles,instructions,cycle_activity.stalls_total,cycle_activity.stalls_l3_miss,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles"
for g in core fp mem stall; do
  EV=$(eval echo "\$$g")
  echo "== tool $INST $g =="
  CID=$(sudo docker run -d "$IMG" sleep 7200)
  sudo docker cp "$EVAL" "$CID:/run_eval.sh"
  FULL=$(sudo docker inspect -f '{{.Id}}' "$CID")
  CG="system.slice/docker-${FULL}.scope"
  # perf on the container cgroup while it runs the pytest suite (aggregate, no -I -> totals)
  sudo perf stat -e "$EV" -G "$CG" -a -x, -o "$OUT/tool_${g}.csv" -- \
    sudo docker exec "$CID" bash -lc "bash /run_eval.sh" > "$OUT/eval_${g}.log" 2>&1 || true
  sudo docker rm -f "$CID" >/dev/null 2>&1
  echo "  -> $OUT/tool_${g}.csv ($(wc -l < "$OUT/tool_${g}.csv" 2>/dev/null || echo 0) lines)"
done
echo "TOOL_DONE $INST"
