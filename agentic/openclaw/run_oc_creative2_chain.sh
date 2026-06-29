#!/usr/bin/env bash
# Unattended chain (NO VIDEO): wait for the calendar multi-pass to finish, then prep + run the
# Creative Synthesis task_3 product_poster (image composition; self-contained, briefcase.png) with
# clean per-group passes. Same 6-pass methodology as calendar.
set -uo pipefail
cd "$(dirname "$0")"
ROOT="external/WildClawBench"
T3="tasks/05_Creative_Synthesis/05_Creative_Synthesis_task_3_product_poster.md"

echo "[chain2] waiting for calendar passes to finish..."
for i in $(seq 1 600); do grep -q "ALLDONE" /tmp/oc_passes_status 2>/dev/null && break; sleep 10; done
echo "[chain2] calendar done -> archiving its passes to runs/passes_calendar"
rm -rf runs/passes_calendar; cp -r runs/passes runs/passes_calendar 2>/dev/null

echo "[chain2] downloading task_3 workspace (briefcase.png + gt) from HF ..."
( cd "$ROOT" && source .venv/bin/activate && \
  hf download internlm/WildClawBench --repo-type dataset \
    --include "workspace/05_Creative_Synthesis/task_3_product_poster/*" --local-dir . ) 2>&1 | tail -2
IN="$ROOT/workspace/05_Creative_Synthesis/task_3_product_poster/exec"
echo "[chain2] input present: $(ls "$IN" 2>/dev/null | tr '\n' ' ')"
ls "$IN"/*.png >/dev/null 2>&1 || { echo "[chain2] FATAL: no input image"; exit 1; }

echo "[chain2] running creative task_3 clean per-group passes ..."
rm -f runs/passes/group_*.txt runs/passes/freq_*
MODEL=claude-sonnet-4-6 REPEATS=1 bash run_all_passes.sh "$T3"
rm -rf runs/passes_creative; cp -r runs/passes runs/passes_creative 2>/dev/null
echo "[chain2] ALL DONE -> runs/passes_calendar + runs/passes_creative"
