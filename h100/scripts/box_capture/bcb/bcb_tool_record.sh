#!/usr/bin/env bash
# BCB OUTSIDE tool-exec function attribution: replay ALL recorded programs under ONE perf record
# (task-clock, on-host -> native symbols), flat report -> perf_flat.txt.
set -uo pipefail
CORPUS=/home/ubuntu/bcb/corpus
PY=/home/ubuntu/bcb/.venv/bin/python
OUT=/home/ubuntu/bcb/runs/tool_record; mkdir -p "$OUT"; rm -f "$OUT"/*
sudo sysctl -w kernel.perf_event_paranoid=-1 >/dev/null 2>&1
sudo pkill -9 -x perf 2>/dev/null; sleep 1
cd /tmp
perf record -e task-clock -F 199 -o "$OUT/bcb_tool.data" -- \
  bash -c "export MPLBACKEND=Agg; cd /tmp; for p in $CORPUS/prog_*.py; do timeout 40 \"$PY\" \"\$p\" >/dev/null 2>&1; done" >/dev/null 2>&1
perf report -i "$OUT/bcb_tool.data" -g none --no-children --stdio 2>/dev/null > "$OUT/perf_flat.txt"
echo "BCB_TOOL_RECORD_DONE samples=$(grep -c "%" "$OUT/perf_flat.txt") lines=$(wc -l < "$OUT/perf_flat.txt")"
