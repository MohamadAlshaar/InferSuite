#!/usr/bin/env bash
# REPLAY a saved program corpus under ONE perf group -> per-program CSVs. The SAME programs are run
# under each group, so the tool-exec microarch is directly comparable across core/fp/mem/stall
# (deterministic; not stochastic per-group live re-generation).
# Usage: replay_tool.sh <corpus_dir> <perf_events> <out_dir>
set -uo pipefail
CORPUS="${1:?corpus dir}"; EVENTS="${2:?perf events}"; OUT="${3:?out dir}"
PY=/home/ubuntu/bcb/.venv/bin/python
sudo sysctl -w kernel.perf_event_paranoid=-1 >/dev/null 2>&1
mkdir -p "$OUT"; rm -f "$OUT"/*.csv
n=0
for p in "$CORPUS"/prog_*.py; do
  [ -e "$p" ] || continue
  perf stat -x, -o "$OUT/$(basename "$p").csv" -e "$EVENTS" -- timeout 40 "$PY" "$p" >/dev/null 2>&1 || true
  n=$((n+1))
done
echo "replayed $n programs under [$(echo "$EVENTS" | cut -d, -f3)...] -> $OUT"
