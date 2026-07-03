#!/usr/bin/env bash
# BCB tool-exec SOFTWARE VIEW (local): replay ALL programs recorded from the Sonnet-driven run
# (agentic_claude/executed.jsonl) natively in the bigcodebench venv under ONE perf record
# (task-clock). Deterministic, no model, native symbols. -> local_agents/data/bcb_tool/
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PERF=$(ls -d /usr/lib/linux-tools-6.8*/perf 2>/dev/null | tail -1)
PY="$REPO/agentic/bigcodebench/.venv/bin/python"
JSONL="$REPO/agentic/bigcodebench/runs/agentic_claude/executed.jsonl"
OUT="$REPO/local_agents/data/bcb_tool"; mkdir -p "$OUT"
sudo pkill -9 -x perf 2>/dev/null; sleep 1
cd "$REPO/agentic/bigcodebench"
sudo "$PERF" record -e task-clock -F 199 -g -o "$OUT/tool.data" -- \
  bash -c "export MPLBACKEND=Agg; \"$PY\" replay_executions.py \"$JSONL\" all" > "$OUT/replay.log" 2>&1 || true
sudo "$PERF" report -i "$OUT/tool.data" --stdio -g none --no-children 2>/dev/null | grep -E "^\s+[0-9]" > "$OUT/perf_flat.txt"
sudo "$PERF" report -i "$OUT/tool.data" --stdio --sort=dso -g none 2>/dev/null | grep -E "^\s+[0-9]" > "$OUT/perf_dso.txt"
sz=$(stat -c%s "$OUT/tool.data" 2>/dev/null || echo 0)
echo "BCB_TOOL_RECORD_DONE rec=${sz}B dso_lines=$(wc -l < "$OUT/perf_dso.txt") top=$(head -1 "$OUT/perf_dso.txt" | sed 's/^ *//')"
