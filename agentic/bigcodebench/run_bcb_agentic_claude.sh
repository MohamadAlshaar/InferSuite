#!/usr/bin/env bash
# Phase 1: agentic BCB-Hard with Claude, to completion. Records solved/loops + per-turn exec_times
# + markers + executed.jsonl. NO perf here (time split = markers+exec_times; microarch = replay phase).
set -uo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
set -a; . ./.env; set +a; export ANTHROPIC_KEY="$ANTHROPIC_API_KEY"
[ -n "${ANTHROPIC_API_KEY:-}" ] || { echo "FATAL: no key"; exit 1; }
export MODEL="${MODEL:-claude-sonnet-4-6}" OUTDIR="${OUTDIR:-runs/agentic_claude}"
rm -rf "$OUTDIR" /tmp/bcb_agentic_markers.txt; mkdir -p "$OUTDIR"
echo "$(date +%s.%N) WALL_START" > "$OUTDIR/wall.txt"
python3 agentic_bcb_claude.py "${1:-all}" "${2:-4}"
echo "$(date +%s.%N) WALL_END" >> "$OUTDIR/wall.txt"
cp /tmp/bcb_agentic_markers.txt "$OUTDIR/markers.txt" 2>/dev/null
echo "[bcb-agentic] done -> $OUTDIR"
