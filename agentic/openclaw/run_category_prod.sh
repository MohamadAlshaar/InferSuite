#!/usr/bin/env bash
# Full Productivity_Flow category: run all 10 tasks CONSECUTIVELY, each with one (multiplexed)
# perf pass, archiving per-task timeline + score. Aggregated over 10 tasks -> solid category microarch.
set -uo pipefail
cd "$(dirname "$0")"
ROOT="external/WildClawBench"
OUTBASE="runs/perf_cat_prod"; mkdir -p "$OUTBASE"
STAT=/tmp/oc_cat_status; : > "$STAT"

# wait for the HF data download to finish (started separately)
echo "$(date +%T) waiting for HF data..." >> "$STAT"
for i in $(seq 1 180); do pgrep -f "hf download" >/dev/null || break; sleep 5; done

cleanup(){ docker ps -aq --filter ancestor=wildclawbench-ubuntu:v1.3 2>/dev/null | xargs -r docker rm -f >/dev/null 2>&1; }

for f in $(ls "$ROOT"/tasks/01_Productivity_Flow/01_Productivity_Flow_task_*.md | sort -V); do
  base=$(basename "$f"); name=$(echo "$base" | sed 's/01_Productivity_Flow_//; s/\.md$//')
  rel="tasks/01_Productivity_Flow/$base"
  echo "$(date +%T) START $name" >> "$STAT"
  cleanup
  rm -f runs/perf/container_timeline.csv runs/perf/markers.txt
  bash run_perf.sh "$rel" >/dev/null 2>&1 || echo "$(date +%T) WARN run_perf rc=$? $name" >> "$STAT"
  d="$OUTBASE/$name"; mkdir -p "$d"
  cp runs/perf/container_timeline.csv runs/perf/markers.txt "$d/" 2>/dev/null
  rd=$(find "$ROOT/output/openclaw/01_Productivity_Flow/01_Productivity_Flow_$name" -type d -name 'claude-sonnet-4-6_*' 2>/dev/null | sort | tail -1)
  cp "$rd/score.json" "$d/score.json" 2>/dev/null
  cp "$rd/agent.log" "$d/agent.log" 2>/dev/null
  sc=$(python3 -c "import json;print(json.load(open('$d/score.json')).get('overall_score','NA'))" 2>/dev/null || echo NA)
  echo "$(date +%T) DONE $name score=$sc" >> "$STAT"
done
cleanup
echo "$(date +%T) ALLDONE" >> "$STAT"
