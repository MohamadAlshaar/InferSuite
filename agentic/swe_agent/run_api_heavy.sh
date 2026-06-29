#!/usr/bin/env bash
# FULL hosted-Claude run over the 5 HEAVY (numeric) SWE-bench Verified instances.
# Captures EVERYTHING measurable in the hosted setup:
#   - per-step execution_time + model_stats (loops, api_calls, cost, tokens) -> from trajectory
#   - total wall-clock per instance -> markers.txt  => time IN-agent (inference) vs OUTSIDE (tool-exec)
#   - tool-exec CPU (sandbox cgroup, TMA interval) -> sandbox_perf.csv (also = active/idle timeline)
#   - agent-orchestration CPU (sweagent host process) -> agent_perf.csv
# Stock default.yaml (function-calling, the right config for a capable model) + top_p null
# (Claude 4.x rejects temperature+top_p together). Key from .env (gitignored, never echoed).
# Light instances intentionally SKIPPED to save cost. One instance at a time -> clean attribution.
# Detailed un-multiplexed microarch comes from the replay pass afterwards (run_replay_perf.sh).
set -uo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
set -a; . ./.env; set +a
[ -n "${ANTHROPIC_API_KEY:-}" ] || { echo "FATAL: ANTHROPIC_API_KEY not set"; exit 1; }
. ../common/perf_events.sh; . ../common/lib_perf.sh
PERF="$(perf_bin)" || { echo "FATAL: no working perf"; exit 1; }
EVENTS="$(tma_group)"
MODEL="${MODEL:-anthropic/claude-sonnet-4-6}"
INSTANCES="${INSTANCES:-scikit-learn__scikit-learn-25232 astropy__astropy-14096 sympy__sympy-14248 matplotlib__matplotlib-24627 pydata__xarray-6744}"
echo "[api-heavy] model=$MODEL | perf=$PERF | instances=$(echo $INSTANCES|wc -w) (heavy only)"

for INST in $INSTANCES; do
  OUT="runs/api/$INST"; rm -rf "$OUT"; mkdir -p "$OUT"
  echo "===== RUN $INST ====="
  date +%s.%N > "$OUT/wall_start.txt"
  sweagent run-batch \
    --config external/SWE-agent/config/default.yaml \
    --instances.type swe_bench --instances.subset verified --instances.split test \
    --instances.filter "$INST" \
    --agent.model.name "$MODEL" \
    --agent.model.top_p null \
    --agent.model.per_instance_cost_limit 4.0 \
    --agent.model.total_cost_limit 30.0 \
    --num_workers 1 \
    --output_dir "$OUT" > "$OUT/agent.log" 2>&1 &
  AGENT=$!
  # attach sandbox-cgroup perf (tool-exec CPU + active/idle timeline) once the container is up
  CID=""
  for i in $(seq 1 180); do
    CID=$(docker ps --format '{{.ID}} {{.Names}}' | grep -i sweb | awk '{print $1}' | head -1)
    [ -n "$CID" ] && break
    kill -0 "$AGENT" 2>/dev/null || { echo "  [warn] agent exited before sandbox"; break; }
    sleep 2
  done
  P1=""; P5=""
  if [ -n "$CID" ]; then
    FULL=$(docker inspect -f '{{.Id}}' "$CID"); CG="system.slice/docker-${FULL}.scope"
    echo "$(date +%s.%N) perf_start cg=$CG" > "$OUT/markers.txt"
    "$PERF" stat -e "$EVENTS" -G "$CG" -a -I 1000 -x, -o "$OUT/sandbox_perf.csv" & P1=$!
    APID=$(pgrep -f "sweagent run-batch" | head -1)
    [ -n "$APID" ] && { "$PERF" stat -p "$APID" -e "$EVENTS" -I 1000 -x, -o "$OUT/agent_perf.csv" 2>/dev/null & P5=$!; }
  else
    echo "  [warn] no sandbox container; perf skipped"
  fi
  wait "$AGENT"
  date +%s.%N > "$OUT/wall_end.txt"
  echo "$(date +%s.%N) agent_done" >> "$OUT/markers.txt" 2>/dev/null || true
  kill -INT ${P1:+"$P1"} ${P5:+"$P5"} 2>/dev/null; sleep 1; kill ${P1:+"$P1"} ${P5:+"$P5"} 2>/dev/null
  for c in $(docker ps -aq --filter "name=sweb" 2>/dev/null); do docker rm -f "$c" >/dev/null 2>&1; done
  # per-instance quick summary
  TJ=$(find "$OUT" -name "$INST.traj" | head -1)
  [ -f "$TJ" ] && python3 -c "
import json
d=json.load(open('$TJ')); tr=d['trajectory']; ms=d['info'].get('model_stats',{})
texec=sum(s.get('execution_time',0) or 0 for s in tr)
print(f'  -> steps(loops)={len(tr)} api_calls={ms.get(\"api_calls\")} cost=\${ms.get(\"instance_cost\",0):.2f} exit={d[\"info\"].get(\"exit_status\")} tool_exec_s={texec:.1f}')
"
done
echo "ALL_API_RUNS_DONE"
