#!/usr/bin/env bash
# OpenClaw tool-exec SOFTWARE VIEW (local, LIVE, Sonnet-driven): browser agents cannot be replayed,
# so run the task live (model = claude-sonnet-4-6 via the litellm proxy) with perf record
# (task-clock) scoped to the task container (agent + all tools). Adapted from
# h100/scripts/box_capture/oc/oc_tool_record.sh. Usage: TASK=<relpath> LABEL=<name> oc_tool_record_local.sh
set -uo pipefail
TASK="${TASK:?task}"; LABEL="${LABEL:?label}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PERF=$(ls -d /usr/lib/linux-tools-6.8*/perf 2>/dev/null | tail -1)
IMG=wildclawbench-ubuntu:v1.3
OUT="$REPO/local_agents/data/oc_tool_${LABEL}"; mkdir -p "$OUT"; rm -f "$OUT"/*
cd "$REPO/agentic/openclaw/external/WildClawBench"; . .venv/bin/activate
sudo pkill -9 -x perf 2>/dev/null; sleep 1
docker ps -aq --filter ancestor=$IMG | xargs -r docker rm -f >/dev/null 2>&1
python3 eval/run_batch.py --task "$TASK" --models-config my_api.json \
  --model my-openai-proxy/claude-sonnet-4-6 --parallel 1 </dev/null > "$OUT/agent.log" 2>&1 & AG=$!
CID=""; for i in $(seq 1 150); do CID=$(docker ps -q --filter ancestor=$IMG | head -1); [ -n "$CID" ] && break; kill -0 $AG 2>/dev/null || break; sleep 2; done
if [ -z "$CID" ]; then echo "NO_CONTAINER $LABEL"; kill $AG 2>/dev/null; wait $AG 2>/dev/null; exit 0; fi
for i in $(seq 1 300); do grep -q "Waiting for agent to finish" "$OUT/agent.log" 2>/dev/null && break; kill -0 $AG 2>/dev/null || break; sleep 1; done
FULL=$(docker inspect -f '{{.Id}}' "$CID"); INITPID=$(docker inspect -f '{{.State.Pid}}' "$CID")
CG="system.slice/docker-${FULL}.scope"
echo "== $LABEL cgroup ok, initpid=$INITPID =="
sudo "$PERF" record -e task-clock -F 199 -a -G "$CG" -g -o "$OUT/tool.data" -- \
  bash -c "s=0; while kill -0 $AG 2>/dev/null && [ \$s -lt 220 ]; do sleep 2; s=\$((s+2)); done" > "$OUT/perf_record.err" 2>&1
sudo "$PERF" report -i "$OUT/tool.data" --stdio --sort=dso -g none --no-children 2>/dev/null | grep -E "^\s+[0-9]" > "$OUT/perf_dso.txt"
sudo "$PERF" report -i "$OUT/tool.data" --stdio --symfs="/proc/${INITPID}/root" -g none --no-children 2>/dev/null | grep -E "^\s+[0-9]" > "$OUT/perf_flat.txt" || true
kill -9 $AG 2>/dev/null; wait $AG 2>/dev/null
docker ps -aq --filter ancestor=$IMG | xargs -r docker rm -f >/dev/null 2>&1
sz=$(stat -c%s "$OUT/tool.data" 2>/dev/null || echo 0)
echo "OC_TOOLREC_DONE $LABEL rec=${sz}B dso_lines=$(wc -l < "$OUT/perf_dso.txt") top=$(head -1 "$OUT/perf_dso.txt" | sed 's/^ *//' | cut -c1-55)"
