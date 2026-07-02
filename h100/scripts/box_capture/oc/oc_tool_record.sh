#!/usr/bin/env bash
# OpenClaw OUTSIDE tool-exec DSO/function attribution, per task: LIVE agent run (Instruct-32B; browser
# agent can't be replayed) under HOST perf record, cgroup-scoped to the task container (agent+tools).
# DSO-level view (from mmap records -> robust, no container fs needed) + best-effort symbol view.
# Usage: TASK=<relpath> LABEL=<name> bash oc_tool_record.sh
set -uo pipefail
cd /home/ubuntu/oc/WildClawBench; . .venv/bin/activate
TASK="${TASK:?task}"; LABEL="${LABEL:?label}"
IMG=wildclawbench-ubuntu:v1.3
OUT="/home/ubuntu/oc/runs/toolrec_${LABEL}"; mkdir -p "$OUT"; rm -f "$OUT"/*
sudo sysctl -w kernel.perf_event_paranoid=-1 >/dev/null 2>&1
sudo pkill -9 -x perf 2>/dev/null; sleep 1
docker ps -aq --filter ancestor=$IMG | xargs -r docker rm -f >/dev/null 2>&1
python3 eval/run_batch.py --task "$TASK" --models-config my_api.json \
  --model my-openai-proxy/instruct-32b --parallel 1 </dev/null > "$OUT/agent.log" 2>&1 & AG=$!
CID=""; for i in $(seq 1 150); do CID=$(docker ps -q --filter ancestor=$IMG|head -1); [ -n "$CID" ] && break; kill -0 $AG 2>/dev/null || break; sleep 2; done
if [ -z "$CID" ]; then echo "NO_CONTAINER $LABEL"; wait $AG 2>/dev/null; exit 0; fi
for i in $(seq 1 300); do grep -q "Waiting for agent to finish" "$OUT/agent.log" 2>/dev/null && break; kill -0 $AG 2>/dev/null || break; sleep 1; done
FULL=$(docker inspect -f '{{.Id}}' "$CID"); INITPID=$(docker inspect -f '{{.State.Pid}}' "$CID")
CG="system.slice/docker-${FULL}.scope"
echo "== $LABEL cgroup=$CG initpid=$INITPID =="
sudo perf record -e task-clock -F 199 -a -G "$CG" -o "$OUT/tool.data" -- \
  bash -c "s=0; while kill -0 $AG 2>/dev/null && [ \$s -lt 220 ]; do sleep 2; s=\$((s+2)); done" >/dev/null 2>&1
# DSO view first (robust even after container exits — DSO names come from recorded mmaps)
sudo perf report -i "$OUT/tool.data" --sort=dso -g none --stdio 2>/dev/null > "$OUT/perf_dso.txt"
# best-effort symbols while container may still be alive
sudo perf report -i "$OUT/tool.data" --symfs="/proc/${INITPID}/root" -g none --no-children --stdio 2>/dev/null > "$OUT/perf_flat.txt" || true
kill -9 $AG 2>/dev/null; wait $AG 2>/dev/null
docker ps -aq --filter ancestor=$IMG | xargs -r docker rm -f >/dev/null 2>&1
echo "OC_TOOLREC_DONE $LABEL dso_lines=$(grep -cE '%' "$OUT/perf_dso.txt")"
