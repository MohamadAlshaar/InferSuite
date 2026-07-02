#!/usr/bin/env bash
# OpenClaw DURING-inference engine perf RECORD (function attribution). Run a task; while the agent
# loop drives the engine, perf-record the engine login-session cgroup with task-clock call-graph;
# emit perf_flat.txt (flat, no children) -> role attribution (matches BCB/SWE during-record).
set -uo pipefail
cd /home/ubuntu/oc/WildClawBench; . .venv/bin/activate
TASK="tasks/01_Productivity_Flow/01_Productivity_Flow_task_6_calendar_scheduling.md"
IMG=wildclawbench-ubuntu:v1.3
OUT=/home/ubuntu/oc/runs/oc_during_record; mkdir -p "$OUT"
sudo sysctl -w kernel.perf_event_paranoid=-1 >/dev/null 2>&1
docker ps -aq --filter ancestor=$IMG | xargs -r docker rm -f >/dev/null 2>&1
python3 eval/run_batch.py --task "$TASK" --models-config my_api.json \
  --model my-openai-proxy/instruct-32b --parallel 1 </dev/null > "$OUT/agent.log" 2>&1 & AG=$!
CID=""; for i in $(seq 1 150); do CID=$(docker ps -q --filter ancestor=$IMG|head -1); [ -n "$CID" ] && break; kill -0 $AG 2>/dev/null || break; sleep 2; done
for i in $(seq 1 300); do grep -q "Waiting for agent to finish" "$OUT/agent.log" 2>/dev/null && break; kill -0 $AG 2>/dev/null || break; sleep 1; done
ECPID=$(pgrep -f "VLLM::EngineCore"|head -1); CG=$(sed 's#^0::/##' "/proc/$ECPID/cgroup")
echo "recording engine cgroup=$CG"
sudo perf record -e task-clock -F 99 -g --call-graph dwarf -a -G "$CG" -o "$OUT/engine.data" \
  -- bash -c "s=0; while kill -0 $AG 2>/dev/null && [ \$s -lt 220 ]; do sleep 2; s=\$((s+2)); done" >/dev/null 2>&1
kill -9 $AG 2>/dev/null; wait $AG 2>/dev/null
docker ps -aq --filter ancestor=$IMG | xargs -r docker rm -f >/dev/null 2>&1
sudo perf report -i "$OUT/engine.data" -g none --no-children --stdio 2>/dev/null > "$OUT/perf_flat.txt"
echo "OC_RECORD_DONE lines=$(wc -l < "$OUT/perf_flat.txt")"
