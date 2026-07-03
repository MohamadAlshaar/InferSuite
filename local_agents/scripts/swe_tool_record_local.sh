#!/usr/bin/env bash
# SWE tool-exec SOFTWARE VIEW (local): replay the Sonnet-recorded trajectory (sweagent run-replay,
# no LLM) with perf record (task-clock) scoped to the sandbox container cgroup — same replay scope
# as the CANONICAL tool-exec TMA. DSO attribution needs no symfs; flat view resolved via a twin
# container of the same image. Usage: swe_tool_record_local.sh <name> <trajectory.traj>
set -uo pipefail
NAME="${1:?name}"; TRAJ="$(realpath "${2:?traj}")"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PERF=$(ls -d /usr/lib/linux-tools-6.8*/perf 2>/dev/null | tail -1)
OUT="$REPO/local_agents/data/swe_tool_$NAME"; mkdir -p "$OUT"
cd "$REPO/agentic/swe_agent"; source .venv/bin/activate
sudo pkill -9 -x perf 2>/dev/null; sleep 1
for c in $(docker ps -aq --filter "name=sweb" 2>/dev/null); do docker rm -f "$c" >/dev/null 2>&1; done

sweagent run-replay --traj_path "$TRAJ" > "$OUT/replay.log" 2>&1 &
AG=$!
CID=""
for i in $(seq 1 120); do
  CID=$(docker ps --format '{{.ID}}' --filter "name=sweb" | head -1)
  [ -n "$CID" ] && break
  kill -0 "$AG" 2>/dev/null || { echo "replay died before sandbox"; exit 1; }
  sleep 2
done
[ -n "$CID" ] || { echo "no sandbox"; exit 1; }
FULL=$(docker inspect -f '{{.Id}}' "$CID"); IMG=$(docker inspect -f '{{.Config.Image}}' "$CID")
CG="system.slice/docker-${FULL}.scope"
echo "[$NAME] sandbox=$CID img=$IMG"
# record until the sandbox disappears (= replay done); wait_gone is the perf workload
sudo "$PERF" record -e task-clock -F 199 -a -G "$CG" -g -o "$OUT/tool.data" -- \
  bash -c "while docker ps --format '{{.Names}}' | grep -qi sweb; do sleep 1; done" >/dev/null 2>&1
wait "$AG" 2>/dev/null
# twin container of the same image for symbol resolution
TWIN=$(docker run -d "$IMG" sleep 600); TPID=$(docker inspect -f '{{.State.Pid}}' "$TWIN")
sudo "$PERF" report -i "$OUT/tool.data" --stdio -g none --no-children --symfs="/proc/$TPID/root" 2>/dev/null | grep -E "^\s+[0-9]" > "$OUT/perf_flat.txt"
sudo "$PERF" report -i "$OUT/tool.data" --stdio --sort=dso -g none 2>/dev/null | grep -E "^\s+[0-9]" > "$OUT/perf_dso.txt"
docker rm -f "$TWIN" >/dev/null 2>&1
sz=$(stat -c%s "$OUT/tool.data" 2>/dev/null || echo 0)
echo "SWE_TOOLREC_DONE $NAME rec=${sz}B dso_lines=$(wc -l < "$OUT/perf_dso.txt") top=$(head -1 "$OUT/perf_dso.txt" | sed 's/^ *//' | cut -c1-60)"
