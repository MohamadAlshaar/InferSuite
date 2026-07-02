#!/usr/bin/env bash
# SWE OUTSIDE tool-exec function attribution, per task: REPLAY the instance's pytest eval inside its
# SWE-bench Docker image under HOST perf record (task-clock, cgroup-scoped), then resolve the
# container's DSOs via --symfs=/proc/<container-init-pid>/root while the container is still alive.
# Deterministic replay (no model). Usage: swe_tool_record.sh <instance_id>
set -uo pipefail
INST="${1:?instance_id}"
IMGTAG=$(echo "$INST" | sed "s/__/_1776_/")
IMG="swebench/sweb.eval.x86_64.${IMGTAG}:latest"
EVAL="/home/ubuntu/swe/eval_${INST}.sh"
OUT="/home/ubuntu/swe/runs/toolrec_${INST}"; mkdir -p "$OUT"; rm -f "$OUT"/*
sudo sysctl -w kernel.perf_event_paranoid=-1 >/dev/null 2>&1
sudo pkill -9 -x perf 2>/dev/null; sleep 1
# fresh container, kept alive so /proc/<pid>/root stays mounted for symbol resolution
CID=$(sudo docker run -d "$IMG" sleep 7200)
sudo docker cp "$EVAL" "$CID:/run_eval.sh"
FULL=$(sudo docker inspect -f '{{.Id}}' "$CID")
INITPID=$(sudo docker inspect -f '{{.State.Pid}}' "$CID")
CG="system.slice/docker-${FULL}.scope"
echo "== $INST cgroup=$CG initpid=$INITPID =="
# record the in-container pytest (captured via -a -G cgroup) for the duration of the exec
sudo perf record -e task-clock -F 199 -a -G "$CG" -o "$OUT/tool.data" -- \
  sudo docker exec "$CID" bash -lc "bash /run_eval.sh" > "$OUT/eval.log" 2>&1 || true
# flat report; resolve container DSOs via the live container rootfs
sudo perf report -i "$OUT/tool.data" --symfs="/proc/${INITPID}/root" -g none --no-children --stdio 2>/dev/null > "$OUT/perf_flat.txt"
# also a DSO-level view (no symbols needed) as a fallback attribution
sudo perf report -i "$OUT/tool.data" --sort=dso -g none --stdio 2>/dev/null > "$OUT/perf_dso.txt"
sudo docker rm -f "$CID" >/dev/null 2>&1
echo "SWE_TOOLREC_DONE $INST flat_samples=$(grep -cE '^[[:space:]]*[0-9]' "$OUT/perf_flat.txt") dso_lines=$(grep -cE '%' "$OUT/perf_dso.txt")"
