#!/usr/bin/env bash
# BCB DURING-inference CLEAN attribution: replay recorded requests to the engine, perf record the
# engine login-session cgroup (-a -G; NO py-spy, NO concurrent stat) -> uncontaminated attribution,
# same method as SWE/OpenClaw during records. Coder-32B (BCB provenance).
set -uo pipefail
cd /home/ubuntu/bcb
REQ=/home/ubuntu/bcb/corpus/requests.jsonl
OUT=/home/ubuntu/bcb/runs/during_record_clean; mkdir -p "$OUT"; rm -f "$OUT"/*
sudo sysctl -w kernel.perf_event_paranoid=-1 >/dev/null 2>&1
sudo pkill -9 -x perf 2>/dev/null; sleep 1
ECPID=$(pgrep -f "VLLM::EngineCore"|head -1); CG=$(sed "s#^0::/##" /proc/$ECPID/cgroup)
echo "engine cgroup=$CG  ecpid=$ECPID"
MODEL=coder-32b VLLM=http://localhost:8000/v1 python3 replay_engine.py "$REQ" >"$OUT/replay.log" 2>&1 & AG=$!
sleep 3
sudo perf record -e task-clock -F 199 -a -G "$CG" -o "$OUT/engine.data" -- \
  bash -c "while kill -0 $AG 2>/dev/null; do sleep 1; done" >/dev/null 2>&1
wait $AG 2>/dev/null
sudo perf report -i "$OUT/engine.data" -g none --no-children --stdio 2>/dev/null > "$OUT/perf_flat.txt"
echo "BCB_DURING_RECORD_CLEAN_DONE samples=$(grep -cE "^[[:space:]]+[0-9]" "$OUT/perf_flat.txt")"
