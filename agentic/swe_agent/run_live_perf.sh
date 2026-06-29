#!/usr/bin/env bash
# LIVE perf, per-instance. Runs the agent live (vLLM serving) on ONE instance at a time and attaches
# 5 probes so we can split CPU:
#   DURING inference  = vLLM engine: vllm_perf.csv (TMA) + vllm_cores.csv (cores sampler)
#   OUTSIDE inference = agent brain (agent_perf.csv, the sweagent host process)
#                       + tool-exec  (sandbox_perf.csv, the sandbox container cgroup)
#   GPU activity      = gpu_timeline.csv (inference proxy)
# One instance at a time (num_workers=1) => clean cgroup/PID attribution. Coarse TMA group only;
# the DETAILED un-multiplexed per-group microarch comes from the replay passes (run_replay_perf.sh).
set -uo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
. ../common/perf_events.sh; . ../common/lib_perf.sh
PERF="$(perf_bin)" || { echo "FATAL: no working perf"; exit 1; }
EVENTS="$(tma_group)"
MODEL="${MODEL:-qwen2.5-7b}"; VLLM_API_BASE="${VLLM_API_BASE:-http://localhost:8000/v1}"
export HOSTED_VLLM_API_BASE="$VLLM_API_BASE" HOSTED_VLLM_API_KEY="dummy" OPENAI_API_KEY="dummy"
# 7 instances that DID real work (3 immediate-submits excluded per direction)
INSTANCES="${INSTANCES:-django__django-10880 django__django-10973 django__django-10999 astropy__astropy-14096 matplotlib__matplotlib-24627 pydata__xarray-6744 scikit-learn__scikit-learn-25232}"
echo "[live] perf=$PERF | TMA=$(echo $EVENTS|cut -c1-30)... | instances=$(echo $INSTANCES|wc -w)"

for INST in $INSTANCES; do
  OUT="runs/live/$INST"; rm -rf "$OUT"; mkdir -p "$OUT"
  echo "===== LIVE $INST ====="
  sweagent run-batch \
    --config external/SWE-agent/config/sweagent_0_7/07_thought_action.yaml \
    --instances.type swe_bench --instances.subset verified --instances.split test \
    --instances.filter "$INST" \
    --agent.model.name "hosted_vllm/${MODEL}" --agent.model.api_base "$VLLM_API_BASE" --agent.model.api_key dummy \
    --agent.model.per_instance_cost_limit 0 --agent.model.total_cost_limit 0 \
    --agent.model.per_instance_call_limit 30 --agent.model.max_input_tokens 28000 --agent.model.max_output_tokens 4096 \
    --agent.model.temperature 0.2 --agent.model.completion_kwargs '{"frequency_penalty":0.2,"presence_penalty":0.1}' \
    --agent.tools.execution_timeout 300 --num_workers 1 --output_dir "$OUT" > "$OUT/agent.log" 2>&1 &
  AGENT=$!
  CID=""
  for i in $(seq 1 150); do
    CID=$(docker ps --format '{{.ID}} {{.Names}}' | grep -i sweb | awk '{print $1}' | head -1)
    [ -n "$CID" ] && break
    kill -0 "$AGENT" 2>/dev/null || { echo "  [warn] agent exited before sandbox"; break; }
    sleep 2
  done
  if [ -z "$CID" ]; then echo "  [warn] no sandbox; skipping probes"; wait "$AGENT" 2>/dev/null; continue; fi
  FULL=$(docker inspect -f '{{.Id}}' "$CID"); CG="system.slice/docker-${FULL}.scope"
  echo "$(date +%s.%N) perf_start cg=$CG cid=$CID" > "$OUT/markers.txt"
  # outside-inference: tool-exec (sandbox cgroup)
  "$PERF" stat -e "$EVENTS" -G "$CG" -a -I 1000 -x, -o "$OUT/sandbox_perf.csv" & P1=$!
  # GPU (inference proxy)
  ( while sleep 1; do printf "%s,%s\n" "$(date +%s.%N)" "$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits|head -1|tr -d ' ')"; done ) > "$OUT/gpu_timeline.csv" 2>/dev/null & P2=$!
  # during-inference: vLLM server cores
  /usr/bin/python3 vllm_cpu_sampler.py "$OUT/vllm_cores.csv" & P3=$!
  # during-inference: vLLM engine TMA
  VPIDS=$(pgrep -f "VLLM::EngineCore|EngineCore|vllm.*serve|vllm.entrypoints" | paste -sd, -)
  if [ -n "$VPIDS" ]; then echo "$(date +%s.%N) vllm_pids=$VPIDS" >> "$OUT/markers.txt"; "$PERF" stat -p "$VPIDS" -e "$EVENTS" -I 1000 -x, -o "$OUT/vllm_perf.csv" 2>/dev/null & P4=$!; else echo "  [warn] no vLLM pids"; P4=""; fi
  # outside-inference: agent brain (sweagent host process)
  APID=$(pgrep -f "sweagent run-batch" | head -1)
  if [ -n "$APID" ]; then "$PERF" stat -p "$APID" -e "$EVENTS" -I 1000 -x, -o "$OUT/agent_perf.csv" 2>/dev/null & P5=$!; else P5=""; fi
  wait "$AGENT"
  echo "$(date +%s.%N) agent_done" >> "$OUT/markers.txt"
  kill -INT "$P1" ${P4:+"$P4"} ${P5:+"$P5"} 2>/dev/null; sleep 1; kill "$P1" "$P2" "$P3" ${P4:+"$P4"} ${P5:+"$P5"} 2>/dev/null
  for c in $(docker ps -aq --filter "name=sweb" 2>/dev/null); do docker rm -f "$c" >/dev/null 2>&1; done
  # quick per-instance work check
  TJ=$(find "$OUT" -name '*.traj' | head -1)
  if [ -n "$TJ" ]; then
    python3 -c "import json;d=json.load(open('$TJ'))['trajectory'];print('  -> turns:',len(d),'| nonempty actions:',sum(1 for s in d if (s.get('action') or '').strip()))"
  fi
done
echo "ALL_LIVE_DONE"
