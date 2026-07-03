#!/usr/bin/env bash
# gpu_time_chain.sh — two patches to the live local campaign, in one chain:
#  (1) GPU-vs-CPU TIME: sample nvidia-smi (2 Hz) from work-guard to agent exit for all six live
#      loops -> gpu_timeline.csv per workload (basis for the wall-time split donuts).
#  (2) OC engine TMA L2: the original chain ran td2 in the 24-36 s window, after the short-lived
#      OC agents had exited (idle engine). Re-capture group_tma2 FIRST (in-window), plus a core
#      window as load proof (group_core2, the original group_core stays untouched).
# Engine phases: Coder-7B (SWE live + BCB) then Instruct-7B (OC x4).
set -o pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export KUBECONFIG="$HOME/.kube/k3s-local.yaml"
PERF=$(ls -d /usr/lib/linux-tools-6.8*/perf 2>/dev/null | tail -1)
DATA="$REPO/local_agents/data"
SCRATCH="${SCRATCH:-/tmp}"
log(){ printf '[gpu-time] %s\n' "$*"; }

cleanup(){
  pkill -f "sweagent run-batch" 2>/dev/null; pkill -f "agentic_bcb.py" 2>/dev/null
  pkill -f "eval/run_batch.py" 2>/dev/null; pkill -f "kubectl port-forward" 2>/dev/null
  docker ps -aq --filter ancestor=wildclawbench-ubuntu:v1.3 | xargs -r docker rm -f >/dev/null 2>&1
  kubectl scale deploy -n llm-d-local ms-local-llm-d-modelservice-decode --replicas=0 >/dev/null 2>&1
}
trap cleanup EXIT
sudo pkill -9 -x perf 2>/dev/null

engine_up(){  # $1 = expected served-model substring
  kubectl scale deploy -n llm-d-local ms-local-llm-d-modelservice-decode --replicas=1
  kubectl rollout status deploy -n llm-d-local ms-local-llm-d-modelservice-decode --timeout=600s || return 1
  pkill -f "kubectl port-forward" 2>/dev/null; sleep 1
  kubectl port-forward --address 0.0.0.0 -n llm-d-local svc/ms-local-decode-direct 8000:8000 > /tmp/gt_pf.log 2>&1 &
  for i in $(seq 1 60); do
    curl -s --max-time 3 "http://127.0.0.1:8000/v1/models" | grep -q "$1" && { log "engine serving $1"; return 0; }
    sleep 5
  done
  log "ERROR: engine never served $1"; return 1
}

swap_model(){  # $1 = hostPath, $2 = served name (patch while scaled to 0)
  kubectl scale deploy -n llm-d-local ms-local-llm-d-modelservice-decode --replicas=0 >/dev/null
  sleep 5
  local ARGS CUR
  ARGS=$(kubectl get deploy -n llm-d-local ms-local-llm-d-modelservice-decode -o jsonpath='{.spec.template.spec.containers[?(@.name=="vllm")].args[0]}')
  CUR=$(echo "$ARGS" | grep -oE "served-model-name [a-z0-9.-]+" | awk '{print $2}')
  python3 - "$ARGS" "$CUR" "$2" "$1" <<'PYEOF' > "$SCRATCH/gt_patch.json"
import json, sys
a, cur, new, path = sys.argv[1:5]
a = a.replace(cur, new)
print(json.dumps([
  {"op":"replace","path":"/spec/template/spec/containers/0/args/0","value":a},
  {"op":"replace","path":"/spec/template/spec/volumes/2","value":{"name":"model-storage","hostPath":{"path":path,"type":"Directory"}}},
]))
PYEOF
  kubectl patch deploy -n llm-d-local ms-local-llm-d-modelservice-decode --type=json --patch-file="$SCRATCH/gt_patch.json" >/dev/null
}

engine_busy(){  # wait until engine reports a running request; echoes epoch on success
  for i in $(seq 1 90); do
    r=$(kubectl logs -n llm-d-local deploy/ms-local-llm-d-modelservice-decode --tail=5 2>/dev/null \
        | grep -oE "Running: [0-9]+" | tail -1 | grep -oE "[0-9]+" || echo 0)
    [ "${r:-0}" -ge 1 ] && { date +%s.%N; return 0; }
    sleep 2
  done
  return 1
}

gpu_sample(){  # $1 = agent pid, $2 = out csv; 2 Hz until agent exits
  while kill -0 "$1" 2>/dev/null; do
    printf "%s,%s\n" "$(date +%s.%N)" "$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    sleep 0.5
  done >> "$2"
}

eng_cg(){
  local pod cid pid
  pod=$(kubectl get pod -n llm-d-local -l llm-d.ai/role=decode --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')
  cid=$(kubectl get pod "$pod" -n llm-d-local -o jsonpath='{.status.containerStatuses[0].containerID}'); cid=${cid##*://}
  pid=$(sudo k3s crictl inspect --output go-template --template '{{.info.pid}}' "$cid")
  sudo cat /proc/$pid/cgroup | sed 's/^0:://' | sed 's|^/||'
}

# ================= Phase A: Coder-7B (SWE live + BCB) =================
engine_up qwen2.5-coder || { log "trying model swap to coder"; swap_model /data/qwen-coder-model qwen2.5-coder-7b-instruct-awq; engine_up qwen2.5-coder || exit 1; }

# ---- SWE live: GPU timeline only ----
log "================ swe_live (gpu timeline) ================"
rm -rf "$REPO/agentic/swe_agent/runs/live_local_7b"
bash -c "cd '$REPO/agentic/swe_agent' && source .venv/bin/activate &&
  export HOSTED_VLLM_API_BASE=http://localhost:8000/v1 HOSTED_VLLM_API_KEY=dummy OPENAI_API_KEY=dummy &&
  sweagent run-batch --config external/SWE-agent/config/fc_local.yaml \
    --instances.type swe_bench --instances.subset verified --instances.split test \
    --instances.filter astropy__astropy-14096 \
    --agent.model.name hosted_vllm/qwen2.5-coder-7b-instruct-awq \
    --agent.model.api_base http://localhost:8000/v1 --agent.model.api_key dummy \
    --agent.model.per_instance_cost_limit 0 --agent.model.total_cost_limit 0 \
    --agent.model.max_input_tokens 28000 --agent.model.max_output_tokens 4096 \
    --agent.model.temperature 0.4 \
    --agent.model.completion_kwargs '{\"tool_choice\":\"required\",\"frequency_penalty\":0.5,\"presence_penalty\":0.3}' \
    --agent.tools.execution_timeout 90 --agent.tools.max_consecutive_execution_timeouts 6 \
    --num_workers 1 --output_dir runs/live_local_7b" > "$DATA/swe_live/agent_gpu.log" 2>&1 &
AG=$!
G=$(engine_busy) || { log "ERROR swe never busy"; kill $AG; exit 1; }
echo "guard,$G" > "$DATA/swe_live/gpu_timeline.csv"
log "WORK VERIFIED swe_live"
gpu_sample $AG "$DATA/swe_live/gpu_timeline.csv"
wait $AG; log "swe_live done ($(grep -c "" "$DATA/swe_live/gpu_timeline.csv") samples)"

# ---- BCB: GPU timeline only (short loop) ----
log "================ bcb_live (gpu timeline) ================"
rm -f /tmp/bcb_agentic_markers.txt
bash -c "cd '$REPO/agentic/bigcodebench' && VLLM='http://127.0.0.1:8000/v1' MODEL=qwen2.5-coder-7b-instruct-awq .venv/bin/python agentic_bcb.py 4 2" \
  > "$DATA/bcb_live/agent_gpu.log" 2>&1 &
AG=$!
G=$(engine_busy) || { log "ERROR bcb never busy"; kill $AG; exit 1; }
echo "guard,$G" > "$DATA/bcb_live/gpu_timeline.csv"
log "WORK VERIFIED bcb_live"
gpu_sample $AG "$DATA/bcb_live/gpu_timeline.csv"
wait $AG
cp /tmp/bcb_agentic_markers.txt "$DATA/bcb_live/markers_gpu.txt" 2>/dev/null
log "bcb_live done ($(grep -c "" "$DATA/bcb_live/gpu_timeline.csv") samples)"

# ================= Phase B: Instruct-7B (OC x4: gpu timeline + in-window td2) =================
swap_model /data/qwen-model qwen2.5-7b-instruct-awq
engine_up "qwen2.5-7b-instruct-awq" || exit 1
ECG=$(eng_cg); log "engine cgroup ok"
cd "$REPO/agentic/openclaw/external/WildClawBench"
cp my_api.json my_api.sonnet.keep 2>/dev/null; cp my_api.local7b.json my_api.json
. .venv/bin/activate

oc_task(){  # $1 task path, $2 label
  local OUTD="$DATA/oc_live_$2"
  log "================ oc $2 ================"
  docker ps -aq --filter ancestor=wildclawbench-ubuntu:v1.3 | xargs -r docker rm -f >/dev/null 2>&1
  python3 eval/run_batch.py --task "$1" --models-config my_api.json \
    --model my-openai-proxy/qwen2.5-7b-instruct-awq --parallel 1 </dev/null > "$OUTD/agent_gpu.log" 2>&1 &
  local AG=$!
  local G; G=$(engine_busy) || { log "WORK-FAIL $2"; kill -9 $AG 2>/dev/null; wait $AG 2>/dev/null; return; }
  echo "guard,$G" > "$OUTD/gpu_timeline.csv"
  log "WORK VERIFIED $2"
  gpu_sample $AG "$OUTD/gpu_timeline.csv" &
  local GS=$!
  sudo "$PERF" stat -a -e "slots,topdown-heavy-ops,topdown-br-mispredict,topdown-fetch-lat,topdown-mem-bound" \
    --for-each-cgroup="$ECG" -- sleep 12 2> "$OUTD/group_tma2.txt"
  local a=1; kill -0 $AG 2>/dev/null || a=0
  sudo "$PERF" stat -a -e "task-clock,cycles,instructions,branches,branch-misses" \
    --for-each-cgroup="$ECG" -- sleep 12 2> "$OUTD/group_core2.txt"
  log "$2 td2 done (agent_alive_after_td2=$a)"
  local s=0; while kill -0 $AG 2>/dev/null && [ $s -lt 420 ]; do sleep 5; s=$((s+5)); done
  kill -9 $AG 2>/dev/null; wait $AG 2>/dev/null; wait $GS 2>/dev/null
  docker ps -aq --filter ancestor=wildclawbench-ubuntu:v1.3 | xargs -r docker rm -f >/dev/null 2>&1
  sudo chown -R "$USER:$USER" "$OUTD" 2>/dev/null
  log "VALIDATE $2: td2_in_window=$a samples=$(grep -c "" "$OUTD/gpu_timeline.csv")"
}
T=tasks/01_Productivity_Flow
oc_task "$T/01_Productivity_Flow_task_6_calendar_scheduling.md"  calendar
oc_task "$T/01_Productivity_Flow_task_1_arxiv_digest.md"         web-digest
oc_task "$T/01_Productivity_Flow_task_10_pdf_digest.md"          pdf-digest
oc_task "tasks/05_Creative_Synthesis/05_Creative_Synthesis_task_10_social_poster_multi_crop.md" image-crop
mv my_api.sonnet.keep my_api.json 2>/dev/null
log "GPU-TIME-CHAIN-DONE"
