#!/usr/bin/env bash
# tma_pair_chain.sh — last two gaps in the 3x4 matrix:
#  (1) AGENT-side TMA L2 for SWE + OC: tma1 and td2 must sample the SAME phase to nest, so run
#      them back-to-back FIRST (8 s each) on the harness cgroups -> group_p_tma1/group_p_tma2.
#  (2) pdf-digest harness FP groups (agent died before them in the microarch chain).
# BCB is skipped (stationary loop already window-consistent).
set -o pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export KUBECONFIG="$HOME/.kube/k3s-local.yaml"
PERF=$(ls -d /usr/lib/linux-tools-6.8*/perf 2>/dev/null | tail -1)
DATA="$REPO/local_agents/data"
SCRATCH="${SCRATCH:-/tmp}"
log(){ printf '[tma-pair] %s\n' "$*"; }
cleanup(){
  pkill -f "sweagent run-batch" 2>/dev/null; pkill -f "eval/run_batch.py" 2>/dev/null
  pkill -f "kubectl port-forward" 2>/dev/null
  docker ps -aq --filter ancestor=wildclawbench-ubuntu:v1.3 | xargs -r docker rm -f >/dev/null 2>&1
  kubectl scale deploy -n llm-d-local ms-local-llm-d-modelservice-decode --replicas=0 >/dev/null 2>&1
}
trap cleanup EXIT
sudo pkill -9 -x perf 2>/dev/null

engine_up(){
  kubectl scale deploy -n llm-d-local ms-local-llm-d-modelservice-decode --replicas=1
  kubectl rollout status deploy -n llm-d-local ms-local-llm-d-modelservice-decode --timeout=600s || return 1
  pkill -f "kubectl port-forward" 2>/dev/null; sleep 1
  kubectl port-forward --address 0.0.0.0 -n llm-d-local svc/ms-local-decode-direct 8000:8000 > /tmp/tp_pf.log 2>&1 &
  for i in $(seq 1 60); do
    curl -s --max-time 3 "http://127.0.0.1:8000/v1/models" | grep -q "$1" && { log "engine serving $1"; return 0; }
    sleep 5
  done
  return 1
}
swap_model(){
  kubectl scale deploy -n llm-d-local ms-local-llm-d-modelservice-decode --replicas=0 >/dev/null; sleep 5
  local ARGS CUR
  ARGS=$(kubectl get deploy -n llm-d-local ms-local-llm-d-modelservice-decode -o jsonpath='{.spec.template.spec.containers[?(@.name=="vllm")].args[0]}')
  CUR=$(echo "$ARGS" | grep -oE "served-model-name [a-z0-9.-]+" | awk '{print $2}')
  python3 - "$ARGS" "$CUR" "$2" "$1" <<'PYEOF' > "$SCRATCH/tp_patch.json"
import json, sys
a, cur, new, path = sys.argv[1:5]
print(json.dumps([
  {"op":"replace","path":"/spec/template/spec/containers/0/args/0","value":a.replace(cur, new)},
  {"op":"replace","path":"/spec/template/spec/volumes/2","value":{"name":"model-storage","hostPath":{"path":path,"type":"Directory"}}},
]))
PYEOF
  kubectl patch deploy -n llm-d-local ms-local-llm-d-modelservice-decode --type=json --patch-file="$SCRATCH/tp_patch.json" >/dev/null
}
engine_busy(){
  for i in $(seq 1 90); do
    r=$(kubectl logs -n llm-d-local deploy/ms-local-llm-d-modelservice-decode --tail=5 2>/dev/null \
        | grep -oE "Running: [0-9]+" | tail -1 | grep -oE "[0-9]+" || echo 0)
    [ "${r:-0}" -ge 1 ] && return 0
    sleep 2
  done
  return 1
}
TMA1EV="slots,topdown-retiring,topdown-bad-spec,topdown-fe-bound,topdown-be-bound"
TD2EV="slots,topdown-heavy-ops,topdown-br-mispredict,topdown-fetch-lat,topdown-mem-bound"
FP1EV="cycles,instructions,fp_arith_inst_retired.scalar_single,fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.128b_packed_single,fp_arith_inst_retired.128b_packed_double"
FP2EV="cycles,instructions,fp_arith_inst_retired.256b_packed_single,fp_arith_inst_retired.256b_packed_double,fp_arith_inst_retired.512b_packed_single,fp_arith_inst_retired.512b_packed_double"

pair_capture(){  # $1 out dir, $2 cgroups, $3 agent pid, $4 extra_fp(0/1)
  local a1=1; kill -0 "$3" 2>/dev/null || a1=0
  sudo "$PERF" stat -a -e "$TMA1EV" --for-each-cgroup="$2" -- sleep 8 2> "$1/group_p_tma1.txt"
  local a2=1; kill -0 "$3" 2>/dev/null || a2=0
  sudo "$PERF" stat -a -e "$TD2EV" --for-each-cgroup="$2" -- sleep 8 2> "$1/group_p_tma2.txt"
  echo "pair tma1_alive=$a1 td2_alive=$a2" > "$1/pair_alive.txt"
  if [ "$4" = 1 ]; then
    local a3=1; kill -0 "$3" 2>/dev/null || a3=0
    sudo "$PERF" stat -a -e "$FP1EV" --for-each-cgroup="$2" -- sleep 8 2> "$1/group_h_fp1.txt"
    sudo "$PERF" stat -a -e "$FP2EV" --for-each-cgroup="$2" -- sleep 8 2> "$1/group_h_fp2.txt"
    echo "fp_alive=$a3" >> "$1/pair_alive.txt"
  fi
  log "pair done (tma1_alive=$a1 td2_alive=$a2)"
}

# ---- Phase A: Coder (SWE) ----
engine_up qwen2.5-coder || { swap_model /data/qwen-coder-model qwen2.5-coder-7b-instruct-awq; engine_up qwen2.5-coder || exit 1; }
log "================ swe_live ================"
rm -rf "$REPO/agentic/swe_agent/runs/live_local_7b"
systemd-run --user --scope --unit=swe-tp --collect -- bash -c "
  cd '$REPO/agentic/swe_agent' && source .venv/bin/activate &&
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
    --num_workers 1 --output_dir runs/live_local_7b" > "$DATA/swe_live/agent_tp.log" 2>&1 &
sleep 3
AG=$(pgrep -f "sweagent run-batch" | head -1)
[ -n "$AG" ] || { log "ERROR sweagent did not start"; exit 1; }
SCOPE=$(cat /proc/$AG/cgroup | sed 's/^0:://'); SCOPE=${SCOPE#/}
SB=""; for i in $(seq 1 200); do
  SB=$(docker ps --format '{{.ID}} {{.Image}} {{.Names}}' | grep "astropy_1776_astropy-14096" | awk '{print $1}' | head -1)
  [ -n "$SB" ] && break; kill -0 $AG 2>/dev/null || break; sleep 1
done
[ -n "$SB" ] || { log "ERROR: no sandbox"; exit 1; }
SBF=$(docker inspect -f '{{.Id}}' "$SB")
engine_busy || { log "WORK-FAIL swe"; exit 1; }
log "WORK VERIFIED swe_live"
pair_capture "$DATA/swe_live" "$SCOPE,system.slice/docker-${SBF}.scope" $AG 0
kill -9 $AG 2>/dev/null; wait $AG 2>/dev/null
sudo chown -R "$USER:$USER" "$DATA/swe_live" 2>/dev/null

# ---- Phase B: Instruct (OC x4) ----
swap_model /data/qwen-model qwen2.5-7b-instruct-awq
engine_up "qwen2.5-7b-instruct-awq" || exit 1
cd "$REPO/agentic/openclaw/external/WildClawBench"
cp my_api.json my_api.sonnet.keep 2>/dev/null; cp my_api.local7b.json my_api.json
. .venv/bin/activate
oc_task(){
  local OUTD="$DATA/oc_live_$2"
  log "================ oc $2 ================"
  docker ps -aq --filter ancestor=wildclawbench-ubuntu:v1.3 | xargs -r docker rm -f >/dev/null 2>&1
  python3 eval/run_batch.py --task "$1" --models-config my_api.json \
    --model my-openai-proxy/qwen2.5-7b-instruct-awq --parallel 1 </dev/null > "$OUTD/agent_tp.log" 2>&1 &
  local AG=$!
  local CID=""; for i in $(seq 1 150); do CID=$(docker ps -q --filter ancestor=wildclawbench-ubuntu:v1.3 | head -1); [ -n "$CID" ] && break; kill -0 $AG 2>/dev/null || break; sleep 2; done
  [ -n "$CID" ] || { log "NO_CONTAINER $2"; kill -9 $AG 2>/dev/null; wait $AG 2>/dev/null; return; }
  local FULL=$(docker inspect -f '{{.Id}}' "$CID")
  engine_busy || { log "WORK-FAIL $2"; kill -9 $AG 2>/dev/null; wait $AG 2>/dev/null; return; }
  log "WORK VERIFIED $2"
  pair_capture "$OUTD" "system.slice/docker-${FULL}.scope" $AG "$3"
  kill -9 $AG 2>/dev/null; wait $AG 2>/dev/null
  docker ps -aq --filter ancestor=wildclawbench-ubuntu:v1.3 | xargs -r docker rm -f >/dev/null 2>&1
  sudo chown -R "$USER:$USER" "$OUTD" 2>/dev/null
  log "VALIDATE $2: $(cat "$OUTD/pair_alive.txt" | tr '\n' ' ')"
}
T=tasks/01_Productivity_Flow
oc_task "$T/01_Productivity_Flow_task_6_calendar_scheduling.md"  calendar   0
oc_task "$T/01_Productivity_Flow_task_1_arxiv_digest.md"         web-digest 0
oc_task "$T/01_Productivity_Flow_task_10_pdf_digest.md"          pdf-digest 1
oc_task "tasks/05_Creative_Synthesis/05_Creative_Synthesis_task_10_social_poster_multi_crop.md" image-crop 0
mv my_api.sonnet.keep my_api.json 2>/dev/null
log "TMA-PAIR-DONE"
