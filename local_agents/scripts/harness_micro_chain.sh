#!/usr/bin/env bash
# harness_micro_chain.sh — fill the verified gap in the local live campaign: the AGENT-HARNESS
# microarch groups (cache/mlp/fp1/fp2) for SWE and the four OC tasks. Earlier chains ran these
# groups late in the sequence and the short-lived agents were already dead (files show
# "no access to cgroup"). Here they run FIRST, 8 s each (32 s total), scoped to the harness
# cgroups only, with per-group aliveness flags. Output: group_h_<grp>.txt in each existing dir.
# BCB is skipped (its 20-min loop already produced valid in-window harness groups).
set -o pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export KUBECONFIG="$HOME/.kube/k3s-local.yaml"
PERF=$(ls -d /usr/lib/linux-tools-6.8*/perf 2>/dev/null | tail -1)
DATA="$REPO/local_agents/data"
SCRATCH="${SCRATCH:-/tmp}"
log(){ printf '[harness-micro] %s\n' "$*"; }

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
  kubectl port-forward --address 0.0.0.0 -n llm-d-local svc/ms-local-decode-direct 8000:8000 > /tmp/hm_pf.log 2>&1 &
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
  python3 - "$ARGS" "$CUR" "$2" "$1" <<'PYEOF' > "$SCRATCH/hm_patch.json"
import json, sys
a, cur, new, path = sys.argv[1:5]
print(json.dumps([
  {"op":"replace","path":"/spec/template/spec/containers/0/args/0","value":a.replace(cur, new)},
  {"op":"replace","path":"/spec/template/spec/volumes/2","value":{"name":"model-storage","hostPath":{"path":path,"type":"Directory"}}},
]))
PYEOF
  kubectl patch deploy -n llm-d-local ms-local-llm-d-modelservice-decode --type=json --patch-file="$SCRATCH/hm_patch.json" >/dev/null
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

declare -A GRP
GRP[cache]="cycles,instructions,mem_load_retired.l1_hit,mem_load_retired.l2_hit,mem_load_retired.l3_hit,mem_load_retired.l3_miss"
GRP[mlp]="cycles,instructions,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles,uops_executed.thread"
GRP[fp1]="cycles,instructions,fp_arith_inst_retired.scalar_single,fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.128b_packed_single,fp_arith_inst_retired.128b_packed_double"
GRP[fp2]="cycles,instructions,fp_arith_inst_retired.256b_packed_single,fp_arith_inst_retired.256b_packed_double,fp_arith_inst_retired.512b_packed_single,fp_arith_inst_retired.512b_packed_double"

run_groups(){  # $1 out dir, $2 cgroup list (comma), $3 agent pid
  for g in cache mlp fp1 fp2; do
    local a=1; kill -0 "$3" 2>/dev/null || a=0
    echo "$g agent_alive=$a" >> "$1/harness_groups_alive.txt"
    sudo "$PERF" stat -a -e "${GRP[$g]}" --for-each-cgroup="$2" -- sleep 8 2> "$1/group_h_${g}.txt"
  done
}

# ================= Phase A: Instruct (OC x4) =================
engine_up "qwen2.5-7b-instruct-awq" || { swap_model /data/qwen-model qwen2.5-7b-instruct-awq; engine_up "qwen2.5-7b-instruct-awq" || exit 1; }
cd "$REPO/agentic/openclaw/external/WildClawBench"
cp my_api.json my_api.sonnet.keep 2>/dev/null; cp my_api.local7b.json my_api.json
. .venv/bin/activate
oc_task(){
  local OUTD="$DATA/oc_live_$2"; rm -f "$OUTD/harness_groups_alive.txt"
  log "================ oc $2 ================"
  docker ps -aq --filter ancestor=wildclawbench-ubuntu:v1.3 | xargs -r docker rm -f >/dev/null 2>&1
  python3 eval/run_batch.py --task "$1" --models-config my_api.json \
    --model my-openai-proxy/qwen2.5-7b-instruct-awq --parallel 1 </dev/null > "$OUTD/agent_hm.log" 2>&1 &
  local AG=$!
  local CID=""; for i in $(seq 1 150); do CID=$(docker ps -q --filter ancestor=wildclawbench-ubuntu:v1.3 | head -1); [ -n "$CID" ] && break; kill -0 $AG 2>/dev/null || break; sleep 2; done
  [ -n "$CID" ] || { log "NO_CONTAINER $2"; kill -9 $AG 2>/dev/null; wait $AG 2>/dev/null; return; }
  local FULL=$(docker inspect -f '{{.Id}}' "$CID")
  engine_busy || { log "WORK-FAIL $2"; kill -9 $AG 2>/dev/null; wait $AG 2>/dev/null; return; }
  log "WORK VERIFIED $2"
  run_groups "$OUTD" "system.slice/docker-${FULL}.scope" $AG
  local ok=$(grep -c "agent_alive=1" "$OUTD/harness_groups_alive.txt")
  local s=0; while kill -0 $AG 2>/dev/null && [ $s -lt 420 ]; do sleep 5; s=$((s+5)); done
  kill -9 $AG 2>/dev/null; wait $AG 2>/dev/null
  docker ps -aq --filter ancestor=wildclawbench-ubuntu:v1.3 | xargs -r docker rm -f >/dev/null 2>&1
  sudo chown -R "$USER:$USER" "$OUTD" 2>/dev/null
  log "VALIDATE $2: harness groups in-window $ok/4"
}
T=tasks/01_Productivity_Flow
oc_task "$T/01_Productivity_Flow_task_6_calendar_scheduling.md"  calendar
oc_task "$T/01_Productivity_Flow_task_1_arxiv_digest.md"         web-digest
oc_task "$T/01_Productivity_Flow_task_10_pdf_digest.md"          pdf-digest
oc_task "tasks/05_Creative_Synthesis/05_Creative_Synthesis_task_10_social_poster_multi_crop.md" image-crop
mv my_api.sonnet.keep my_api.json 2>/dev/null

# ================= Phase B: Coder (SWE live) =================
swap_model /data/qwen-coder-model qwen2.5-coder-7b-instruct-awq
engine_up qwen2.5-coder || exit 1
log "================ swe_live ================"
OUTD="$DATA/swe_live"; rm -f "$OUTD/harness_groups_alive.txt"
rm -rf "$REPO/agentic/swe_agent/runs/live_local_7b"
systemd-run --user --scope --unit=swe-hm --collect -- bash -c "
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
    --num_workers 1 --output_dir runs/live_local_7b" > "$OUTD/agent_hm.log" 2>&1 &
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
run_groups "$OUTD" "$SCOPE,system.slice/docker-${SBF}.scope" $AG
ok=$(grep -c "agent_alive=1" "$OUTD/harness_groups_alive.txt")
kill -9 $AG 2>/dev/null; wait $AG 2>/dev/null
sudo chown -R "$USER:$USER" "$OUTD" 2>/dev/null
log "VALIDATE swe_live: harness groups in-window $ok/4"
log "HARNESS-MICRO-DONE"
