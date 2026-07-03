#!/usr/bin/env bash
# swe_live_two_view.sh — LOCAL self-served SWE-agent campaign: run sweagent LIVE (function-calling
# config, guided tool_choice) on astropy__astropy-14096 against the k3s engine (Coder-7B-AWQ,
# hermes parser, 32K ctx) and capture THREE cgroups in the same windows:
#   DURING  = engine pod cgroup            (vLLM serving CPU)
#   OUTSIDE = sweagent scope + sandbox container (agent loop + tool/command execution)
# Records: 3 parallel task-clock records. Stats: 7 groups x 12s, --for-each-cgroup all three,
# TMA first (same short-window rationale as the OpenClaw live capture).
set -o pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export KUBECONFIG="$HOME/.kube/k3s-local.yaml"
PERF=$(ls -d /usr/lib/linux-tools-6.8*/perf 2>/dev/null | tail -1)
OUT="$REPO/local_agents/data/swe_live"; mkdir -p "$OUT"; rm -f "$OUT"/*
INSTANCE="astropy__astropy-14096"
SANDBOX_IMG_SUB="astropy_1776_astropy-14096"
REC_SEC="${REC_SEC:-30}"
log(){ printf '[swe-live] %s\n' "$*"; }

cleanup(){
  pkill -f "sweagent run-batch" 2>/dev/null
  pkill -f "kubectl port-forward" 2>/dev/null
  docker ps -aq --filter "ancestor=docker.io/swebench/sweb.eval.x86_64.${SANDBOX_IMG_SUB}:latest" | xargs -r docker rm -f >/dev/null 2>&1
  kubectl scale deploy -n llm-d-local ms-local-llm-d-modelservice-decode --replicas=0 >/dev/null 2>&1
}
trap cleanup EXIT

sudo pkill -9 -x perf 2>/dev/null; sleep 2

# ---- engine up (Coder-7B, hermes, 32K) ----
kubectl scale deploy -n llm-d-local ms-local-llm-d-modelservice-decode --replicas=1
kubectl rollout status deploy -n llm-d-local ms-local-llm-d-modelservice-decode --timeout=600s || { log "ERROR rollout"; exit 1; }
pkill -f "kubectl port-forward" 2>/dev/null; sleep 1
kubectl port-forward --address 0.0.0.0 -n llm-d-local svc/ms-local-decode-direct 8000:8000 > /tmp/swe_pf.log 2>&1 &
for i in $(seq 1 60); do
  m=$(curl -s --max-time 3 "http://127.0.0.1:8000/v1/models" | grep -o "qwen2.5-coder-7b-instruct-awq" || true)
  [ -n "$m" ] && break; sleep 5
done
[ -n "${m:-}" ] || { log "ERROR: engine never served coder model"; exit 1; }
log "engine serving qwen2.5-coder-7b-instruct-awq on :8000"

pod=$(kubectl get pod -n llm-d-local -l llm-d.ai/role=decode --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')
cid=$(kubectl get pod "$pod" -n llm-d-local -o jsonpath='{.status.containerStatuses[0].containerID}'); cid=${cid##*://}
ENG_PID=$(sudo k3s crictl inspect --output go-template --template '{{.info.pid}}' "$cid")
ENG_CG=$(sudo cat /proc/$ENG_PID/cgroup | sed 's/^0:://'); ENG_CG=${ENG_CG#/}
log "engine pod=$pod pid=$ENG_PID"

# ---- sweagent live in its own scope ----
rm -rf "$REPO/agentic/swe_agent/runs/live_local_7b"   # run-batch skips instances with an existing trajectory
docker ps -aq --filter "ancestor=docker.io/swebench/sweb.eval.x86_64.${SANDBOX_IMG_SUB}:latest" | xargs -r docker rm -f >/dev/null 2>&1
systemd-run --user --scope --unit=swe-live --collect -- bash -c "
  cd '$REPO/agentic/swe_agent' && source .venv/bin/activate &&
  export HOSTED_VLLM_API_BASE=http://localhost:8000/v1 HOSTED_VLLM_API_KEY=dummy OPENAI_API_KEY=dummy &&
  sweagent run-batch \
    --config external/SWE-agent/config/fc_local.yaml \
    --instances.type swe_bench --instances.subset verified --instances.split test \
    --instances.filter '$INSTANCE' \
    --agent.model.name hosted_vllm/qwen2.5-coder-7b-instruct-awq \
    --agent.model.api_base http://localhost:8000/v1 --agent.model.api_key dummy \
    --agent.model.per_instance_cost_limit 0 --agent.model.total_cost_limit 0 \
    --agent.model.max_input_tokens 28000 --agent.model.max_output_tokens 4096 \
    --agent.model.temperature 0.4 \
    --agent.model.completion_kwargs '{\"tool_choice\":\"required\",\"frequency_penalty\":0.5,\"presence_penalty\":0.3}' \
    --agent.tools.execution_timeout 90 --agent.tools.max_consecutive_execution_timeouts 6 \
    --num_workers 1 --output_dir runs/live_local_7b" > "$OUT/agent.log" 2>&1 &
sleep 3
AG_PID=$(pgrep -f "sweagent run-batch" | head -1)
[ -n "$AG_PID" ] || { log "ERROR: sweagent did not start"; tail -8 "$OUT/agent.log"; exit 1; }
DRV_CG=$(cat /proc/$AG_PID/cgroup | sed 's/^0:://'); DRV_CG=${DRV_CG#/}
log "sweagent pid=$AG_PID scope=$DRV_CG"

# sandbox container appears once the env boots
SB=""; for i in $(seq 1 200); do
  SB=$(docker ps --format '{{.ID}} {{.Image}} {{.Names}}' | grep "$SANDBOX_IMG_SUB" | awk '{print $1}' | head -1)
  [ -n "$SB" ] && break; kill -0 $AG_PID 2>/dev/null || break; sleep 1
done
[ -n "$SB" ] || { log "ERROR: no sandbox container"; tail -8 "$OUT/agent.log"; exit 1; }
SB_FULL=$(docker inspect -f '{{.Id}}' "$SB"); SB_PID=$(docker inspect -f '{{.State.Pid}}' "$SB")
SB_CG="system.slice/docker-${SB_FULL}.scope"
log "sandbox=$SB"

# ---- WORK GUARD: engine generating for the loop ----
ok=0
for i in $(seq 1 60); do
  kill -0 $AG_PID 2>/dev/null || { log "ERROR: sweagent died early"; tail -8 "$OUT/agent.log"; exit 1; }
  r=$(kubectl logs -n llm-d-local deploy/ms-local-llm-d-modelservice-decode --tail=5 2>/dev/null \
      | grep -oE "Running: [0-9]+" | tail -1 | grep -oE "[0-9]+" || echo 0)
  [ "${r:-0}" -ge 1 ] && { ok=1; log "WORK VERIFIED (Running:$r)"; break; }
  sleep 4
done
[ "$ok" = 1 ] || { log "ERROR: engine never busy"; tail -8 "$OUT/agent.log"; exit 1; }

# ---- capture: 3 records + 7 stat groups, all in the same window ----
declare -A GRP
GRP[core]="task-clock,cycles,instructions,branches,branch-misses"
GRP[fp1]="cycles,instructions,fp_arith_inst_retired.scalar_single,fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.128b_packed_single,fp_arith_inst_retired.128b_packed_double"
GRP[fp2]="cycles,instructions,fp_arith_inst_retired.256b_packed_single,fp_arith_inst_retired.256b_packed_double,fp_arith_inst_retired.512b_packed_single,fp_arith_inst_retired.512b_packed_double"
GRP[cache]="cycles,instructions,mem_load_retired.l1_hit,mem_load_retired.l2_hit,mem_load_retired.l3_hit,mem_load_retired.l3_miss"
GRP[mlp]="cycles,instructions,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles,uops_executed.thread"
GRP[tma1]="slots,topdown-retiring,topdown-bad-spec,topdown-fe-bound,topdown-be-bound"
GRP[tma2]="slots,topdown-heavy-ops,topdown-br-mispredict,topdown-fetch-lat,topdown-mem-bound"
log "record x3 ${REC_SEC}s + stats 7 groups x 12s (parallel)"
( for g in tma1 core tma2 cache fp1 fp2 mlp; do
    a=1; kill -0 $AG_PID 2>/dev/null || a=0
    echo "$g agent_alive=$a" >> "$OUT/stat_groups_alive.txt"
    [ "$a" = 0 ] && log "WARN: agent finished before group $g"
    sudo "$PERF" stat -a -e "${GRP[$g]}" --for-each-cgroup="$ENG_CG,$DRV_CG,$SB_CG" -- sleep 12 2> "$OUT/group_${g}.txt"
  done ) & STATS=$!
sudo "$PERF" record -e task-clock -a --cgroup="$ENG_CG" -g -F 199 -o "$OUT/rec_engine.data" -- sleep "$REC_SEC" > "$OUT/rec_engine.err" 2>&1 &
P1=$!
sudo "$PERF" record -e task-clock -a --cgroup="$DRV_CG" -g -F 199 -o "$OUT/rec_driver.data" -- sleep "$REC_SEC" > "$OUT/rec_driver.err" 2>&1 &
P2=$!
sudo "$PERF" record -e task-clock -a --cgroup="$SB_CG" -g -F 199 -o "$OUT/rec_sandbox.data" -- sleep "$REC_SEC" > "$OUT/rec_sandbox.err" 2>&1
wait $P1 $P2 $STATS
kubectl logs -n llm-d-local deploy/ms-local-llm-d-modelservice-decode --tail=40 2>/dev/null \
  | grep -E "Running|Waiting" | tail -12 > "$OUT/vllm_status_tail.txt" || true
ALIVE=0; kill -0 $AG_PID 2>/dev/null && ALIVE=1

# ---- reports ----
sudo "$PERF" report -i "$OUT/rec_engine.data" --stdio -g none --symfs="/proc/$ENG_PID/root" 2>/dev/null | grep -E "^\s+[0-9]" > "$OUT/engine_flat.txt" || true
sudo "$PERF" report -i "$OUT/rec_engine.data" --stdio --sort=dso 2>/dev/null | grep -E "^\s+[0-9]" > "$OUT/engine_dso.txt" || true
sudo "$PERF" report -i "$OUT/rec_driver.data" --stdio -g none 2>/dev/null | grep -E "^\s+[0-9]" > "$OUT/driver_flat.txt" || true
sudo "$PERF" report -i "$OUT/rec_driver.data" --stdio --sort=dso 2>/dev/null | grep -E "^\s+[0-9]" > "$OUT/driver_dso.txt" || true
sudo "$PERF" report -i "$OUT/rec_sandbox.data" --stdio -g none --symfs="/proc/${SB_PID}/root" 2>/dev/null | grep -E "^\s+[0-9]" > "$OUT/tool_flat.txt" || true
sudo "$PERF" report -i "$OUT/rec_sandbox.data" --stdio --sort=dso 2>/dev/null | grep -E "^\s+[0-9]" > "$OUT/tool_dso.txt" || true
sudo chown -R "$USER:$USER" "$OUT" 2>/dev/null

# ---- bounded natural end ----
s=0; while kill -0 $AG_PID 2>/dev/null && [ $s -lt 600 ]; do sleep 10; s=$((s+10)); done
tail -6 "$OUT/agent.log" | grep -v "^$" | tail -3

# ---- validation ----
se=$(stat -c%s "$OUT/rec_engine.data" 2>/dev/null || echo 0)
sd=$(stat -c%s "$OUT/rec_driver.data" 2>/dev/null || echo 0)
sb=$(stat -c%s "$OUT/rec_sandbox.data" 2>/dev/null || echo 0)
le=$(wc -l < "$OUT/engine_flat.txt" 2>/dev/null || echo 0)
tma_ok=$(head -2 "$OUT/stat_groups_alive.txt" 2>/dev/null | grep -c "agent_alive=1")
if [ "$se" -gt 100000 ] && [ "$sd" -gt 50000 ] && [ "$le" -gt 5 ] && [ "$tma_ok" = 2 ]; then
  log "VALIDATE-OK swe_live (eng=${se}B/${le}sym, drv=${sd}B, sandbox=${sb}B, tma+core in-window, alive-post=$ALIVE)"
else
  log "VALIDATE-FAIL swe_live (eng=${se}B/${le}sym, drv=${sd}B, sandbox=${sb}B, in-window=$tma_ok/2)"
fi
log "SWE-LIVE-DONE"
