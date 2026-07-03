#!/usr/bin/env bash
# oc_live_two_view.sh — LOCAL self-served OpenClaw campaign: run the 4 CANONICAL WildClawBench
# tasks LIVE against the k3s engine (Qwen2.5-Instruct-7B-AWQ, hermes tool parser, 32K ctx) and
# capture BOTH sides in the same windows:
#   DURING  = engine pod cgroup       (vLLM serving CPU)
#   OUTSIDE = task docker container   (OpenClaw agent + all its tools: node, browser, python)
# Per task: 2 parallel perf records (task-clock) + 7 stat groups x 20s via --for-each-cgroup.
# WORK GUARD per task: engine must report running requests AND the container must burn CPU;
# the agent process must survive the capture (protects against plan-only early-quit at 7B).
set -o pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export KUBECONFIG="$HOME/.kube/k3s-local.yaml"
PERF=$(ls -d /usr/lib/linux-tools-6.8*/perf 2>/dev/null | tail -1)
IMG=wildclawbench-ubuntu:v1.3
MODEL=my-openai-proxy/qwen2.5-7b-instruct-awq
REC_SEC="${REC_SEC:-30}"
log(){ printf '[oc-live] %s\n' "$*"; }

cleanup(){
  pkill -f "kubectl port-forward" 2>/dev/null
  docker ps -aq --filter ancestor=$IMG | xargs -r docker rm -f >/dev/null 2>&1
  kubectl scale deploy -n llm-d-local ms-local-llm-d-modelservice-decode --replicas=0 >/dev/null 2>&1
}
trap cleanup EXIT

sudo pkill -9 -x perf 2>/dev/null; sleep 2

# ---- engine up (Instruct-7B + hermes) ----
kubectl scale deploy -n llm-d-local ms-local-llm-d-modelservice-decode --replicas=1
kubectl rollout status deploy -n llm-d-local ms-local-llm-d-modelservice-decode --timeout=600s || { log "ERROR rollout"; exit 1; }
pkill -f "kubectl port-forward" 2>/dev/null; sleep 1
kubectl port-forward --address 0.0.0.0 -n llm-d-local svc/ms-local-decode-direct 8000:8000 > /tmp/oc_pf.log 2>&1 &
for i in $(seq 1 60); do
  m=$(curl -s --max-time 3 "http://127.0.0.1:8000/v1/models" | grep -o "qwen2.5-7b-instruct-awq" || true)
  [ -n "$m" ] && break; sleep 5
done
[ -n "${m:-}" ] || { log "ERROR: engine/port-forward never served instruct model"; exit 1; }
log "engine serving qwen2.5-7b-instruct-awq on :8000 (hermes tools, 32K)"

pod=$(kubectl get pod -n llm-d-local -l llm-d.ai/role=decode --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')
cid=$(kubectl get pod "$pod" -n llm-d-local -o jsonpath='{.status.containerStatuses[0].containerID}'); cid=${cid##*://}
ENG_PID=$(sudo k3s crictl inspect --output go-template --template '{{.info.pid}}' "$cid")
ENG_CG=$(sudo cat /proc/$ENG_PID/cgroup | sed 's/^0:://'); ENG_CG=${ENG_CG#/}
log "engine pod=$pod pid=$ENG_PID"

declare -A GRP
GRP[core]="task-clock,cycles,instructions,branches,branch-misses"
GRP[fp1]="cycles,instructions,fp_arith_inst_retired.scalar_single,fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.128b_packed_single,fp_arith_inst_retired.128b_packed_double"
GRP[fp2]="cycles,instructions,fp_arith_inst_retired.256b_packed_single,fp_arith_inst_retired.256b_packed_double,fp_arith_inst_retired.512b_packed_single,fp_arith_inst_retired.512b_packed_double"
GRP[cache]="cycles,instructions,mem_load_retired.l1_hit,mem_load_retired.l2_hit,mem_load_retired.l3_hit,mem_load_retired.l3_miss"
GRP[mlp]="cycles,instructions,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles,uops_executed.thread"
GRP[tma1]="slots,topdown-retiring,topdown-bad-spec,topdown-fe-bound,topdown-be-bound"
GRP[tma2]="slots,topdown-heavy-ops,topdown-br-mispredict,topdown-fetch-lat,topdown-mem-bound"

run_task(){
  local TASK="$1" LABEL="$2"
  local OUT="$REPO/local_agents/data/oc_live_${LABEL}"; mkdir -p "$OUT"; rm -f "$OUT"/*
  log "================ $LABEL ================"
  cd "$REPO/agentic/openclaw/external/WildClawBench"; . .venv/bin/activate
  docker ps -aq --filter ancestor=$IMG | xargs -r docker rm -f >/dev/null 2>&1
  python3 eval/run_batch.py --task "$TASK" --models-config my_api.json \
    --model "$MODEL" --parallel 1 </dev/null > "$OUT/agent.log" 2>&1 & AG=$!
  local CID=""; for i in $(seq 1 150); do CID=$(docker ps -q --filter ancestor=$IMG | head -1); [ -n "$CID" ] && break; kill -0 $AG 2>/dev/null || break; sleep 2; done
  if [ -z "$CID" ]; then log "NO_CONTAINER $LABEL"; kill $AG 2>/dev/null; wait $AG 2>/dev/null; return; fi
  for i in $(seq 1 300); do grep -q "Waiting for agent to finish" "$OUT/agent.log" 2>/dev/null && break; kill -0 $AG 2>/dev/null || break; sleep 1; done
  local FULL=$(docker inspect -f '{{.Id}}' "$CID"); local INITPID=$(docker inspect -f '{{.State.Pid}}' "$CID")
  local DRV_CG="system.slice/docker-${FULL}.scope"
  # WORK GUARD: engine must be generating for this container's agent
  local ok=0
  for i in $(seq 1 45); do
    kill -0 $AG 2>/dev/null || break
    r=$(kubectl logs -n llm-d-local deploy/ms-local-llm-d-modelservice-decode --tail=5 2>/dev/null \
        | grep -oE "Running: [0-9]+" | tail -1 | grep -oE "[0-9]+" || echo 0)
    [ "${r:-0}" -ge 1 ] && { ok=1; log "WORK VERIFIED $LABEL (Running:$r)"; break; }
    sleep 4
  done
  [ "$ok" = 1 ] || { log "WORK-FAIL $LABEL: engine never busy (agent quit?)"; tail -6 "$OUT/agent.log"; kill -9 $AG 2>/dev/null; wait $AG 2>/dev/null; docker rm -f "$CID" >/dev/null 2>&1; return; }
  # 7B agents quit early: records and stat groups must share the short active window.
  # task-clock records are software events, so they run concurrently with the HW-counter
  # stat groups without PMU contention; TMA groups go first (most important in-window).
  log "record x2 ${REC_SEC}s + stats 7 groups x 12s (parallel)"
  ( for g in tma1 core tma2 cache fp1 fp2 mlp; do
      local a=1; kill -0 $AG 2>/dev/null || a=0
      echo "$g agent_alive=$a" >> "$OUT/stat_groups_alive.txt"
      [ "$a" = 0 ] && log "WARN $LABEL: agent finished before group $g"
      sudo "$PERF" stat -a -e "${GRP[$g]}" --for-each-cgroup="$ENG_CG,$DRV_CG" -- sleep 12 2> "$OUT/group_${g}.txt"
    done ) & local STATS=$!
  sudo "$PERF" record -e task-clock -a --cgroup="$ENG_CG" -g -F 199 -o "$OUT/rec_engine.data" -- sleep "$REC_SEC" > "$OUT/rec_engine.err" 2>&1 &
  local P1=$!
  sudo "$PERF" record -e task-clock -a --cgroup="$DRV_CG" -g -F 199 -o "$OUT/rec_tool.data" -- sleep "$REC_SEC" > "$OUT/rec_tool.err" 2>&1
  wait $P1 $STATS
  kubectl logs -n llm-d-local deploy/ms-local-llm-d-modelservice-decode --tail=40 2>/dev/null \
    | grep -E "Running|Waiting" | tail -12 > "$OUT/vllm_status_tail.txt" || true
  local ALIVE=0; kill -0 $AG 2>/dev/null && ALIVE=1
  sudo "$PERF" report -i "$OUT/rec_engine.data" --stdio -g none --symfs="/proc/$ENG_PID/root" 2>/dev/null | grep -E "^\s+[0-9]" > "$OUT/engine_flat.txt" || true
  sudo "$PERF" report -i "$OUT/rec_engine.data" --stdio --sort=dso 2>/dev/null | grep -E "^\s+[0-9]" > "$OUT/engine_dso.txt" || true
  sudo "$PERF" report -i "$OUT/rec_tool.data" --stdio -g none --symfs="/proc/${INITPID}/root" 2>/dev/null | grep -E "^\s+[0-9]" > "$OUT/tool_flat.txt" || true
  sudo "$PERF" report -i "$OUT/rec_tool.data" --stdio --sort=dso 2>/dev/null | grep -E "^\s+[0-9]" > "$OUT/tool_dso.txt" || true
  sudo chown -R "$USER:$USER" "$OUT" 2>/dev/null
  # let the task run to its natural end (bounded)
  local s=0; while kill -0 $AG 2>/dev/null && [ $s -lt 420 ]; do sleep 5; s=$((s+5)); done
  kill -9 $AG 2>/dev/null; wait $AG 2>/dev/null
  docker ps -aq --filter ancestor=$IMG | xargs -r docker rm -f >/dev/null 2>&1
  local se=$(stat -c%s "$OUT/rec_engine.data" 2>/dev/null || echo 0); local st=$(stat -c%s "$OUT/rec_tool.data" 2>/dev/null || echo 0)
  local le=$(wc -l < "$OUT/engine_flat.txt" 2>/dev/null || echo 0); local lt=$(wc -l < "$OUT/tool_flat.txt" 2>/dev/null || echo 0)
  local tma_ok=$(head -2 "$OUT/stat_groups_alive.txt" 2>/dev/null | grep -c "agent_alive=1")
  local score=$(grep -oE "overall_score.*" "$OUT/agent.log" | tail -1 | grep -oE "[0-9]+\.[0-9]+" | tail -1)
  if [ "$se" -gt 100000 ] && [ "$st" -gt 50000 ] && [ "$le" -gt 5 ] && [ "$lt" -gt 5 ] && [ "$tma_ok" = 2 ]; then
    log "VALIDATE-OK $LABEL (eng=${se}B/${le}sym, tool=${st}B/${lt}sym, tma+core in-window, score=${score:-?}, alive-post=$ALIVE)"
  else
    log "VALIDATE-FAIL $LABEL (eng=${se}B/${le}sym, tool=${st}B/${lt}sym, in-window-groups=$tma_ok/2, score=${score:-?})"
  fi
}

# optional args = subset of task labels to run (default: all four)
WANT="${*:-calendar web-digest pdf-digest image-crop}"
has(){ case " $WANT " in *" $1 "*) return 0;; *) return 1;; esac; }
T=tasks/01_Productivity_Flow
has calendar   && run_task "$T/01_Productivity_Flow_task_6_calendar_scheduling.md"  calendar
has web-digest && run_task "$T/01_Productivity_Flow_task_1_arxiv_digest.md"         web-digest
has pdf-digest && run_task "$T/01_Productivity_Flow_task_10_pdf_digest.md"          pdf-digest
has image-crop && run_task "tasks/05_Creative_Synthesis/05_Creative_Synthesis_task_10_social_poster_multi_crop.md" image-crop
log "OC-CHAIN-DONE"
