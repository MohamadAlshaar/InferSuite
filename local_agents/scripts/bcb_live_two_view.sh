#!/usr/bin/env bash
# bcb_live_two_view.sh — LOCAL self-served BCB campaign: run agentic_bcb.py LIVE against the k3s
# engine (Qwen2.5-Coder-7B-AWQ) and capture BOTH sides in the same windows:
#   DURING  = engine pod cgroup   (vLLM serving CPU)
#   OUTSIDE = driver scope cgroup (agent loop + tool/test execution)
# Per side: perf record task-clock (software view) + 7 stat groups (portable suite + TMA L1 + td2),
# stats via one --for-each-cgroup pass per group so both cgroups share each 20 s window.
# WORK GUARD: aborts unless tool-exec markers advance and the engine reports running requests
# (protects against the degenerate-agent failure mode seen with temp=0).
set -o pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export KUBECONFIG="$HOME/.kube/k3s-local.yaml"
PERF=$(ls -d /usr/lib/linux-tools-6.8*/perf 2>/dev/null | tail -1)
OUT="$REPO/local_agents/data/bcb_live"; mkdir -p "$OUT"
N_TASKS="${N_TASKS:-12}"; MAX_TURNS="${MAX_TURNS:-3}"
REC_SEC="${REC_SEC:-30}"; WARMUP_S="${WARMUP_S:-45}"
ENDPOINT="http://10.43.21.159:8000/v1"
log(){ printf '[bcb-live] %s\n' "$*"; }

cleanup(){
  pkill -f "agentic_bcb.py" 2>/dev/null
  kubectl scale deploy -n llm-d-local ms-local-llm-d-modelservice-decode --replicas=0 >/dev/null 2>&1
}
trap cleanup EXIT

sudo pkill -9 -x perf 2>/dev/null; sleep 2

# ---- engine up (Coder-7B) ----
kubectl scale deploy -n llm-d-local ms-local-llm-d-modelservice-decode --replicas=1
kubectl rollout status deploy -n llm-d-local ms-local-llm-d-modelservice-decode --timeout=600s || { log "ERROR rollout"; exit 1; }
for i in $(seq 1 60); do
  m=$(curl -s --max-time 3 "$ENDPOINT/models" | grep -o "qwen2.5-coder-7b-instruct-awq" || true)
  [ -n "$m" ] && break; sleep 5
done
[ -n "${m:-}" ] || { log "ERROR: engine never served coder model"; exit 1; }
log "engine serving qwen2.5-coder-7b-instruct-awq"

pod=$(kubectl get pod -n llm-d-local -l llm-d.ai/role=decode --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')
cid=$(kubectl get pod "$pod" -n llm-d-local -o jsonpath='{.status.containerStatuses[0].containerID}'); cid=${cid##*://}
ENG_PID=$(sudo k3s crictl inspect --output go-template --template '{{.info.pid}}' "$cid")
ENG_CG=$(sudo cat /proc/$ENG_PID/cgroup | sed 's/^0:://'); ENG_CG=${ENG_CG#/}
log "engine pod=$pod pid=$ENG_PID"

# ---- driver in its own scope cgroup ----
rm -f /tmp/bcb_agentic_markers.txt
systemd-run --user --scope --unit=bcb-live --collect -- bash -c \
  "cd '$REPO/agentic/bigcodebench' && HEAVY_LIBS=${HEAVY_LIBS:-} VLLM='$ENDPOINT' MODEL=qwen2.5-coder-7b-instruct-awq .venv/bin/python agentic_bcb.py $N_TASKS $MAX_TURNS" \
  > "$OUT/agent.log" 2>&1 &
sleep 3
DRV_PID=$(pgrep -f "agentic_bcb.py $N_TASKS" | head -1)
[ -n "$DRV_PID" ] || { log "ERROR: driver did not start"; tail -5 "$OUT/agent.log"; exit 1; }
DRV_CG=$(cat /proc/$DRV_PID/cgroup | sed 's/^0:://'); DRV_CG=${DRV_CG#/}
log "driver pid=$DRV_PID cgroup=$DRV_CG"

# ---- WORK GUARD: engine busy + first tool-exec marker before we spend capture time ----
ok=0
for i in $(seq 1 60); do
  kill -0 "$DRV_PID" 2>/dev/null || { log "ERROR: driver died early"; tail -8 "$OUT/agent.log"; exit 1; }
  r=$(kubectl logs -n llm-d-local deploy/ms-local-llm-d-modelservice-decode --tail=5 2>/dev/null \
      | grep -oE "Running: [0-9]+" | tail -1 | grep -oE "[0-9]+" || echo 0)
  mk=$(grep -c toolexec /tmp/bcb_agentic_markers.txt 2>/dev/null || echo 0)
  if [ "${r:-0}" -ge 1 ] && [ "$mk" -ge 2 ]; then ok=1; log "WORK VERIFIED (Running:$r, $mk exec markers)"; break; fi
  sleep 5
done
[ "$ok" = 1 ] || { log "ERROR: no real agent work within 5min (markers=$(grep -c toolexec /tmp/bcb_agentic_markers.txt 2>/dev/null))"; tail -8 "$OUT/agent.log"; exit 1; }

( while kill -0 "$DRV_PID" 2>/dev/null; do
    printf "%s,%s\n" "$(date +%s.%N)" "$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    sleep 0.5
  done >> "$OUT/gpu_timeline.csv" ) & GPUSAMP=$!
log "warmup ${WARMUP_S}s"; sleep "$WARMUP_S"
MK0=$(grep -c toolexec /tmp/bcb_agentic_markers.txt 2>/dev/null || echo 0)

# ---- software view: two parallel task-clock records, same window ----
log "perf record x2 (engine + driver) ${REC_SEC}s"
sudo "$PERF" record -e task-clock -a --cgroup="$ENG_CG" -g -F 199 -o "$OUT/rec_engine.data" -- sleep "$REC_SEC" > "$OUT/rec_engine.err" 2>&1 &
P1=$!
sudo "$PERF" record -e task-clock -a --cgroup="$DRV_CG" -g -F 199 -o "$OUT/rec_driver.data" -- sleep "$REC_SEC" > "$OUT/rec_driver.err" 2>&1
wait $P1

# ---- micro measures: 7 groups, both cgroups per window ----
log "perf stat: 7 groups x 20s on {engine,driver} cgroups"
declare -A GRP
GRP[core]="task-clock,cycles,instructions,branches,branch-misses"
GRP[fp1]="cycles,instructions,fp_arith_inst_retired.scalar_single,fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.128b_packed_single,fp_arith_inst_retired.128b_packed_double"
GRP[fp2]="cycles,instructions,fp_arith_inst_retired.256b_packed_single,fp_arith_inst_retired.256b_packed_double,fp_arith_inst_retired.512b_packed_single,fp_arith_inst_retired.512b_packed_double"
GRP[cache]="cycles,instructions,mem_load_retired.l1_hit,mem_load_retired.l2_hit,mem_load_retired.l3_hit,mem_load_retired.l3_miss"
GRP[mlp]="cycles,instructions,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles,uops_executed.thread"
GRP[tma1]="slots,topdown-retiring,topdown-bad-spec,topdown-fe-bound,topdown-be-bound"
GRP[tma2]="slots,topdown-heavy-ops,topdown-br-mispredict,topdown-fetch-lat,topdown-mem-bound"
for g in core fp1 fp2 cache mlp tma1 tma2; do
  kill -0 "$DRV_PID" 2>/dev/null || log "WARN: driver finished before group $g"
  sudo "$PERF" stat -a -e "${GRP[$g]}" --for-each-cgroup="$ENG_CG,$DRV_CG" -- sleep 20 2> "$OUT/group_${g}.txt"
done
MK1=$(grep -c toolexec /tmp/bcb_agentic_markers.txt 2>/dev/null || echo 0)
kubectl logs -n llm-d-local deploy/ms-local-llm-d-modelservice-decode --tail=40 2>/dev/null \
  | grep -E "Running|Waiting" | tail -12 > "$OUT/vllm_status_tail.txt" || true

# ---- reports ----
sudo "$PERF" report -i "$OUT/rec_engine.data" --stdio -g none --symfs="/proc/$ENG_PID/root" 2>/dev/null | grep -E "^\s+[0-9]" > "$OUT/engine_flat.txt" || true
sudo "$PERF" report -i "$OUT/rec_engine.data" --stdio --sort=dso 2>/dev/null | grep -E "^\s+[0-9]" > "$OUT/engine_dso.txt" || true
sudo "$PERF" report -i "$OUT/rec_driver.data" --stdio -g none 2>/dev/null | grep -E "^\s+[0-9]" > "$OUT/driver_flat.txt" || true
sudo "$PERF" report -i "$OUT/rec_driver.data" --stdio --sort=dso 2>/dev/null | grep -E "^\s+[0-9]" > "$OUT/driver_dso.txt" || true
sudo chown -R "$USER:$USER" "$OUT" 2>/dev/null

# ---- let the loop finish (its final tally is the work evidence) ----
log "capture done; waiting for agent loop to finish"
while kill -0 "$DRV_PID" 2>/dev/null; do sleep 15; done
cp /tmp/bcb_agentic_markers.txt "$OUT/markers.txt" 2>/dev/null
tail -3 "$OUT/agent.log" | grep -v "^$" || true

# ---- validation ----
se=$(stat -c%s "$OUT/rec_engine.data" 2>/dev/null || echo 0); sd=$(stat -c%s "$OUT/rec_driver.data" 2>/dev/null || echo 0)
le=$(wc -l < "$OUT/engine_flat.txt" 2>/dev/null || echo 0); ld=$(wc -l < "$OUT/driver_flat.txt" 2>/dev/null || echo 0)
mkd=$((MK1 - MK0))
if [ "$se" -gt 100000 ] && [ "$sd" -gt 50000 ] && [ "$le" -gt 5 ] && [ "$ld" -gt 5 ] && [ "$mkd" -ge 1 ]; then
  log "VALIDATE-OK (eng=${se}B/${le}sym, drv=${sd}B/${ld}sym, +${mkd} exec markers during capture)"
else
  log "VALIDATE-FAIL (eng=${se}B/${le}sym, drv=${sd}B/${ld}sym, markers+${mkd})"
fi
log "CHAIN-DONE"
