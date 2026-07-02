#!/usr/bin/env bash
# service_capture2.sh — perf-only capture (load is driven externally by the in-cluster `loadgen`
# Deployment, so NO host port-forward and NO killing of query_runner here). Measures the 7 service
# pods DURING a verified continuous RAG tok320 load (vLLM Running: 6 reqs). Writes to data2/.
set -o pipefail
export HOME=/home/ubuntu
export KUBECONFIG="$HOME/.kube/config"
PERF=/usr/bin/perf
OUT="${OUT:-$HOME/service_perf/data2}"
REC_SEC="${REC_SEC:-25}"
STAT_SEC="${STAT_SEC:-20}"
STATUS="$HOME/service_perf/status2"
mkdir -p "$OUT"
status(){ echo "$(date +%H:%M:%S) $*" > "$STATUS"; }
log(){ printf '[capture] %s\n' "$*"; }

SPECS=(
  "vllm|llm-d-local|llm-d.ai/role=decode|inside"
  "llmd_gateway|llm-d-local|app.kubernetes.io/component=inference-gateway|routing"
  "fastapi|llm-service|app=llm-service-kernel|outside"
  "milvus|llm-service|app=milvus|outside"
  "mongodb|llm-service|app=mongodb|outside"
  "seaweed_volume|llm-service|app=seaweed-volume|outside"
  "seaweed_filer|llm-service|app=seaweed-filer|outside"
)
MICRO_PODS=("vllm" "fastapi" "milvus" "mongodb")

declare -A CG PID
resolve_all(){
  for s in "${SPECS[@]}"; do
    IFS='|' read -r key ns label cls <<<"$s"
    local pod cid pid cg
    pod=$(kubectl get pod -n "$ns" -l "$label" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    [ -z "$pod" ] && { log "WARN $key: no pod"; continue; }
    cid=$(kubectl get pod "$pod" -n "$ns" -o jsonpath='{.status.containerStatuses[0].containerID}' 2>/dev/null); cid=${cid##*://}
    pid=$(sudo k3s crictl inspect --output go-template --template '{{.info.pid}}' "$cid" 2>/dev/null)
    [ -z "$pid" ] && { log "WARN $key: no pid"; continue; }
    cg=$(sudo cat /proc/$pid/cgroup 2>/dev/null | sed 's/^0:://'); cg=${cg#/}
    CG[$key]="$cg"; PID[$key]="$pid"
    printf '[capture]  %-15s pid=%-7s cg=%s\n' "$key" "$pid" "$cg"
  done
}

declare -A GRP
GRP[core]="task-clock,cycles,instructions,branches,branch-misses"
GRP[fp1]="cycles,instructions,fp_arith_inst_retired.scalar_single,fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.128b_packed_single,fp_arith_inst_retired.128b_packed_double"
GRP[fp2]="cycles,instructions,fp_arith_inst_retired.256b_packed_single,fp_arith_inst_retired.256b_packed_double,fp_arith_inst_retired.512b_packed_single,fp_arith_inst_retired.512b_packed_double"
GRP[cache]="cycles,instructions,mem_load_retired.l1_hit,mem_load_retired.l2_hit,mem_load_retired.l3_hit,mem_load_retired.l3_miss"
GRP[mlp]="cycles,instructions,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles,uops_executed.thread"
GRP_ORDER=(core fp1 fp2 cache mlp)

attribution(){
  log "=== (A) ATTRIBUTION: parallel perf record task-clock, ${REC_SEC}s ==="
  declare -A RP
  for key in "${!CG[@]}"; do
    sudo "$PERF" record -e task-clock -a --cgroup="${CG[$key]}" -g -F 199 \
      -o "$OUT/rec_${key}.data" -- sleep "$REC_SEC" >/dev/null 2>&1 & RP[$key]=$!
  done
  for key in "${!RP[@]}"; do wait "${RP[$key]}" 2>/dev/null; done
  for key in "${!CG[@]}"; do
    local symfs="/proc/${PID[$key]}/root"
    sudo "$PERF" report -i "$OUT/rec_${key}.data" --stdio -g none --symfs="$symfs" 2>/dev/null \
      | grep -E "^\s+[0-9]" > "$OUT/${key}_flat.txt" || true
    sudo "$PERF" report -i "$OUT/rec_${key}.data" --stdio --sort=dso 2>/dev/null \
      | grep -E "^\s+[0-9]" > "$OUT/${key}_dso.txt" || true
    printf '[capture]  %-15s top-dso: ' "$key"; head -1 "$OUT/${key}_dso.txt" 2>/dev/null | sed 's/^ *//' | cut -c1-70
  done
}

microarch(){
  log "=== (B) MICROARCH: perf stat groups (${STAT_SEC}s) on: ${MICRO_PODS[*]} ==="
  for key in "${MICRO_PODS[@]}"; do
    [ -z "${CG[$key]:-}" ] && { log "skip $key (no cgroup)"; continue; }
    status "microarch:$key"
    for g in "${GRP_ORDER[@]}"; do
      sudo "$PERF" stat -a -e "${GRP[$g]}" --for-each-cgroup="${CG[$key]}" -- sleep "$STAT_SEC" \
        2> "$OUT/group_${key}_${g}.txt"
    done
    local ipc; ipc=$(python3 - "$OUT/group_${key}_core.txt" <<'PY'
import sys
c=i=0
for ln in open(sys.argv[1]):
    p=ln.split()
    if len(p)>=2:
        try: v=float(p[0].replace(',',''))
        except: continue
        if p[1]=='cycles': c=v
        elif p[1]=='instructions': i=v
print(f"{i/c:.2f}" if c else "n/a")
PY
)
    log "  $key IPC=${ipc}"
  done
}

status "START"
sudo pkill -9 -x perf 2>/dev/null; sleep 2
log "resolving pod cgroups..."; status "resolve"; resolve_all
status "attribution"; attribution
status "microarch"; microarch
status "DONE"
log "DONE. data in $OUT"
