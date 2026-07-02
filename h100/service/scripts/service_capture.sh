#!/usr/bin/env bash
# service_capture.sh — during/outside CPU characterization of the RAG service on k3s (H100).
# NEW script for this run (does NOT touch run_benchmark.sh). Learns the pod set from run_benchmark
# but uses the CGROUP method (perf -a --cgroup) instead of `perf stat -p <pid>` (which undercounts
# the many-threaded vLLM EngineCore). Captures, DURING a sustained RAG tok320 load:
#   (A) ATTRIBUTION  — perf record -e task-clock (call-graph) per pod cgroup -> perf_flat/perf_dso
#       for ALL work pods (inside vllm / routing envoy / outside fastapi+milvus+mongo+seaweed).
#   (B) MICROARCH    — perf stat core/fp1/fp2/cache/mlp groups per pod cgroup (CANONICAL events)
#       on vllm(INSIDE) + fastapi/milvus/seaweed(OUTSIDE) -> group_<pod>_<grp>.txt (parsed by
#       agentic/CANONICAL/microarch.py). No TMA (this KVM guest lacks the 'slots' PMU event).
set -o pipefail
export HOME=/home/ubuntu
export KUBECONFIG="$HOME/.kube/config"
STATUS="$HOME/service_perf/status"
status(){ echo "$(date +%H:%M:%S) $*" > "$STATUS"; }
PERF=/usr/bin/perf
OUT="${OUT:-$HOME/service_perf/data}"
REC_SEC="${REC_SEC:-25}"
STAT_SEC="${STAT_SEC:-20}"
FASTAPI_PORT="${FASTAPI_PORT:-18081}"
KROOT="$HOME/llm-service-kernel"
RAG_BUCKET="${RAG_BUCKET:-short}"
RAG_FILE="$KROOT/benchmark_queries/rag/${RAG_BUCKET}.txt"
CONC="${CONC:-3}"
MAXTOK="${MAXTOK:-320}"
export BENCHMARK_MODEL="${BENCHMARK_MODEL:-qwen2.5-32b-instruct}"
mkdir -p "$OUT"

log(){ printf '[capture] %s\n' "$*"; }

# key|namespace|label|class   (single-container pods -> containerStatuses[0])
SPECS=(
  "vllm|llm-d-local|llm-d.ai/role=decode|inside"
  "llmd_gateway|llm-d-local|app.kubernetes.io/component=inference-gateway|routing"
  "fastapi|llm-service|app=llm-service-kernel|outside"
  "milvus|llm-service|app=milvus|outside"
  "mongodb|llm-service|app=mongodb|outside"
  "seaweed_volume|llm-service|app=seaweed-volume|outside"
  "seaweed_filer|llm-service|app=seaweed-filer|outside"
)
# pods that get the full microarch counter suite (the interesting during/outside contrast)
MICRO_PODS=("vllm" "fastapi" "milvus" "seaweed_volume")

declare -A CG PID NS CLASS
resolve_all() {
  for s in "${SPECS[@]}"; do
    IFS='|' read -r key ns label cls <<<"$s"
    local pod cid pid cg
    pod=$(kubectl get pod -n "$ns" -l "$label" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    [ -z "$pod" ] && { log "WARN $key: no pod for label $label"; continue; }
    cid=$(kubectl get pod "$pod" -n "$ns" -o jsonpath='{.status.containerStatuses[0].containerID}' 2>/dev/null)
    cid=${cid##*://}
    pid=$(sudo k3s crictl inspect --output go-template --template '{{.info.pid}}' "$cid" 2>/dev/null)
    [ -z "$pid" ] && { log "WARN $key: no pid"; continue; }
    cg=$(sudo cat /proc/$pid/cgroup 2>/dev/null | sed 's/^0:://')
    cg=${cg#/}
    CG[$key]="$cg"; PID[$key]="$pid"; NS[$key]="$ns"; CLASS[$key]="$cls"
    printf '[capture]  %-15s pid=%-7s cg=%s\n' "$key" "$pid" "$cg"
  done
}

# ---- microarch counter groups (CANONICAL events; <=6 -> no multiplexing on ~8 GP counters) ----
declare -A GRP
GRP[core]="task-clock,cycles,instructions,branches,branch-misses"
GRP[fp1]="cycles,instructions,fp_arith_inst_retired.scalar_single,fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.128b_packed_single,fp_arith_inst_retired.128b_packed_double"
GRP[fp2]="cycles,instructions,fp_arith_inst_retired.256b_packed_single,fp_arith_inst_retired.256b_packed_double,fp_arith_inst_retired.512b_packed_single,fp_arith_inst_retired.512b_packed_double"
GRP[cache]="cycles,instructions,mem_load_retired.l1_hit,mem_load_retired.l2_hit,mem_load_retired.l3_hit,mem_load_retired.l3_miss"
GRP[mlp]="cycles,instructions,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles,uops_executed.thread"
GRP_ORDER=(core fp1 fp2 cache mlp)

LOAD_PID=""; PF_PID=""
start_load() {
  log "port-forward svc/llm-service-kernel :$FASTAPI_PORT -> 8080"
  ( kubectl port-forward -n llm-service svc/llm-service-kernel "$FASTAPI_PORT:8080" >/tmp/pf.log 2>&1 ) & PF_PID=$!
  sleep 4
  log "steady-state RAG load: mode=rag bucket=$RAG_BUCKET tok=$MAXTOK conc=$CONC (loop)"
  (
    export BENCHMARK_URL="http://localhost:$FASTAPI_PORT" BENCHMARK_MODEL="qwen2.5-32b-instruct"
    while true; do
      python3 "$KROOT/scripts/query_runner.py" --mode rag --queries "$RAG_FILE" \
        --size-bucket "$RAG_BUCKET" --count 1000 --warmup 0 --max-tokens "$MAXTOK" \
        --concurrency "$CONC" --out-dir "$OUT/loadgen" >/tmp/loadgen.log 2>&1 || sleep 1
    done
  ) & LOAD_PID=$!
}
stop_load() {
  [ -n "$LOAD_PID" ] && kill "$LOAD_PID" 2>/dev/null; pkill -f query_runner.py 2>/dev/null
  [ -n "$PF_PID" ] && kill "$PF_PID" 2>/dev/null
}
trap stop_load EXIT

verify_rag() {
  export BENCHMARK_URL="http://localhost:$FASTAPI_PORT"
  python3 "$KROOT/scripts/query_runner.py" --mode rag --queries "$RAG_FILE" \
    --size-bucket "$RAG_BUCKET" --count 3 --warmup 0 --max-tokens 32 \
    --concurrency 1 --out-dir "$OUT/verify" >/tmp/verify.log 2>&1 || true
  local csv; csv=$(ls -t "$OUT"/verify/*.csv 2>/dev/null | head -1)
  if [ -n "$csv" ]; then
    log "RAG verify (num_chunks / top_score / route):"
    python3 - "$csv" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1])))
for r in rows[:3]:
    print(f"   chunks={r.get('rag_num_chunks')} top_score={r.get('rag_top_score')} route={r.get('route','?')} n_out={r.get('n_output_tokens')}")
PY
  fi
}

attribution() {
  log "=== (A) ATTRIBUTION: parallel perf record task-clock, ${REC_SEC}s window, all pods ==="
  declare -A RP
  for key in "${!CG[@]}"; do
    sudo "$PERF" record -e task-clock -a --cgroup="${CG[$key]}" -g -F 199 \
      -o "$OUT/rec_${key}.data" -- sleep "$REC_SEC" >/dev/null 2>&1 & RP[$key]=$!
  done
  for key in "${!RP[@]}"; do wait "${RP[$key]}" 2>/dev/null; done
  log "records done; generating flat + dso reports (symfs=/proc/<pid>/root)"
  for key in "${!CG[@]}"; do
    local symfs="/proc/${PID[$key]}/root"
    sudo "$PERF" report -i "$OUT/rec_${key}.data" --stdio -g none --symfs="$symfs" 2>/dev/null \
      | grep -E "^\s+[0-9]" > "$OUT/${key}_flat.txt" || true
    sudo "$PERF" report -i "$OUT/rec_${key}.data" --stdio --sort=dso 2>/dev/null \
      | grep -E "^\s+[0-9]" > "$OUT/${key}_dso.txt" || true
    printf '[capture]  %-15s top-dso: ' "$key"; head -1 "$OUT/${key}_dso.txt" 2>/dev/null | sed 's/^ *//' | cut -c1-70
  done
}

microarch() {
  log "=== (B) MICROARCH: perf stat groups (${STAT_SEC}s each) on: ${MICRO_PODS[*]} ==="
  for key in "${MICRO_PODS[@]}"; do
    [ -z "${CG[$key]:-}" ] && { log "skip $key (no cgroup)"; continue; }
    status "microarch:$key"
    for g in "${GRP_ORDER[@]}"; do
      sudo "$PERF" stat -a -e "${GRP[$g]}" --for-each-cgroup="${CG[$key]}" -- sleep "$STAT_SEC" \
        2> "$OUT/group_${key}_${g}.txt"
    done
    # quick IPC read-back
    local ipc; ipc=$(python3 - "$OUT/group_${key}_core.txt" <<'PY'
import sys,re
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
# self-clean stale artifacts from prior runs (safe: these patterns do NOT match this script's own
# cmdline 'bash .../service_capture.sh', and we skip our own PID when killing sibling scripts).
for pid in $(pgrep -f service_capture.sh 2>/dev/null); do
  [ "$pid" = "$$" ] && continue; [ "$pid" = "$PPID" ] && continue
  sudo kill -9 "$pid" 2>/dev/null
done
sudo pkill -9 -f query_runner.py 2>/dev/null
sudo pkill -9 -f "kubectl.*port-forward" 2>/dev/null
sudo pkill -9 -x perf 2>/dev/null
sleep 2
log "resolving pod cgroups..."; status "resolve"; resolve_all
status "start_load"; start_load
sleep 8
status "verify_rag"; verify_rag
status "attribution"; attribution
status "microarch"; microarch
stop_load
status "DONE"
log "DONE. data in $OUT"
