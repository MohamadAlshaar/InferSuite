#!/usr/bin/env bash
# capture_tiers.sh — phase 3 of the LOCAL service run: per-tier (tok64/192/320) CPU capture of all
# service pods DURING a verified continuous RAG load, plus one idle control. Local box has the full
# PMU, so on top of the 5 portable groups (CANONICAL) we add two TMA groups (fixed-counter topdown
# L1 + td2 L2 events) — the measurement neither cloud campaign could take.
# Collection only (no plotting). Adapted from h100/service/scripts/service_capture2.sh.
set -o pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export KUBECONFIG="$HOME/.kube/k3s-local.yaml"
PERF=$(ls -d /usr/lib/linux-tools-6.8*/perf 2>/dev/null | tail -1)  # newest 6.8-series tools (6.17 wrapper broken)
OUT_ROOT="${OUT_ROOT:-$REPO/local_service/data}"
REC_SEC="${REC_SEC:-25}"
STAT_SEC="${STAT_SEC:-20}"
TIERS=(${TIERS:-64 192 320})
WARMUP_S="${WARMUP_S:-45}"
log(){ printf '[capture] %s\n' "$*"; }

[ -x "$PERF" ] || { log "ERROR: $PERF missing"; exit 1; }

SPECS=(
  "vllm|llm-d-local|llm-d.ai/role=decode|inside"
  "llmd_gateway|llm-d-local|app.kubernetes.io/component=inference-gateway|routing"
  "fastapi|llm-service|app=llm-service-kernel|outside"
  "milvus|llm-service|app=milvus|outside"
  "mongodb|llm-service|app=mongodb|outside"
  "seaweed_volume|llm-service|app=seaweed-volume|outside"
  "seaweed_filer|llm-service|app=seaweed-filer|outside"
)
MICRO_PODS=("vllm" "fastapi" "milvus" "mongodb" "seaweed_filer" "seaweed_volume")

declare -A CG PID
resolve_all(){
  CG=(); PID=()
  for s in "${SPECS[@]}"; do
    IFS='|' read -r key ns label cls <<<"$s"
    local pod cid pid cg
    pod=$(kubectl get pod -n "$ns" -l "$label" --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    [ -z "$pod" ] && { log "WARN $key: no pod"; continue; }
    cid=$(kubectl get pod "$pod" -n "$ns" -o jsonpath='{.status.containerStatuses[0].containerID}' 2>/dev/null); cid=${cid##*://}
    pid=$(sudo k3s crictl inspect --output go-template --template '{{.info.pid}}' "$cid" 2>/dev/null)
    [ -z "$pid" ] && { log "WARN $key: no pid"; continue; }
    cg=$(sudo cat /proc/$pid/cgroup 2>/dev/null | sed 's/^0:://'); cg=${cg#/}
    CG[$key]="$cg"; PID[$key]="$pid"
    printf '[capture]  %-15s pid=%-8s cg=%s\n' "$key" "$pid" "${cg:0:80}"
  done
}

declare -A GRP
GRP[core]="task-clock,cycles,instructions,branches,branch-misses"
GRP[fp1]="cycles,instructions,fp_arith_inst_retired.scalar_single,fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.128b_packed_single,fp_arith_inst_retired.128b_packed_double"
GRP[fp2]="cycles,instructions,fp_arith_inst_retired.256b_packed_single,fp_arith_inst_retired.256b_packed_double,fp_arith_inst_retired.512b_packed_single,fp_arith_inst_retired.512b_packed_double"
GRP[cache]="cycles,instructions,mem_load_retired.l1_hit,mem_load_retired.l2_hit,mem_load_retired.l3_hit,mem_load_retired.l3_miss"
GRP[mlp]="cycles,instructions,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles,uops_executed.thread"
# TMA — the local-only lens (fixed-counter topdown; slots first = group leader)
GRP[tma1]="slots,topdown-retiring,topdown-bad-spec,topdown-fe-bound,topdown-be-bound"
GRP[tma2]="slots,topdown-heavy-ops,topdown-br-mispredict,topdown-fetch-lat,topdown-mem-bound"
GRP_ORDER=(core fp1 fp2 cache mlp tma1 tma2)

attribution(){ # $1 = outdir
  local OUT="$1"
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

microarch(){ # $1 = outdir
  local OUT="$1"
  log "=== (B) MICROARCH+TMA: perf stat ${#GRP_ORDER[@]} groups x ${STAT_SEC}s on: ${MICRO_PODS[*]} ==="
  for key in "${MICRO_PODS[@]}"; do
    [ -z "${CG[$key]:-}" ] && { log "skip $key (no cgroup)"; continue; }
    for g in "${GRP_ORDER[@]}"; do
      sudo "$PERF" stat -a -e "${GRP[$g]}" --for-each-cgroup="${CG[$key]}" -- sleep "$STAT_SEC" \
        2> "$OUT/group_${key}_${g}.txt"
    done
    log "  $key done"
  done
}

provenance(){ # $1 = outdir
  local OUT="$1"
  kubectl logs -n llm-d-local deploy/ms-local-llm-d-modelservice-decode --tail=80 2>/dev/null \
    | grep -E "Running|Waiting|throughput" | tail -25 > "$OUT/vllm_status_tail.txt" || true
  kubectl logs -n llm-service deploy/loadgen --tail=12 > "$OUT/loadgen_tail.txt" 2>/dev/null || true
}

set_loadgen(){ # $1 = max_tokens ("" = delete)
  if [ -z "$1" ]; then
    kubectl scale deploy/loadgen -n llm-service --replicas=0 2>/dev/null || true
    return
  fi
  sed "s/__MAX_TOKENS__/$1/" "$REPO/local_service/k3s_deploy/loadgen-tier.yaml" | kubectl apply -f -
  kubectl scale deploy/loadgen -n llm-service --replicas=1
  kubectl rollout status deploy/loadgen -n llm-service --timeout=120s
}

wait_engine_busy(){
  log "waiting for sustained load (vLLM 'Running: N>0')..."
  for i in $(seq 1 40); do
    r=$(kubectl logs -n llm-d-local deploy/ms-local-llm-d-modelservice-decode --tail=5 2>/dev/null \
        | grep -oE "Running: [0-9]+" | tail -1 | grep -oE "[0-9]+" || echo 0)
    [ "${r:-0}" -ge 1 ] && { log "engine busy (Running: $r)"; return 0; }
    sleep 5
  done
  log "WARN: engine never reported Running>=1 — check loadgen"; return 1
}

validate_tier(){ # $1 = outdir ; hard checks: mux, not-counted, zeroed cycles, empty rec, load verified
  local OUT="$1" bad=0
  # (1) counter integrity: no <not counted>/<not supported>, no multiplex tag < 99.5%
  for f in "$OUT"/group_*.txt; do
    [ -e "$f" ] || continue
    if grep -qE "<not counted>|<not supported>" "$f"; then
      log "VALIDATE-FAIL: $f has not-counted/not-supported rows"; bad=1
    fi
    muxn=$(grep -oE '\([0-9]{1,2}\.[0-9]+%\)' "$f" 2>/dev/null | awk -F'[(%]' '$2+0 < 99.5 {c++} END {print c+0}')
    if [ "${muxn:-0}" -gt 0 ]; then
      log "VALIDATE-FAIL: $f has $muxn multiplexed rows (<99.5% active)"; bad=1
    fi
    if ! grep -E "cycles|slots" "$f" | grep -qE "[0-9][0-9,]{4,}"; then
      log "VALIDATE-FAIL: $f has no substantial cycles/slots counts (near-zero capture?)"; bad=1
    fi
  done
  # (2) attribution non-empty for the pods that matter
  for key in vllm fastapi; do
    if [ ! -s "$OUT/${key}_flat.txt" ]; then log "VALIDATE-FAIL: ${key}_flat.txt empty"; bad=1; fi
    sz=$(stat -c%s "$OUT/rec_${key}.data" 2>/dev/null || echo 0)
    [ "$sz" -lt 20000 ] && { log "VALIDATE-FAIL: rec_${key}.data only ${sz}B"; bad=1; }
  done
  # (3) load verified during the window (skip for idle control)
  if [ "$(basename "$OUT")" != "idle_control" ]; then
    if ! grep -qE "Running: [1-9]" "$OUT/vllm_status_tail.txt" 2>/dev/null; then
      log "VALIDATE-FAIL: no 'Running: >=1' in vLLM status during window (engine idle?)"; bad=1
    fi
  fi
  if [ "$bad" = 0 ]; then log "VALIDATE-OK: $(basename "$OUT")"; else
    log "VALIDATE: $(basename "$OUT") HAS FAILURES — see above"
    echo "$(basename "$OUT")" >> "$OUT_ROOT/FAILED_TIERS.txt"
  fi
  return $bad
}

sudo pkill -9 -x perf 2>/dev/null; sleep 2   # stale-orphan-perf gotcha
mkdir -p "$OUT_ROOT"

for T in "${TIERS[@]}"; do
  OUT="$OUT_ROOT/tok$T"; mkdir -p "$OUT"
  log "################ TIER tok$T ################"
  set_loadgen "$T"
  wait_engine_busy || true
  log "warmup ${WARMUP_S}s (steady state, prefix caches settle)"; sleep "$WARMUP_S"
  resolve_all
  attribution "$OUT"
  microarch "$OUT"
  provenance "$OUT"
  validate_tier "$OUT" || log "tier tok$T flagged — continuing (re-run later if needed)"
done

log "################ IDLE CONTROL ################"
OUT="$OUT_ROOT/idle_control"; mkdir -p "$OUT"
set_loadgen ""
log "draining (90s) so the engine parks"; sleep 90
resolve_all
attribution "$OUT"
for key in vllm fastapi; do
  sudo "$PERF" stat -a -e "${GRP[core]}" --for-each-cgroup="${CG[$key]}" -- sleep "$STAT_SEC" \
    2> "$OUT/group_${key}_core.txt"
done
provenance "$OUT"
validate_tier "$OUT" || true

if [ -s "$OUT_ROOT/FAILED_TIERS.txt" ]; then
  log "SOME WINDOWS FAILED VALIDATION: $(tr '\n' ' ' < "$OUT_ROOT/FAILED_TIERS.txt")"
else
  log "ALL WINDOWS VALIDATED CLEAN (no mux, no zeroing, non-empty, load verified)"
fi
log "DONE. data in $OUT_ROOT (loadgen left at 0 replicas)"
