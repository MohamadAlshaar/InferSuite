#!/usr/bin/env bash
# run_service_campaign.sh — the LOCAL SERVICE under the GLM-agent methodology:
# same isolation grade (shielded slices, pinned partition, fixed clocks, THP off, IRQ
# steering, zero foreign processes), same 9 zero-multiplexing counter groups (merged TMA,
# fe, icache, priv kernel-split), cycling windows, 10 Hz per-pod timelines, full-episode
# records, E-proofs, and the repetition layer: each cell (bucket x tier) runs REPEATS times
# under an IDENTICAL deterministic load (same queries, exact-token forcing) — cross-repeat
# dispersion certifies measurement precision AND workload stationarity in one number.
#
#   ./run_service_campaign.sh {preflight|dryrun|isolation-test|campaign|validate|all}
#
# k3s-aware isolation (the inverse of the agent campaign):
#   kubepods.slice            -> MEASURED cpus (all workload pods, one hierarchical pin)
#   kube-system pods, loadgen -> HOUSE cpus (per-pod-slice override; loadgen = the driver)
#   k3s control plane          (system.slice/k3s.service) + everything else -> HOUSE
set -o pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$KIT/../../.." && pwd)"
source "$KIT/service_campaign.conf"
export KUBECONFIG="$HOME/.kube/k3s-local.yaml"
DATA="$REPO/local_service/data_iso"; mkdir -p "$DATA"
STATE="$KIT/.state"; mkdir -p "$STATE"
PERF=$(ls -d /usr/lib/linux-tools-6.8*/perf 2>/dev/null | tail -1)
log(){ printf '[svc %s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$KIT/campaign.log"; }

# ---- counter groups: identical to the certified GLM agent set --------------------------------
declare -A GRP
GRP[tma]="slots,topdown-retiring,topdown-bad-spec,topdown-fe-bound,topdown-be-bound,topdown-heavy-ops,topdown-br-mispredict,topdown-fetch-lat,topdown-mem-bound"
GRP[core]="task-clock,cycles,instructions,branches,branch-misses"
GRP[cache]="cycles,instructions,mem_load_retired.l1_hit,mem_load_retired.l2_hit,mem_load_retired.l3_hit,mem_load_retired.l3_miss"
GRP[fp1]="cycles,instructions,fp_arith_inst_retired.scalar_single,fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.128b_packed_single,fp_arith_inst_retired.128b_packed_double"
GRP[fp2]="cycles,instructions,fp_arith_inst_retired.256b_packed_single,fp_arith_inst_retired.256b_packed_double,fp_arith_inst_retired.512b_packed_single,fp_arith_inst_retired.512b_packed_double"
GRP[mlp]="cycles,instructions,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles,uops_executed.thread"
GRP[fe]="cycles,instructions,idq.dsb_uops,idq.mite_uops,idq.ms_uops,lsd.uops"
GRP[icache]="cycles,instructions,l2_rqsts.all_code_rd,l2_rqsts.code_rd_miss,icache_data.stalls"
GRP[priv]="task-clock,context-switches,cpu-migrations,page-faults,cycles:u,cycles:k,instructions:u,instructions:k"
GORDER="tma core cache fp1 fp2 mlp fe icache priv"

# ---- pods (fences). BGE embedder runs in-process in the fastapi pod. -------------------------
SPECS=(
  "vllm|llm-d-local|llm-d.ai/role=decode"
  "fastapi|llm-service|app=llm-service-kernel"
  "milvus|llm-service|app=milvus"
  "mongodb|llm-service|app=mongodb"
  "seaweed_filer|llm-service|app=seaweed-filer"
  "seaweed_volume|llm-service|app=seaweed-volume"
)
declare -A CG PID
resolve_pods(){
  CG=(); PID=()
  for s in "${SPECS[@]}"; do
    IFS='|' read -r key ns label <<<"$s"
    local pod cid pid
    pod=$(kubectl get pod -n "$ns" -l "$label" --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    [ -z "$pod" ] && { log "WARN $key: no pod"; continue; }
    cid=$(kubectl get pod "$pod" -n "$ns" -o jsonpath='{.status.containerStatuses[0].containerID}' 2>/dev/null); cid=${cid##*://}
    pid=$(sudo k3s crictl inspect --output go-template --template '{{.info.pid}}' "$cid" 2>/dev/null)
    [ -z "$pid" ] && { log "WARN $key: no pid"; continue; }
    CG[$key]=$(sudo cat /proc/$pid/cgroup | sed 's/^0:://;s|^/||'); PID[$key]=$pid
  done
  log "pods resolved: ${!CG[*]}"
}
PODORDER=(vllm fastapi milvus mongodb seaweed_filer seaweed_volume)

# ---- cleanup / traps --------------------------------------------------------------------------
POLL_PIDS=(); REC_PIDS=(); RAN_WORK=0
cleanup(){
  local rc=$?
  if [ "$RAN_WORK" = 1 ]; then
    [ ${#REC_PIDS[@]} -gt 0 ] && kill -TERM "${REC_PIDS[@]}" 2>/dev/null
    [ ${#POLL_PIDS[@]} -gt 0 ] && kill "${POLL_PIDS[@]}" 2>/dev/null
    kubectl scale deploy/loadgen -n llm-service --replicas=0 2>/dev/null
  fi
  [ -f "$STATE/iso_applied" ] && restore_isolation
  log "cleanup done (exit $rc)"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

# ---- isolation (k3s-aware: pin pods IN, not kill) ---------------------------------------------
pin_exception_pods(){ # kube-system + loadgen pod slices -> house (they live under kubepods.slice)
  local ns pod uid qos slice
  for spec in "kube-system|" "llm-service|app=loadgen"; do
    IFS='|' read -r ns label <<<"$spec"
    for pod in $(kubectl get pods -n "$ns" ${label:+-l "$label"} -o jsonpath='{.items[*].metadata.name}' 2>/dev/null); do
      uid=$(kubectl get pod "$pod" -n "$ns" -o jsonpath='{.metadata.uid}' | tr '-' '_')
      qos=$(kubectl get pod "$pod" -n "$ns" -o jsonpath='{.status.qosClass}' | tr 'A-Z' 'a-z')
      case "$qos" in
        guaranteed) slice="kubepods-pod${uid}.slice" ;;
        *)          slice="kubepods-${qos}-pod${uid}.slice" ;;
      esac
      sudo systemctl set-property --runtime "$slice" AllowedCPUs="$CPUS_HOUSE" 2>/dev/null \
        && log "  exception pod $ns/$pod -> house" || log "  WARN could not pin $ns/$pod ($slice)"
    done
  done
}

pin_workload_pods(){ # each fenced pod's OWN slice -> measured. The pod slice is the parent
  # of the cri scope (kubepods.slice/kubepods-<qos>.slice/kubepods-<qos>-podUID.slice/cri-*);
  # path element 2 is the shared QoS slice — pinning THAT dragged coredns onto measured (caught live).
  local key slice
  for key in "${PODORDER[@]}"; do
    [ -z "${CG[$key]:-}" ] && continue
    slice=$(basename "$(dirname "${CG[$key]}")")
    sudo systemctl set-property --runtime "$slice" AllowedCPUs="$CPUS_MEASURED" 2>/dev/null       && log "  workload pod $key -> measured" || log "  WARN pin failed: $key ($slice)"
  done
}

apply_isolation(){
  if [ -f "$STATE/iso_applied" ]; then
    log "ISOLATION: iso_applied present — keeping baseline snapshot"
  else
    log "ISOLATION: snapshotting baseline"
    cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor > "$STATE/governor"
    cat /sys/devices/system/cpu/intel_pstate/no_turbo          > "$STATE/no_turbo"
    grep -o '\[.*\]' /sys/kernel/mm/transparent_hugepage/enabled | tr -d '[]' > "$STATE/thp"
    grep -o '\[.*\]' /sys/kernel/mm/transparent_hugepage/defrag  | tr -d '[]' > "$STATE/thp_defrag"
    cat /proc/sys/kernel/nmi_watchdog                           > "$STATE/nmi"
    cat /proc/irq/default_smp_affinity                          > "$STATE/irq_default"
    for f in /proc/irq/*/smp_affinity; do
      local n=${f#/proc/irq/}; n=${n%%/*}
      cat "$f" > "$STATE/irq_$n" 2>/dev/null || true
    done
    touch "$STATE/iso_applied"
  fi
  log "ISOLATION: applying (k3s-aware: pods -> measured, control plane -> house)"
  echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null
  echo 1           | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo         >/dev/null
  echo never       | sudo tee /sys/kernel/mm/transparent_hugepage/enabled           >/dev/null
  echo never       | sudo tee /sys/kernel/mm/transparent_hugepage/defrag            >/dev/null
  echo 0           | sudo tee /proc/sys/kernel/nmi_watchdog                         >/dev/null
  echo "$HOUSE_IRQ_MASK" | sudo tee /proc/irq/default_smp_affinity >/dev/null
  for f in /proc/irq/*/smp_affinity; do
    echo "$HOUSE_IRQ_MASK" | sudo tee "$f" >/dev/null 2>&1 || true
  done
  sudo systemctl set-property --runtime system.slice  AllowedCPUs="$CPUS_HOUSE"
  sudo systemctl set-property --runtime user.slice    AllowedCPUs="$CPUS_HOUSE"
  # kubepods.slice stays WIDE: per-pod slice pins decide placement. A child slice cannot
  # escape its parent cpuset (empty intersection falls back to parent — verified live:
  # the first isotest caught coredns "pinned" to house yet effective on measured).
  local ALLC; ALLC=$(cat /sys/devices/system/cpu/online)
  sudo systemctl set-property --runtime kubepods.slice AllowedCPUs="$ALLC"
  sudo systemctl set-property --runtime kubepods-burstable.slice  AllowedCPUs="$ALLC" 2>/dev/null
  sudo systemctl set-property --runtime kubepods-besteffort.slice AllowedCPUs="$ALLC" 2>/dev/null
  pin_exception_pods
  resolve_pods
  pin_workload_pods
  sudo pkill -9 -x perf 2>/dev/null
  log "ISOLATION: applied"
}

restore_isolation(){
  log "ISOLATION: restoring"
  [ -f "$STATE/governor" ] && cat "$STATE/governor" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null
  [ -f "$STATE/no_turbo" ] && cat "$STATE/no_turbo" | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo >/dev/null
  [ -f "$STATE/thp" ]        && cat "$STATE/thp"        | sudo tee /sys/kernel/mm/transparent_hugepage/enabled >/dev/null
  [ -f "$STATE/thp_defrag" ] && cat "$STATE/thp_defrag" | sudo tee /sys/kernel/mm/transparent_hugepage/defrag  >/dev/null
  [ -f "$STATE/nmi" ] && cat "$STATE/nmi" | sudo tee /proc/sys/kernel/nmi_watchdog >/dev/null
  [ -f "$STATE/irq_default" ] && cat "$STATE/irq_default" | sudo tee /proc/irq/default_smp_affinity >/dev/null
  local f n
  for f in "$STATE"/irq_[0-9]*; do
    [ -f "$f" ] || continue
    n=${f##*/irq_}
    cat "$f" | sudo tee "/proc/irq/$n/smp_affinity" >/dev/null 2>&1 || true
    rm -f "$f"
  done
  local ALLCPUS; ALLCPUS=$(cat /sys/devices/system/cpu/online)
  sudo systemctl set-property --runtime system.slice   AllowedCPUs="$ALLCPUS"
  sudo systemctl set-property --runtime user.slice     AllowedCPUs="$ALLCPUS"
  sudo systemctl set-property --runtime kubepods.slice AllowedCPUs="$ALLCPUS"
  sudo systemctl set-property --runtime kubepods-burstable.slice  AllowedCPUs="$ALLCPUS" 2>/dev/null
  sudo systemctl set-property --runtime kubepods-besteffort.slice AllowedCPUs="$ALLCPUS" 2>/dev/null
  local ps
  for ps in $(systemctl list-units --type=slice --no-legend 'kubepods-*pod*' 2>/dev/null | awk '{print $1}'); do
    sudo systemctl set-property --runtime "$ps" AllowedCPUs="$ALLCPUS" 2>/dev/null
  done
  rm -f "$STATE/iso_applied"
  log "ISOLATION: restored"
}

# ---- capture stack (per-pod fences; identical mechanics to the agent kit) ---------------------
start_pollers(){ # $1 out
  local OUT="$1"; local i=0
  touch "$OUT/.polling"
  for key in "${PODORDER[@]}"; do
    [ -z "${CG[$key]:-}" ] && continue
    i=$((i+1))
    ( exec taskset -c "$CPUS_HOUSE" bash -c '
        while [ -f "'"$OUT"'/.polling" ]; do
          u=$(head -3 "/sys/fs/cgroup/'"${CG[$key]}"'/cpu.stat" 2>/dev/null | tr "\n" " ")
          echo "$EPOCHREALTIME ${u:-usage_usec -1}"
          sleep 0.1
        done >> "'"$OUT/cpustat_${key}.tsv"'" ' ) &
    POLL_PIDS+=($!)
  done
}
stop_pollers(){ rm -f "$1/.polling"; sleep 0.3; POLL_PIDS=(); }

start_records(){ # $1 out — full-cell records on every fenced pod
  local OUT="$1"
  for key in "${PODORDER[@]}"; do
    [ -z "${CG[$key]:-}" ] && continue
    taskset -c "$CPUS_HOUSE" sudo "$PERF" record -e task-clock -a --cgroup="${CG[$key]}" -g -F 99 \
      -o "$OUT/rec_${key}.data" >/dev/null 2>&1 &
    REC_PIDS+=($!)
  done
}
stop_records(){ # $1 out
  local OUT="$1"
  [ ${#REC_PIDS[@]} -gt 0 ] && sudo kill -TERM $(jobs -p 2>/dev/null) 2>/dev/null
  [ ${#REC_PIDS[@]} -gt 0 ] && { sudo pkill -TERM -x perf 2>/dev/null; wait "${REC_PIDS[@]}" 2>/dev/null; }
  REC_PIDS=()
  for key in "${PODORDER[@]}"; do
    [ -f "$OUT/rec_${key}.data" ] || continue
    sudo "$PERF" script -f -i "$OUT/rec_${key}.data" -F comm,period,ip,sym,dso 2>/dev/null | awk '
      /^\t/ { if (want) { dso=$0; sub(/.*\(/,"",dso); sub(/\).*/,"",dso); d[dso]+=per
                          if (dso=="[kernel.kallsyms]") { k[$2]+=per }
                          want=0 } next }
      NF>=2 { comm=$1; per=$NF; c[comm]+=per; T+=per; want=1 }
      END { for (x in c) printf "C %.2f%% %s\n", 100*c[x]/T, x
            for (x in d) printf "D %.2f%% %s\n", 100*d[x]/T, x
            for (x in k) printf "K %.2f%% %s\n", 100*k[x]/T, x }' > "$OUT/${key}.tab"
    grep '^C' "$OUT/${key}.tab" | cut -d' ' -f2- | sort -rn > "$OUT/${key}_comm.txt"
    grep '^D' "$OUT/${key}.tab" | cut -d' ' -f2- | sort -rn > "$OUT/${key}_dso.txt"
    grep '^K' "$OUT/${key}.tab" | cut -d' ' -f2- | sort -rn | head -40 > "$OUT/${key}_ksym.txt"
    rm -f "$OUT/${key}.tab"
  done
}

VLLM_IP=""
vllm_metrics(){
  [ -z "$VLLM_IP" ] && VLLM_IP=$(kubectl get pod -n llm-d-local -l llm-d.ai/role=decode \
    -o jsonpath='{.items[0].status.podIP}' 2>/dev/null)
  curl -s --max-time 2 "http://$VLLM_IP:8000/metrics" 2>/dev/null
}
engine_running(){ # instantaneous in-flight requests (vLLM gauge; log tail was unreliable)
  vllm_metrics | awk '/^vllm:num_requests_running/{n=int($2)} END{print n+0}'
}
engine_tokens(){ # monotonic prompt+gen token counter — a per-window DELTA cannot miss work
  vllm_metrics | awk '/^vllm:(prompt_tokens_total|generation_tokens_total)/{s+=$2} END{printf "%.0f\n", s+0}'
}

cell_capture(){ # $1 out — CYCLES full passes over all groups, all pods same window
  local OUT="$1" w=0 c g CGS=""
  for key in "${PODORDER[@]}"; do [ -n "${CG[$key]:-}" ] && CGS="$CGS,${CG[$key]}"; done
  CGS=${CGS#,}
  printf 'win\tgroup\tt_start\tt_end\tengine_running\tengine_tokens\n' > "$OUT/windows.tsv"
  for c in $(seq 1 "$CYCLES"); do
    for g in $GORDER; do
      local ws=$EPOCHREALTIME
      taskset -c "$CPUS_HOUSE" sudo "$PERF" stat -a -e "${GRP[$g]}" --for-each-cgroup="$CGS" \
        -- sleep "$WINSEC" 2> "$OUT/group_${g}_w$(printf '%03d' $w).txt"
      printf '%03d\t%s\t%s\t%s\t%s\t%s\n' "$w" "$g" "$ws" "$EPOCHREALTIME" "$(engine_running)" "$(engine_tokens)" >> "$OUT/windows.tsv"
      w=$((w+1))
    done
  done
  echo "$w" > "$OUT/n_windows"
  sudo chown -R "$USER:$USER" "$OUT" 2>/dev/null
}

# ---- cell = bucket x tier x repeat ------------------------------------------------------------
run_cell(){ # $1 bucket, $2 tier, $3 repeat
  local B="$1" T="$2" N="$3"
  local OUT="$DATA/${TIER_PREFIX}_${B}_tok${T}/run_${N}"
  [ -f "$OUT/DONE" ] && { log "skip ${B}/tok${T} run$N (DONE)"; return 0; }
  mkdir -p "$OUT"; rm -rf "$OUT"/*
  log "================ cell bucket=$B tier=tok$T repeat=$N ================"
  sed -e "s/__MAX_TOKENS__/$T/" -e "s/__BUCKET__/$B/g" "$KIT/loadgen-cell.yaml" | kubectl apply -f - >/dev/null
  kubectl scale deploy/loadgen -n llm-service --replicas=1 >/dev/null
  kubectl rollout status deploy/loadgen -n llm-service --timeout=120s >/dev/null || { log "ERROR loadgen rollout"; return 1; }
  pin_exception_pods          # the fresh loadgen pod must go to house cpus
  local i r=0
  for i in $(seq 1 40); do r=$(engine_running); [ "${r:-0}" -ge 1 ] && break; sleep 5; done
  [ "${r:-0}" -ge 1 ] || { log "ERROR: engine never busy for $B/tok$T"; return 1; }
  log "WORK VERIFIED $B/tok$T r$N (Running: $r); warmup ${WARMUP_S}s"; sleep "$WARMUP_S"
  resolve_pods
  pin_workload_pods
  python3 - "$OUT" "$B" "$T" "$N" <<PY
import json, os, subprocess, sys, time
out, b, t, n = sys.argv[1:5]
rev = subprocess.run(["git","-C","$REPO","rev-parse","--short","HEAD"],capture_output=True,text=True).stdout.strip()
json.dump({"workload":"service_rag","bucket":b,"tier":int(t),"run":int(n),
  "model":"qwen2.5-7b-instruct-awq (local vLLM)","cpus_measured":"$CPUS_MEASURED",
  "cpus_house":"$CPUS_HOUSE","winsec":$WINSEC,"cycles":$CYCLES,"repo_rev":rev,
  "kernel":os.uname().release,"ts_start":time.time(),
  "pods":{$(for key in "${PODORDER[@]}"; do [ -n "${CG[$key]:-}" ] && printf '"%s":"%s",' "$key" "${CG[$key]}"; done | sed 's/,$//')}},
  open(f"{out}/metadata.json","w"), indent=1)
PY
  start_pollers "$OUT"; start_records "$OUT"
  cell_capture "$OUT"
  stop_records "$OUT"; stop_pollers "$OUT"
  kubectl logs -n llm-service deploy/loadgen --tail=8 > "$OUT/loadgen_tail.txt" 2>/dev/null
  kubectl scale deploy/loadgen -n llm-service --replicas=0 >/dev/null 2>&1
  local nw=$(cat "$OUT/n_windows" 2>/dev/null || echo 0)
  local busy=$(awk -F'\t' 'NR>1{ if($5>=1 || (prev!="" && $6>prev)) c++; prev=$6 } END{print c+0}' "$OUT/windows.tsv")
  if [ "$nw" -ge $((CYCLES*9)) ] && [ "$busy" -ge $((nw-2)) ]; then
    log "CELL-OK $B/tok$T r$N (windows=$nw busy=$busy)"; touch "$OUT/DONE"
  else
    log "CELL-FAIL $B/tok$T r$N (windows=$nw busy-windows=$busy)"
  fi
}

# ---- stages ------------------------------------------------------------------------------------
stage_preflight(){
  log "PREFLIGHT"
  local fail=0
  [ -x "$PERF" ] || { log "FAIL perf"; fail=1; }
  [ "$(cat /proc/sys/kernel/perf_event_paranoid)" = "-1" ] || { log "FAIL paranoid"; fail=1; }
  systemctl is-active k3s >/dev/null || { log "FAIL k3s not active"; fail=1; }
  kubectl get nodes >/dev/null 2>&1 || { log "FAIL kubectl"; fail=1; }
  resolve_pods
  [ -n "${CG[vllm]:-}" ] && [ -n "${CG[fastapi]:-}" ] || { log "FAIL core pods unresolved"; fail=1; }
  [ -d /sys/fs/cgroup/kubepods.slice ] || { log "FAIL kubepods.slice missing (cgroup driver?)"; fail=1; }
  for b in $BUCKETS; do
    [ -f "$REPO/benchmark_queries/rag/$b.txt" ] || { log "FAIL bucket file $b.txt"; fail=1; }
  done
  pgrep -x perf >/dev/null && { log "FAIL stale perf"; fail=1; }
  sudo -n true 2>/dev/null || { log "FAIL passwordless sudo"; fail=1; }
  [ $fail -eq 0 ] && log "PREFLIGHT OK" || { log "PREFLIGHT FAILED"; exit 1; }
}

stage_dryrun(){  # zero-multiplexing gate + kernel-split calibration, same as the agent kit
  log "DRYRUN: 9 groups vs busy dummy scope"
  systemd-run --user --scope --unit="svc-dryrun-$$" --collect -- python3 -c "
import numpy as np, time  # svc_dryrun_marker
a=np.random.rand(600,600); x=0; t=time.time()
while time.time()-t<180:
    a@a
    for i in range(200000): x = x+1 if (x*2654435761)&0x10000 else x-1
" >/dev/null 2>&1 &
  sleep 2
  local BP=$(pgrep -f svc_dryrun_marker | head -1)
  [ -n "$BP" ] || { log "DRYRUN FAIL: dummy did not start"; exit 1; }
  local CGd=$(sed 's/^0:://' /proc/$BP/cgroup | head -1 | sed 's|^/||') fail=0
  for g in $GORDER; do
    "$PERF" stat -a -e "${GRP[$g]}" --for-each-cgroup="$CGd" -- sleep 3 2> "$STATE/dry_$g.txt"
    if grep -qE "<not counted>|<not supported>|Bad event|unknown" "$STATE/dry_$g.txt" \
       || grep -qE '\(\s*[0-9]+[.,][0-9]+%\s*\)\s*$' "$STATE/dry_$g.txt"; then
      log "DRYRUN FAIL: group $g"; fail=1
    else log "  $g OK"; fi
  done
  kill "$BP" 2>/dev/null
  [ $fail -eq 0 ] && { log "DRYRUN OK"; touch "$STATE/dryrun_ok"; } || exit 1
}

stage_isolation_test(){
  apply_isolation
  log "ISOTEST: verifying live"
  local fail=0
  [ "$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)" = performance ] || { log "FAIL governor"; fail=1; }
  [ "$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)" = 1 ] || { log "FAIL no_turbo"; fail=1; }
  local KP=$(cat /sys/fs/cgroup/kubepods.slice/cpuset.cpus.effective)
  [ "$KP" = "$(cat /sys/devices/system/cpu/online)" ] || { log "FAIL kubepods not wide ($KP)"; fail=1; }
  local SY=$(cat /sys/fs/cgroup/system.slice/cpuset.cpus.effective)
  [ "$SY" = "$CPUS_HOUSE" ] || { log "FAIL system.slice cpuset ($SY)"; fail=1; }
  resolve_pods
  for key in vllm fastapi; do
    [ -z "${CG[$key]:-}" ] && continue
    local EFF=$(cat "/sys/fs/cgroup/${CG[$key]}/cpuset.cpus.effective" 2>/dev/null)
    [ "$EFF" = "$CPUS_MEASURED" ] && log "  $key on measured ($EFF)" || { log "FAIL $key cpuset ($EFF)"; fail=1; }
  done
  # coredns must be on HOUSE (exception pin worked)
  local CDP=$(kubectl get pod -n kube-system -l k8s-app=kube-dns -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
  if [ -n "$CDP" ]; then
    local cid=$(kubectl get pod "$CDP" -n kube-system -o jsonpath='{.status.containerStatuses[0].containerID}'); cid=${cid##*://}
    local cpid=$(sudo k3s crictl inspect --output go-template --template '{{.info.pid}}' "$cid" 2>/dev/null)
    local ccg=$(sudo cat /proc/$cpid/cgroup | sed 's/^0:://;s|^/||')
    local CEFF=$(cat "/sys/fs/cgroup/$ccg/cpuset.cpus.effective" 2>/dev/null)
    [ "$CEFF" = "$CPUS_HOUSE" ] && log "  coredns on house ($CEFF)" || { log "FAIL coredns cpuset ($CEFF)"; fail=1; }
  fi
  restore_isolation
  [ $fail -eq 0 ] && log "ISOTEST OK (applied + verified + reverted)" || { log "ISOTEST FAILED"; exit 1; }
}

stage_campaign(){
  [ -f "$STATE/dryrun_ok" ] || { [ "${FORCE:-0}" = 1 ] || { log "run dryrun first (or FORCE=1)"; exit 1; }; }
  RAN_WORK=1
  apply_isolation
  for B in $BUCKETS; do
    for T in $TIERS; do
      for N in $(seq 1 "$REPEATS"); do run_cell "$B" "$T" "$N"; done
    done
  done
  restore_isolation
  log "CAMPAIGN COMPLETE — grid $(echo $BUCKETS | wc -w)x$(echo $TIERS | wc -w)x$REPEATS"
}

case "${1:-all}" in
  preflight)      stage_preflight ;;
  dryrun)         stage_preflight; stage_dryrun ;;
  isolation-test) stage_isolation_test ;;
  campaign)       stage_campaign ;;
  all)            stage_preflight; stage_dryrun; stage_campaign ;;
  *) echo "usage: $0 {preflight|dryrun|isolation-test|campaign|all}"; exit 1 ;;
esac
