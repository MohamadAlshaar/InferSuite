#!/usr/bin/env bash
# capture_agents_record.sh — per-task SOFTWARE VIEW of the local engine during agent-trajectory
# replay (perf record task-clock on the vLLM pod cgroup). TMA/counters intentionally NOT captured
# here (already covered by the standalone and service TMA); this adds the missing per-task
# during-inference attribution for the main text. Engine = the running local k3s vLLM (7B-AWQ).
set -o pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export KUBECONFIG="$HOME/.kube/k3s-local.yaml"
PERF=$(ls -d /usr/lib/linux-tools-6.8*/perf 2>/dev/null | tail -1)  # newest 6.8-series tools (6.17 wrapper broken)
OUT_ROOT="$REPO/local_agents/data"
REC_SEC="${REC_SEC:-25}"
WARMUP_S="${WARMUP_S:-30}"
log(){ printf '[agents-rec] %s\n' "$*"; }

declare -A TRAJ
TRAJ[astropy]="$REPO/agentic/swe_agent/runs/api/astropy__astropy-14096/astropy__astropy-14096/astropy__astropy-14096.traj"
TRAJ[scikit-learn]="$REPO/agentic/swe_agent/runs/api/scikit-learn__scikit-learn-25232/scikit-learn__scikit-learn-25232/scikit-learn__scikit-learn-25232.traj"
TRAJ[sympy]="$REPO/agentic/swe_agent/runs/api/sympy__sympy-14248/sympy__sympy-14248/sympy__sympy-14248.traj"

# resolve engine pod cgroup
pod=$(kubectl get pod -n llm-d-local -l llm-d.ai/role=decode --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')
cid=$(kubectl get pod "$pod" -n llm-d-local -o jsonpath='{.status.containerStatuses[0].containerID}'); cid=${cid##*://}
pid=$(sudo k3s crictl inspect --output go-template --template '{{.info.pid}}' "$cid")
cg=$(sudo cat /proc/$pid/cgroup | sed 's/^0:://'); cg=${cg#/}
log "engine pod=$pod pid=$pid"
[ -n "$cg" ] || { log "ERROR: no cgroup"; exit 1; }

sudo pkill -9 -x perf 2>/dev/null; sleep 2

for task in astropy scikit-learn sympy; do
  OUT="$OUT_ROOT/$task"; mkdir -p "$OUT"
  log "================ $task ================"
  python3 "$REPO/local_agents/scripts/traj_replay_local.py" "${TRAJ[$task]}" > "$OUT/replay.log" 2>&1 &
  RPID=$!
  # wait until the engine reports running requests
  busy=0
  for i in $(seq 1 30); do
    r=$(kubectl logs -n llm-d-local deploy/ms-local-llm-d-modelservice-decode --tail=5 2>/dev/null \
        | grep -oE "Running: [0-9]+" | tail -1 | grep -oE "[0-9]+" || echo 0)
    [ "${r:-0}" -ge 1 ] && { busy=1; log "engine busy (Running: $r)"; break; }
    sleep 4
  done
  [ "$busy" = 1 ] || log "WARN: engine never busy for $task"
  log "warmup ${WARMUP_S}s"; sleep "$WARMUP_S"
  log "perf record task-clock ${REC_SEC}s"
  sudo "$PERF" record -e task-clock -a --cgroup="$cg" -g -F 199 -o "$OUT/rec_engine.data" -- sleep "$REC_SEC" > "$OUT/perf_record.err" 2>&1
  log "perf stat: 7 groups x 20s (5 portable + TMA L1 + td2)"
  declare -A GRP
  GRP[core]="task-clock,cycles,instructions,branches,branch-misses"
  GRP[fp1]="cycles,instructions,fp_arith_inst_retired.scalar_single,fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.128b_packed_single,fp_arith_inst_retired.128b_packed_double"
  GRP[fp2]="cycles,instructions,fp_arith_inst_retired.256b_packed_single,fp_arith_inst_retired.256b_packed_double,fp_arith_inst_retired.512b_packed_single,fp_arith_inst_retired.512b_packed_double"
  GRP[cache]="cycles,instructions,mem_load_retired.l1_hit,mem_load_retired.l2_hit,mem_load_retired.l3_hit,mem_load_retired.l3_miss"
  GRP[mlp]="cycles,instructions,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles,uops_executed.thread"
  GRP[tma1]="slots,topdown-retiring,topdown-bad-spec,topdown-fe-bound,topdown-be-bound"
  GRP[tma2]="slots,topdown-heavy-ops,topdown-br-mispredict,topdown-fetch-lat,topdown-mem-bound"
  for g in core fp1 fp2 cache mlp tma1 tma2; do
    sudo "$PERF" stat -a -e "${GRP[$g]}" --for-each-cgroup="$cg" -- sleep 20 2> "$OUT/group_engine_${g}.txt"
  done
  sudo "$PERF" report -i "$OUT/rec_engine.data" --stdio -g none --symfs="/proc/$pid/root" 2>/dev/null \
    | grep -E "^\s+[0-9]" > "$OUT/engine_flat.txt" || true
  sudo "$PERF" report -i "$OUT/rec_engine.data" --stdio --sort=dso 2>/dev/null \
    | grep -E "^\s+[0-9]" > "$OUT/engine_dso.txt" || true
  kubectl logs -n llm-d-local deploy/ms-local-llm-d-modelservice-decode --tail=40 2>/dev/null \
    | grep -E "Running|Waiting" | tail -12 > "$OUT/vllm_status_tail.txt" || true
  tail -4 "$OUT/replay.log" > "$OUT/replay_tail.txt" 2>/dev/null || true
  kill $RPID 2>/dev/null; wait $RPID 2>/dev/null
  # validation
  sz=$(stat -c%s "$OUT/rec_engine.data" 2>/dev/null || echo 0)
  lines=$(wc -l < "$OUT/engine_flat.txt" 2>/dev/null || echo 0)
  topdso=$(head -1 "$OUT/engine_dso.txt" | sed 's/^ *//' | cut -c1-60)
  if [ "$sz" -gt 100000 ] && [ "$lines" -gt 5 ] && grep -qE "Running: [1-9]" "$OUT/vllm_status_tail.txt"; then
    log "VALIDATE-OK $task (rec=${sz}B, ${lines} symbols, load verified) top: $topdso"
  else
    log "VALIDATE-FAIL $task (rec=${sz}B, symbols=$lines)"
  fi
  log "draining 20s"; sleep 20
done
log "DONE -> $OUT_ROOT"
