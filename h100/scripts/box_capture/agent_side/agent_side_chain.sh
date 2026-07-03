#!/usr/bin/env bash
# agent_side_chain.sh — H100 box: the AGENT-SIDE campaign the 32B runs missed, mirroring the
# validated local chains (local_agents/scripts/{bcb,oc,swe}_live_two_view.sh) on bare vLLM.
# Per workload, THREE scopes in the same windows:
#   engine (vLLM scope) | agent harness (driver scope) | tool container
# Captures: parallel task-clock records + portable counter groups stats-first (core, cache,
# mlp, fp1, fp2 at ${GRP_SEC}s each; NO tma groups — KVM guest has no topdown events) +
# nvidia-smi GPU timeline (2 Hz) from work-guard to agent exit.
# Engine must already be serving (serve_h100.sh, coder-32b for swe/bcb, instruct for oc);
# start it under:  systemd-run --user --scope --unit=vllm-serve ./serve_h100.sh
# Usage: ./agent_side_chain.sh swe|bcb|oc-calendar|oc-web|oc-pdf|oc-crop
# Output: ~/agent_side_data/<workload>/  (rsync back into repo h100/data_agent_side/)
set -o pipefail
WORK="${1:?workload}"
OUT_ROOT="$HOME/agent_side_data"
GRP_SEC="${GRP_SEC:-10}"
REC_SEC="${REC_SEC:-30}"
PERF="${PERF:-perf}"
log(){ printf '[agent-side] %s\n' "$*"; }

running_reqs(){ curl -s --max-time 2 localhost:8000/metrics | grep -E '^vllm:num_requests_running' | awk '{print int($2)}' | head -1; }
engine_guard(){
  for i in $(seq 1 120); do
    r=$(running_reqs); [ "${r:-0}" -ge 1 ] && { log "WORK VERIFIED (running=$r)"; return 0; }
    kill -0 "$1" 2>/dev/null || { log "ERROR: agent died before engine busy"; return 1; }
    sleep 2
  done
  log "ERROR: engine never busy"; return 1
}
cg_of(){ sed 's/^0:://' "/proc/$1/cgroup" | head -1 | sed 's|^/||'; }
gpu_sample(){ while kill -0 "$1" 2>/dev/null; do
    printf "%s,%s\n" "$(date +%s.%N)" "$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    sleep 0.5
  done >> "$2"; }

declare -A GRP
GRP[core]="task-clock,cycles,instructions,branches,branch-misses"
GRP[cache]="cycles,instructions,mem_load_retired.l1_hit,mem_load_retired.l2_hit,mem_load_retired.l3_hit,mem_load_retired.l3_miss"
GRP[mlp]="cycles,instructions,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles,uops_executed.thread"
GRP[fp1]="cycles,instructions,fp_arith_inst_retired.scalar_single,fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.128b_packed_single,fp_arith_inst_retired.128b_packed_double"
GRP[fp2]="cycles,instructions,fp_arith_inst_retired.256b_packed_single,fp_arith_inst_retired.256b_packed_double,fp_arith_inst_retired.512b_packed_single,fp_arith_inst_retired.512b_packed_double"

# resolve engine cgroup from the live vLLM EngineCore (NOT the api server: the worker burns the CPU)
ENG_PID=$(pgrep -f "VLLM::EngineCore" | head -1); [ -z "$ENG_PID" ] && ENG_PID=$(pgrep -f "vllm serve" | head -1)
[ -n "$ENG_PID" ] || { log "ERROR: no vLLM process (start serve_h100.sh first)"; exit 1; }
ENG_CG=$(cg_of "$ENG_PID")
log "engine pid=$ENG_PID cgroup=$ENG_CG"

# ---- launch the workload in its own scope ----
OUT="$OUT_ROOT/$WORK"; mkdir -p "$OUT"
UNIT="agent-$WORK-$$"
case "$WORK" in
  bcb)
    systemd-run --user --scope --unit="$UNIT" --collect -- bash -c \
      "cd ~/bcb && VLLM=http://localhost:8000/v1 MODEL=coder-32b ./.venv/bin/python3 agentic_bcb.py 12 3" \
      > "$OUT/agent.log" 2>&1 & ;;
  swe|swe-scikit|swe-sympy)
    case "$WORK" in
      swe)        INSTANCE="astropy__astropy-14096" ;;
      swe-scikit) INSTANCE="scikit-learn__scikit-learn-25232" ;;
      swe-sympy)  INSTANCE="sympy__sympy-14248" ;;
    esac
    rm -rf ~/swe/runs/live_32b
    systemd-run --user --scope --unit="$UNIT" --collect -- bash -c \
      "cd ~/swe && source .venv/bin/activate && \
       export HOSTED_VLLM_API_BASE=http://localhost:8000/v1 HOSTED_VLLM_API_KEY=dummy OPENAI_API_KEY=dummy && \
       sweagent run-batch --config config/fc_local.yaml \
         --instances.type swe_bench --instances.subset verified --instances.split test \
         --instances.filter $INSTANCE \
         --agent.model.name hosted_vllm/coder-32b \
         --agent.model.api_base http://localhost:8000/v1 --agent.model.api_key dummy \
         --agent.model.per_instance_cost_limit 0 --agent.model.total_cost_limit 0 \
         --agent.model.max_input_tokens 14000 --agent.model.max_output_tokens 2048 \
         --agent.model.temperature 0.4 \
         --agent.model.completion_kwargs '{\"tool_choice\":\"required\",\"frequency_penalty\":0.5,\"presence_penalty\":0.3}' \
         --agent.tools.execution_timeout 120 --agent.tools.max_consecutive_execution_timeouts 6 \
         --num_workers 1 --output_dir runs/live_32b" > "$OUT/agent.log" 2>&1 & ;;
  oc-*)
    declare -A OCTASK
    OCTASK[oc-calendar]="tasks/01_Productivity_Flow/01_Productivity_Flow_task_6_calendar_scheduling.md"
    OCTASK[oc-web]="tasks/01_Productivity_Flow/01_Productivity_Flow_task_1_arxiv_digest.md"
    OCTASK[oc-pdf]="tasks/01_Productivity_Flow/01_Productivity_Flow_task_10_pdf_digest.md"
    OCTASK[oc-crop]="tasks/05_Creative_Synthesis/05_Creative_Synthesis_task_10_social_poster_multi_crop.md"
    systemd-run --user --scope --unit="$UNIT" --collect -- bash -c \
      "cd ~/WildClawBench && source .venv/bin/activate && \
       python3 eval/run_batch.py --task '${OCTASK[$WORK]}' --models-config my_api.json \
         --model my-openai-proxy/instruct-32b --parallel 1" > "$OUT/agent.log" 2>&1 & ;;
  *) log "unknown workload $WORK"; exit 1 ;;
esac
sleep 3
AG=$(pgrep -f "agentic_bcb.py|sweagent run-batch|eval/run_batch.py" | head -1)
[ -n "$AG" ] || { log "ERROR: agent did not start"; tail -8 "$OUT/agent.log"; exit 1; }
DRV_CG=$(cg_of "$AG")
log "agent pid=$AG scope=$DRV_CG"

# tool container (swe sandbox / oc task container); bcb has no separate tool container
SB_CG=""; SB_PID=""
case "$WORK" in
  swe*) PAT="sweb.eval" ;;
  oc-*) PAT="wildclawbench" ;;
  *)    PAT="" ;;
esac
if [ -n "$PAT" ]; then
  for i in $(seq 1 200); do
    SB=$(docker ps --format '{{.ID}} {{.Image}} {{.Names}}' | grep -i "$PAT" | awk '{print $1}' | head -1)
    [ -n "$SB" ] && break; kill -0 $AG 2>/dev/null || break; sleep 1
  done
  [ -n "$SB" ] || { log "ERROR: no tool container"; exit 1; }
  SB_FULL=$(docker inspect -f '{{.Id}}' "$SB"); SB_PID=$(docker inspect -f '{{.State.Pid}}' "$SB")
  SB_CG="system.slice/docker-${SB_FULL}.scope"
  log "tool container=$SB"
fi

engine_guard $AG || { kill -9 $AG 2>/dev/null; exit 1; }
G=$(date +%s.%N); echo "guard,$G" > "$OUT/gpu_timeline.csv"
gpu_sample $AG "$OUT/gpu_timeline.csv" & GS=$!

CGS="$ENG_CG,$DRV_CG"; [ -n "$SB_CG" ] && CGS="$CGS,$SB_CG"
log "records + stats (parallel), cgroups: $CGS"
( for g in core cache mlp fp1 fp2; do
    a=1; kill -0 $AG 2>/dev/null || a=0
    echo "$g agent_alive=$a" >> "$OUT/stat_groups_alive.txt"
    sudo "$PERF" stat -a -e "${GRP[$g]}" --for-each-cgroup="$CGS" -- sleep "$GRP_SEC" 2> "$OUT/group_${g}.txt"
  done ) & STATS=$!
sudo "$PERF" record -e task-clock -a --cgroup="$ENG_CG" -g -F 199 -o "$OUT/rec_engine.data" -- sleep "$REC_SEC" > /dev/null 2>&1 &
P1=$!
sudo "$PERF" record -e task-clock -a --cgroup="$DRV_CG" -g -F 199 -o "$OUT/rec_driver.data" -- sleep "$REC_SEC" > /dev/null 2>&1 &
P2=$!
if [ -n "$SB_CG" ]; then
  sudo "$PERF" record -e task-clock -a --cgroup="$SB_CG" -g -F 199 -o "$OUT/rec_tool.data" -- sleep "$REC_SEC" > /dev/null 2>&1
fi
wait $P1 $P2 $STATS

# reports (tool container symbols via symfs)
sudo "$PERF" report -i "$OUT/rec_engine.data" --stdio --sort=dso 2>/dev/null | grep -E "^\s+[0-9]" > "$OUT/engine_dso.txt" || true
sudo "$PERF" report -i "$OUT/rec_driver.data" --stdio --sort=dso 2>/dev/null | grep -E "^\s+[0-9]" > "$OUT/driver_dso.txt" || true
sudo "$PERF" report -i "$OUT/rec_driver.data" --stdio -g none 2>/dev/null | grep -E "^\s+[0-9]" > "$OUT/driver_flat.txt" || true
if [ -n "$SB_CG" ]; then
  sudo "$PERF" report -i "$OUT/rec_tool.data" --stdio --sort=dso 2>/dev/null | grep -E "^\s+[0-9]" > "$OUT/tool_dso.txt" || true
  sudo "$PERF" report -i "$OUT/rec_tool.data" --stdio -g none --symfs="/proc/${SB_PID}/root" 2>/dev/null | grep -E "^\s+[0-9]" > "$OUT/tool_flat.txt" || true
fi

# bounded natural end, then validation
s=0; while kill -0 $AG 2>/dev/null && [ $s -lt 1800 ]; do sleep 10; s=$((s+10)); done
kill -9 $AG 2>/dev/null; wait $AG 2>/dev/null; wait $GS 2>/dev/null
sudo chown -R "$USER:$USER" "$OUT" 2>/dev/null
ok=$(grep -c "agent_alive=1" "$OUT/stat_groups_alive.txt" 2>/dev/null || echo 0)
se=$(stat -c%s "$OUT/rec_engine.data" 2>/dev/null || echo 0)
sd=$(stat -c%s "$OUT/rec_driver.data" 2>/dev/null || echo 0)
gl=$(grep -c "" "$OUT/gpu_timeline.csv" 2>/dev/null || echo 0)
if [ "$se" -gt 100000 ] && [ "$sd" -gt 50000 ] && [ "$ok" -ge 3 ]; then
  log "VALIDATE-OK $WORK (eng=${se}B drv=${sd}B groups-in-window=$ok/5 gpu-samples=$gl)"
else
  log "VALIDATE-FAIL $WORK (eng=${se}B drv=${sd}B groups-in-window=$ok/5 gpu-samples=$gl)"
fi
log "DONE -> $OUT"
