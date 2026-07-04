#!/usr/bin/env bash
# harness_api_chain.sh — the last empty cell of the matrix: the AGENT SIDE at frontier quality.
# Live Sonnet-driven runs on the workstation (model remote, no local engine), harness/container
# cgroups captured with the FULL suite: TMA L1+L2 first, then core/cache/fp1/fp2/mlp (15 s each,
# Sonnet episodes run minutes so all windows land in-episode) + 30 s task-clock records.
#   SWE x3  : sweagent (June api flags: default.yaml, top_p null, $4 cap) — harness scope + sandbox
#   BCB     : agentic_bcb_claude.py HEAVY_LIBS subset — driver scope (tests inside, as elsewhere)
#   OC x4   : litellm proxy + run_batch claude-sonnet-4-6 — task container (agent+tools)
# Keys: agentic/swe_agent/.env, agentic/bigcodebench/.env, ~/.anthropic_key (never echoed).
set -o pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PERF=$(ls -d /usr/lib/linux-tools-6.8*/perf 2>/dev/null | tail -1)
DATA="$REPO/local_agents/data"
MODEL_SWE="anthropic/claude-sonnet-4-6"
log(){ printf '[api-harness] %s\n' "$*"; }

cleanup(){
  pkill -f "sweagent run-batch" 2>/dev/null; pkill -f "agentic_bcb_claude" 2>/dev/null
  pkill -f "eval/run_batch.py" 2>/dev/null; pkill -f "litellm --config" 2>/dev/null
  docker ps -aq --filter ancestor=wildclawbench-ubuntu:v1.3 | xargs -r docker rm -f >/dev/null 2>&1
}
trap cleanup EXIT
sudo pkill -9 -x perf 2>/dev/null; sleep 1

declare -A GRP
GRP[tma1]="slots,topdown-retiring,topdown-bad-spec,topdown-fe-bound,topdown-be-bound"
GRP[tma2]="slots,topdown-heavy-ops,topdown-br-mispredict,topdown-fetch-lat,topdown-mem-bound"
GRP[core]="task-clock,cycles,instructions,branches,branch-misses"
GRP[cache]="cycles,instructions,mem_load_retired.l1_hit,mem_load_retired.l2_hit,mem_load_retired.l3_hit,mem_load_retired.l3_miss"
GRP[fp1]="cycles,instructions,fp_arith_inst_retired.scalar_single,fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.128b_packed_single,fp_arith_inst_retired.128b_packed_double"
GRP[fp2]="cycles,instructions,fp_arith_inst_retired.256b_packed_single,fp_arith_inst_retired.256b_packed_double,fp_arith_inst_retired.512b_packed_single,fp_arith_inst_retired.512b_packed_double"
GRP[mlp]="cycles,instructions,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles,uops_executed.thread"

capture(){  # $1 out, $2 cgroups(csv), $3 agent pid, $4 name  — stats-first + parallel records
  local OUT="$1" CGS="$2" AG="$3" NAME="$4"
  log "capture $NAME: 7 groups x 15s + records 30s (cgroups: ${CGS//,/ , })"
  ( for g in tma1 tma2 core cache fp1 fp2 mlp; do
      local a=1; kill -0 "$AG" 2>/dev/null || a=0
      echo "$g agent_alive=$a" >> "$OUT/stat_groups_alive.txt"
      sudo "$PERF" stat -a -e "${GRP[$g]}" --for-each-cgroup="$CGS" -- sleep 15 2> "$OUT/group_${g}.txt"
    done ) & local STATS=$!
  local i=0 RP=()
  IFS=',' read -ra CGA <<< "$CGS"
  for cg in "${CGA[@]}"; do
    i=$((i+1))
    sudo "$PERF" record -e task-clock -a --cgroup="$cg" -g -F 199 -o "$OUT/rec_scope${i}.data" -- sleep 30 > /dev/null 2>&1 &
    RP+=($!)
  done
  wait "${RP[@]}" $STATS
  i=0
  for cg in "${CGA[@]}"; do
    i=$((i+1))
    sudo "$PERF" report -i "$OUT/rec_scope${i}.data" --stdio --sort=dso 2>/dev/null | grep -E "^\s+[0-9]" > "$OUT/scope${i}_dso.txt" || true
    sudo "$PERF" report -i "$OUT/rec_scope${i}.data" --stdio -g none 2>/dev/null | grep -E "^\s+[0-9]" > "$OUT/scope${i}_flat.txt" || true
  done
  sudo chown -R "$USER:$USER" "$OUT" 2>/dev/null
}

validate(){  # $1 out, $2 name
  local ok=$(grep -c "agent_alive=1" "$1/stat_groups_alive.txt" 2>/dev/null || echo 0)
  local sz=$(stat -c%s "$1/rec_scope1.data" 2>/dev/null || echo 0)
  if [ "$ok" -ge 6 ] && [ "$sz" -gt 50000 ]; then log "VALIDATE-OK $2 (groups-in-window $ok/7, rec=${sz}B)"
  else log "VALIDATE-FAIL $2 (groups-in-window $ok/7, rec=${sz}B)"; fi
}

if [ -z "${ONLY_OC:-}" ]; then
# ================= SWE x3 (Sonnet, June flags) =================
cd "$REPO/agentic/swe_agent"; source .venv/bin/activate
set -a; . ./.env; set +a
[ -n "${ANTHROPIC_API_KEY:-}" ] || { log "FATAL: no key in swe .env"; exit 1; }
for INST in astropy__astropy-14096 scikit-learn__scikit-learn-25232 sympy__sympy-14248; do
  SHORT=${INST%%__*}; OUT="$DATA/api_${SHORT}"; mkdir -p "$OUT"; rm -f "$OUT"/*
  log "================ swe $SHORT (Sonnet) ================"
  rm -rf "runs/api_live/$INST"
  systemd-run --user --scope --unit="swe-api-$$" --collect -- bash -c \
    "cd '$REPO/agentic/swe_agent' && source .venv/bin/activate && set -a && . ./.env && set +a && \
     sweagent run-batch --config external/SWE-agent/config/default.yaml \
       --instances.type swe_bench --instances.subset verified --instances.split test \
       --instances.filter '$INST' \
       --agent.model.name '$MODEL_SWE' --agent.model.top_p null \
       --agent.model.per_instance_cost_limit 4.0 --agent.model.total_cost_limit 30.0 \
       --num_workers 1 --output_dir 'runs/api_live/$INST'" > "$OUT/agent.log" 2>&1 &
  sleep 3
  AG=$(pgrep -f "sweagent run-batch" | head -1)
  [ -n "$AG" ] || { log "ERROR: sweagent did not start"; tail -5 "$OUT/agent.log"; continue; }
  SCOPE=$(sed 's/^0:://' /proc/$AG/cgroup | head -1 | sed 's|^/||')
  SB=""; for i in $(seq 1 240); do
    SB=$(docker ps --format '{{.ID}} {{.Image}} {{.Names}}' | grep -i "sweb" | awk '{print $1}' | head -1)
    [ -n "$SB" ] && break; kill -0 $AG 2>/dev/null || break; sleep 1
  done
  [ -n "$SB" ] || { log "ERROR: no sandbox for $SHORT"; kill -9 $AG 2>/dev/null; continue; }
  SBF=$(docker inspect -f '{{.Id}}' "$SB")
  for i in $(seq 1 120); do grep -aq "STEP 2" "$OUT/agent.log" && break; kill -0 $AG 2>/dev/null || break; sleep 2; done
  log "WORK VERIFIED $SHORT (steps advancing)"
  capture "$OUT" "$SCOPE,system.slice/docker-${SBF}.scope" $AG "$SHORT"
  s=0; while kill -0 $AG 2>/dev/null && [ $s -lt 1200 ]; do sleep 10; s=$((s+10)); done
  kill -9 $AG 2>/dev/null; wait $AG 2>/dev/null
  docker ps -aq --filter "ancestor=docker.io/swebench/sweb.eval.x86_64.${SHORT}_1776_${INST##*__}:latest" | xargs -r docker rm -f >/dev/null 2>&1
  docker ps -q --filter "name=sweb" | xargs -r docker rm -f >/dev/null 2>&1
  validate "$OUT" "$SHORT"
done

# ================= BCB (Sonnet, heavy subset) =================
OUT="$DATA/api_bcb"; mkdir -p "$OUT"; rm -f "$OUT"/*
log "================ bcb (Sonnet, HEAVY_LIBS) ================"
rm -f /tmp/bcb_agentic_markers.txt
systemd-run --user --scope --unit="bcb-api-$$" --collect -- bash -c \
  "cd '$REPO/agentic/bigcodebench' && source .venv/bin/activate && set -a && . ./.env && set +a && \
   HEAVY_LIBS=1 MODEL=claude-sonnet-4-6 OUTDIR=runs/api_live python3 agentic_bcb_claude.py 12 3" \
  > "$OUT/agent.log" 2>&1 &
sleep 3
AG=$(pgrep -f "agentic_bcb_claude" | head -1)
if [ -n "$AG" ]; then
  SCOPE=$(sed 's/^0:://' /proc/$AG/cgroup | head -1 | sed 's|^/||')
  for i in $(seq 1 90); do
    mk=$(grep -c toolexec /tmp/bcb_agentic_markers.txt 2>/dev/null || echo 0)
    [ "$mk" -ge 2 ] && { log "WORK VERIFIED bcb ($mk exec markers)"; break; }
    kill -0 $AG 2>/dev/null || break; sleep 2
  done
  capture "$OUT" "$SCOPE" $AG "bcb"
  s=0; while kill -0 $AG 2>/dev/null && [ $s -lt 1800 ]; do sleep 15; s=$((s+15)); done
  kill -9 $AG 2>/dev/null; wait $AG 2>/dev/null
  cp /tmp/bcb_agentic_markers.txt "$OUT/markers.txt" 2>/dev/null
  validate "$OUT" "bcb"
else
  log "ERROR: bcb claude driver did not start"; tail -5 "$OUT/agent.log"
fi

fi
# ================= OC x4 (Sonnet via litellm proxy) =================
KEYFILE="$HOME/.anthropic_key"
[ -s "$KEYFILE" ] || { log "FATAL: $KEYFILE missing"; exit 1; }
export ANTHROPIC_API_KEY="$(tr -d '[:space:]' < "$KEYFILE")"
cd "$REPO/agentic/openclaw"
./.venv_litellm/bin/litellm --config litellm_config.yaml --port 8000 > /tmp/litellm_api.log 2>&1 & PROXY=$!
for i in $(seq 1 30); do curl -sf localhost:8000/health/liveliness >/dev/null 2>&1 && break; sleep 2; done
curl -sf localhost:8000/health/liveliness >/dev/null || { log "ERROR: proxy did not start"; exit 1; }
log "litellm proxy up"
cd "$REPO/agentic/openclaw/external/WildClawBench"; . .venv/bin/activate
declare -A OCT
OCT[calendar]="tasks/01_Productivity_Flow/01_Productivity_Flow_task_6_calendar_scheduling.md"
OCT[web-digest]="tasks/01_Productivity_Flow/01_Productivity_Flow_task_1_arxiv_digest.md"
OCT[pdf-digest]="tasks/01_Productivity_Flow/01_Productivity_Flow_task_10_pdf_digest.md"
OCT[image-crop]="tasks/05_Creative_Synthesis/05_Creative_Synthesis_task_10_social_poster_multi_crop.md"
for T in calendar web-digest pdf-digest image-crop; do
  OUT="$DATA/api_oc_${T}"; mkdir -p "$OUT"; rm -f "$OUT"/*
  log "================ oc $T (Sonnet) ================"
  docker ps -aq --filter ancestor=wildclawbench-ubuntu:v1.3 | xargs -r docker rm -f >/dev/null 2>&1
  python3 eval/run_batch.py --task "${OCT[$T]}" --models-config my_api.json \
    --model my-openai-proxy/claude-sonnet-4-6 --parallel 1 </dev/null > "$OUT/agent.log" 2>&1 &
  AG=$!
  CID=""; for i in $(seq 1 150); do CID=$(docker ps -q --filter ancestor=wildclawbench-ubuntu:v1.3 | head -1); [ -n "$CID" ] && break; kill -0 $AG 2>/dev/null || break; sleep 2; done
  [ -n "$CID" ] || { log "ERROR: no container for $T"; kill -9 $AG 2>/dev/null; wait $AG 2>/dev/null; continue; }
  FULL=$(docker inspect -f '{{.Id}}' "$CID")
  for i in $(seq 1 300); do grep -q "Waiting for agent to finish" "$OUT/agent.log" 2>/dev/null && break; kill -0 $AG 2>/dev/null || break; sleep 1; done
  log "WORK VERIFIED $T (agent running)"
  capture "$OUT" "system.slice/docker-${FULL}.scope" $AG "oc-$T"
  s=0; while kill -0 $AG 2>/dev/null && [ $s -lt 900 ]; do sleep 10; s=$((s+10)); done
  kill -9 $AG 2>/dev/null; wait $AG 2>/dev/null
  docker ps -aq --filter ancestor=wildclawbench-ubuntu:v1.3 | xargs -r docker rm -f >/dev/null 2>&1
  validate "$OUT" "oc-$T"
done
kill $PROXY 2>/dev/null
log "API-HARNESS-DONE"
