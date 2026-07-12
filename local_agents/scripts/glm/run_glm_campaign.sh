#!/usr/bin/env bash
# run_glm_campaign.sh — one-command GLM frontier-tier agent campaign (SWE-agent + OpenClaw).
#
#   ./run_glm_campaign.sh [stage]
#     preflight       fail-fast environment checks (no spend, no state change)
#     dryrun          all 8 counter groups vs a busy dummy scope: zero-multiplexing gate
#     isolation-test  apply full isolation, verify every knob, revert, verify reverted
#     smoke           proxy-path checks (chat + tool-call through litellm; ~2 requests)
#     smoke-swe       ONE full SWE episode (astropy)  — counts toward the campaign
#     smoke-django    ONE full django-10097 episode   — episode-length check
#     smoke-oc        ONE full OC episode (calendar)
#     campaign swe    SWE phase (4 instances x REPEATS)   [review, then:]
#     campaign oc     OC phase  (4 tasks x REPEATS)
#     validate        3-layer validator over all glm_* data
#     all             preflight -> dryrun -> smoke -> campaign swe (stops for review)
#
# Contract: cgroups-not-PIDs; same-window --for-each-cgroup; whole-episode group cycling;
# zero multiplexing; isolation restored by trap on ANY exit (incl. Ctrl-C); resume via DONE
# markers; provenance metadata.json per run. Adversarially reviewed 2026-07-08 (23 findings
# fixed — see plan file).
set -o pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$KIT/../../.." && pwd)"
source "$KIT/campaign.conf"
DATA="${DATA_ROOT:-$REPO/local_agents/data}"   # override for side campaigns (e.g. SWE_long)
STATE="$KIT/.state"; mkdir -p "$STATE"
PERF=$(ls -d /usr/lib/linux-tools-6.8*/perf 2>/dev/null | tail -1)
MSLICE="measured.slice"
log(){ printf '[glm %s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$KIT/campaign.log"; }

# ---------------- counter groups (dry-run-verified 2026-07-08, zero multiplexing) -----------
declare -A GRP
GRP[tma]="slots,topdown-retiring,topdown-bad-spec,topdown-fe-bound,topdown-be-bound,topdown-heavy-ops,topdown-br-mispredict,topdown-fetch-lat,topdown-mem-bound"
GRP[core]="task-clock,cycles,instructions,branches,branch-misses"
GRP[cache]="cycles,instructions,mem_load_retired.l1_hit,mem_load_retired.l2_hit,mem_load_retired.l3_hit,mem_load_retired.l3_miss"
GRP[fp1]="cycles,instructions,fp_arith_inst_retired.scalar_single,fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.128b_packed_single,fp_arith_inst_retired.128b_packed_double"
GRP[fp2]="cycles,instructions,fp_arith_inst_retired.256b_packed_single,fp_arith_inst_retired.256b_packed_double,fp_arith_inst_retired.512b_packed_single,fp_arith_inst_retired.512b_packed_double"
GRP[mlp]="cycles,instructions,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles,uops_executed.thread"
GRP[fe]="cycles,instructions,idq.dsb_uops,idq.mite_uops,idq.ms_uops,lsd.uops"
GRP[icache]="cycles,instructions,l2_rqsts.all_code_rd,l2_rqsts.code_rd_miss,icache_data.stalls"
# priv: kernel-vs-user split + kernel-mediated behavior. context-switches/migrations/
# page-faults are software events (zero PMU counters); :u/:k splits cost 2 GP.
GRP[priv]="task-clock,context-switches,cpu-migrations,page-faults,cycles:u,cycles:k,instructions:u,instructions:k"
GORDER="tma core cache fp1 fp2 mlp fe icache priv"

cg_of(){ sed 's/^0:://' "/proc/$1/cgroup" 2>/dev/null | head -1 | sed 's|^/||'; }
cg_of_container(){ local p; p=$(docker inspect -f '{{.State.Pid}}' "$1" 2>/dev/null); [ -n "$p" ] && cg_of "$p"; }

# ---------------- cleanup trap (EXIT covers normal paths; INT/TERM routed into it) -----------
PROXY_UNIT=""; WATCHER_PID=""; POLL_PIDS=(); REC_PIDS=(); AGENT_PID=""; LG_PID=""; RAN_WORK=0
cleanup(){
  local rc=$?
  if [ "$RAN_WORK" = 1 ]; then
    sudo systemctl stop 'glm-swe-*.scope' 2>/dev/null
    [ -n "$AGENT_PID" ] && kill -9 -- "-$AGENT_PID" 2>/dev/null   # OC process group (setsid)
    pkill -f "eval/run_batch.py" 2>/dev/null
    [ ${#REC_PIDS[@]} -gt 0 ] && kill -TERM "${REC_PIDS[@]}" 2>/dev/null
    [ ${#POLL_PIDS[@]} -gt 0 ] && kill "${POLL_PIDS[@]}" 2>/dev/null
    [ -n "$LG_PID" ] && kill "$LG_PID" 2>/dev/null
    [ -n "$WATCHER_PID" ] && sudo kill "$WATCHER_PID" 2>/dev/null
    [ -n "$PROXY_UNIT" ] && systemctl --user stop "$PROXY_UNIT.scope" 2>/dev/null
    docker ps -aq --filter ancestor=wildclawbench-ubuntu:v1.3 | xargs -r docker rm -f >/dev/null 2>&1
    docker ps -q --filter "name=sweb" | xargs -r docker rm -f >/dev/null 2>&1
  fi
  [ -f "$STATE/iso_applied" ] && restore_isolation
  log "cleanup done (exit $rc)"
}
trap cleanup EXIT
trap 'exit 130' INT TERM      # Ctrl-C/kill do NOT fire EXIT traps by themselves — route them

# ---------------- isolation ------------------------------------------------------------------
apply_isolation(){
  if [ -f "$STATE/iso_applied" ]; then
    log "ISOLATION: iso_applied present — keeping existing baseline snapshot (crash-rerun safe)"
  else
    log "ISOLATION: snapshotting baseline -> $STATE"
    cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor > "$STATE/governor"
    cat /sys/devices/system/cpu/intel_pstate/no_turbo          > "$STATE/no_turbo"
    grep -o '\[.*\]' /sys/kernel/mm/transparent_hugepage/enabled | tr -d '[]' > "$STATE/thp"
    grep -o '\[.*\]' /sys/kernel/mm/transparent_hugepage/defrag  | tr -d '[]' > "$STATE/thp_defrag"
    cat /proc/sys/kernel/nmi_watchdog                           > "$STATE/nmi"
    systemctl is-active k3s                                     > "$STATE/k3s" 2>/dev/null || true
    cat /proc/irq/default_smp_affinity                          > "$STATE/irq_default"
    for f in /proc/irq/*/smp_affinity; do
      local n=${f#/proc/irq/}; n=${n%%/*}
      cat "$f" > "$STATE/irq_$n" 2>/dev/null || true
    done
    if [ -f /etc/docker/daemon.json ]; then cp /etc/docker/daemon.json "$STATE/daemon.json.orig"
    else rm -f "$STATE/daemon.json.orig"; fi
    touch "$STATE/iso_applied"        # BEFORE mutations: a crash mid-apply must trigger restore
  fi
  log "ISOLATION: applying"
  echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null
  echo 1           | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo         >/dev/null
  echo never       | sudo tee /sys/kernel/mm/transparent_hugepage/enabled           >/dev/null
  echo never       | sudo tee /sys/kernel/mm/transparent_hugepage/defrag            >/dev/null
  echo 0           | sudo tee /proc/sys/kernel/nmi_watchdog                         >/dev/null
  echo "$HOUSE_IRQ_MASK" | sudo tee /proc/irq/default_smp_affinity >/dev/null
  for f in /proc/irq/*/smp_affinity; do
    echo "$HOUSE_IRQ_MASK" | sudo tee "$f" >/dev/null 2>&1 || true
  done
  sudo systemctl set-property --runtime "$MSLICE" AllowedCPUs="$CPUS_MEASURED"
  # docker: all containers -> measured.slice (daemon.json swap, checked restart)
  python3 - "$MSLICE" <<'PY'
import json, sys, os
p = "/etc/docker/daemon.json"
d = json.load(open(p)) if os.path.exists(p) else {}
d["cgroup-parent"] = sys.argv[1]
open("/tmp/glm_daemon.json", "w").write(json.dumps(d, indent=2))
PY
  sudo cp /tmp/glm_daemon.json /etc/docker/daemon.json
  sudo systemctl reset-failed docker.service docker.socket 2>/dev/null
  sudo systemctl restart docker || { log "FATAL: docker restart failed"; exit 1; }
  local i; for i in $(seq 1 15); do docker info >/dev/null 2>&1 && break; sleep 2; done
  docker info >/dev/null 2>&1 || { log "FATAL: docker did not come back"; exit 1; }
  sudo systemctl set-property --runtime system.slice AllowedCPUs="$CPUS_HOUSE"
  sudo systemctl set-property --runtime user.slice   AllowedCPUs="$CPUS_HOUSE"
  if [ "${SKIP_K3S:-0}" != 1 ] && grep -q '^active' "$STATE/k3s" 2>/dev/null; then
    sudo systemctl stop k3s
    # 'stop k3s' leaves the PODS alive under kubepods.slice (cpuset 0-23 — OUTSIDE the
    # slice shield; verified live 2026-07-08). killall reaps them; k3s start respawns them.
    sudo /usr/local/bin/k3s-killall.sh >/dev/null 2>&1
    sudo systemctl set-property --runtime kubepods.slice AllowedCPUs="$CPUS_HOUSE" 2>/dev/null
    log "ISOLATION: k3s stopped + pods killed (auto-respawn on restore)"
  fi
  sudo pkill -9 -x perf 2>/dev/null
  log "ISOLATION: applied (measured=$CPUS_MEASURED house=$CPUS_HOUSE)"
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
  # NB: AllowedCPUs="" is NOT honored as a reset, and nproc lies under our own shield
  # (affinity-sensitive) — use the kernel's online range (verified 2026-07-08)
  local ALLCPUS; ALLCPUS=$(cat /sys/devices/system/cpu/online)
  sudo systemctl set-property --runtime system.slice AllowedCPUs="$ALLCPUS"
  sudo systemctl set-property --runtime user.slice   AllowedCPUs="$ALLCPUS"
  sudo systemctl set-property --runtime "$MSLICE"    AllowedCPUs="$ALLCPUS"
  if [ -f "$STATE/daemon.json.orig" ]; then sudo cp "$STATE/daemon.json.orig" /etc/docker/daemon.json
  else sudo rm -f /etc/docker/daemon.json; fi
  sudo systemctl reset-failed docker.service docker.socket 2>/dev/null
  sudo systemctl restart docker || log "WARN: docker restart failed during restore"
  if grep -q '^active' "$STATE/k3s" 2>/dev/null && [ "${SKIP_K3S:-0}" != 1 ]; then
    sudo systemctl start k3s && log "ISOLATION: k3s restarted"
  fi
  rm -f "$STATE/iso_applied"
  log "ISOLATION: restored"
}

# ---------------- capture stack --------------------------------------------------------------
start_pollers(){ # $1 out, $2 cgroups csv — 10 Hz cpu.stat usage_usec per scope
  local OUT="$1"; IFS=',' read -ra CGA <<< "$2"; local i=0
  touch "$OUT/.polling"
  for cg in "${CGA[@]}"; do
    i=$((i+1))
    ( exec taskset -c "$CPUS_HOUSE" bash -c '
        while [ -f "'"$OUT"'/.polling" ]; do
          u=$(head -3 "/sys/fs/cgroup/'"$cg"'/cpu.stat" 2>/dev/null | tr "\n" " ")
          echo "$EPOCHREALTIME ${u:-usage_usec -1}"
          sleep 0.1
        done >> "'"$OUT/cpustat_scope$i.tsv"'" ' ) &
    POLL_PIDS+=($!)
  done
}
stop_pollers(){ rm -f "$1/.polling"; sleep 0.3; POLL_PIDS=(); }

start_records(){ # $1 out, $2 cgroups csv — full-episode task-clock records
  local OUT="$1"; IFS=',' read -ra CGA <<< "$2"; local i=0
  for cg in "${CGA[@]}"; do
    i=$((i+1))
    taskset -c "$CPUS_HOUSE" "$PERF" record -e task-clock -a --cgroup="$cg" -g -F 99 \
      -o "$OUT/rec_scope${i}.data" >/dev/null 2>&1 &
    REC_PIDS+=($!)
  done
}
mk_tables(){ # $1 rec.data, $2 out-prefix, $3 symfs ('-' = none) — perf-script based:
  # tolerates the truncated-tail corruption that aborts perf report when a recorded cgroup
  # is destroyed mid-flight (sweagent removes its sandbox on natural exit; verified 2026-07-08).
  # -f: root reading a user-owned perf.data is refused without it (silently, with 2>/dev/null).
  local REC="$1" PRE="$2" SF="${3:--}" SYMFS=()
  [ "$SF" != "-" ] && [ -d "$SF" ] && SYMFS=(--symfs "$SF")
  sudo "$PERF" script -f -i "$REC" "${SYMFS[@]}" -F comm,period,ip,sym,dso 2>/dev/null | awk '
    /^\t/ { if (want) { dso=$0; sub(/.*\(/,"",dso); sub(/\).*/,"",dso); d[dso]+=per
                        if (dso=="[kernel.kallsyms]") { k[$2]+=per }
                        want=0 } next }
    NF>=2 { comm=$1; per=$NF; c[comm]+=per; T+=per; want=1 }
    END { for (x in c) printf "C %.2f%% %s\n", 100*c[x]/T, x
          for (x in d) printf "D %.2f%% %s\n", 100*d[x]/T, x
          for (x in k) printf "K %.2f%% %s\n", 100*k[x]/T, x }' > "${PRE}.tab"
  grep '^C' "${PRE}.tab" | cut -d' ' -f2- | sort -rn > "${PRE}_comm.txt"
  # per-sample (pid, time) table: consumed by E4 lineage purity + post-hoc re-attribution
  sudo "$PERF" script -f -i "$REC" -F pid,time 2>/dev/null | \
    awk 'NF>=2 {gsub(/:$/,"",$2); print $1, $2}' > "${PRE}_pidtime.txt"
  grep '^D' "${PRE}.tab" | cut -d' ' -f2- | sort -rn > "${PRE}_dso.txt"
  grep '^K' "${PRE}.tab" | cut -d' ' -f2- | sort -rn | head -40 > "${PRE}_ksym.txt"
  rm -f "${PRE}.tab"
}

stop_records(){ # $1 out, $2 cgroups csv, $3 symfs csv ('-' entries = none)
  local OUT="$1"; IFS=',' read -ra CGA <<< "$2"; IFS=',' read -ra SFA <<< "${3:-}"
  [ ${#REC_PIDS[@]} -gt 0 ] && kill -TERM "${REC_PIDS[@]}" 2>/dev/null
  wait "${REC_PIDS[@]}" 2>/dev/null; REC_PIDS=()
  local i=0
  for cg in "${CGA[@]}"; do
    i=$((i+1))
    mk_tables "$OUT/rec_scope${i}.data" "$OUT/scope${i}" "${SFA[$((i-1))]:--}"
  done
}

# ---------------- OC live loop guard (opt-in, LOOP_GUARD_N>0) --------------------------------
# Tails the episode's host-side chat.jsonl (WildClawBench output dir) and kills the agent
# process group when the last N assistant tool calls are identical (name+args). Same rationale
# as the SWE guard: E7 only catches loops after the tokens are spent.
oc_loop_guard(){ # $1 results-parent-dir (contains <run-stamp>/chat.jsonl), $2 agent pgid
  local DIR="$1" APID="$2" N="${LOOP_GUARD_N:-0}" runlen chat
  [ "$N" -gt 0 ] || return 0
  while kill -0 "$APID" 2>/dev/null; do
    chat=$(ls -td "$DIR"/*/ 2>/dev/null | head -1)chat.jsonl
    if [ -f "$chat" ]; then
      runlen=$(tail -c 400000 "$chat" 2>/dev/null | python3 -c '
import sys, json
sig, cur, prev = [], 1, None
for ln in sys.stdin:
    try: m = json.loads(ln)
    except Exception: continue
    if m.get("type") != "message": continue
    msg = m.get("message") or {}
    if msg.get("role") != "assistant": continue
    for c in (msg.get("content") or []):
        if "tool" in str(c.get("type", "")).lower():
            s = json.dumps({k: c.get(k) for k in ("name", "toolName", "arguments", "input")},
                           sort_keys=True, default=str)
            sig.append(s)
run = 1
for a, b in zip(sig, sig[1:]):
    run = run + 1 if a == b else 1
print(run if sig else 0)' 2>/dev/null)
      if [ "${runlen:-0}" -ge "$N" ]; then
        log "OC LOOP-GUARD tripped: last $runlen tool calls identical — killing agent group"
        kill -9 -- "-$APID" 2>/dev/null
        return 0
      fi
    fi
    sleep 15
  done
}

# ---------------- live loop guard (opt-in, LOOP_GUARD_N>0) -----------------------------------
# Kills a SWE episode when the last N actions in agent.log are identical — post-hoc E7 only
# catches loops after the tokens are spent (2026-07-11: babel looped 654x for ~2.3h into the
# 3h fuse). Compares the first echoed line of each action (rich wraps at ~80 cols): identical
# actions match trivially; two DIFFERENT actions sharing an 80-col prefix could in theory
# false-trip, hence N should stay >=10. Runs on house CPUs; stopping the scope ends the
# episode through the normal ALIVE->teardown path, data stays valid (capped class).
loop_guard(){ # $1 agent.log, $2 scope unit
  local LOG="$1" UNIT="$2" N="${LOOP_GUARD_N:-0}" runlen
  [ "$N" -gt 0 ] || return 0
  while systemctl is-active --quiet "$UNIT.scope" 2>/dev/null; do
    # join the FULL wrapped action block (rich wraps at ~80 cols; first line can be a bare
    # command word like "cat" — comparing it alone false-tripped on 13 different cats,
    # 2026-07-11). Block ends at the first padded-blank line.
    runlen=$(tail -c 300000 "$LOG" 2>/dev/null | \
             awk '/ACTION/{blk=""; while ((getline ln) > 0) { gsub(/[ \t]+$/, "", ln); gsub(/^[ \t]+/, "", ln); if (ln == "") break; blk = blk ln }
                  if (blk != "") { if (blk == prev) c++; else c = 1; prev = blk } } END{print c+0}')
    if [ "${runlen:-0}" -ge "$N" ]; then
      log "LOOP-GUARD tripped: last $runlen actions identical — stopping $UNIT"
      sudo systemctl stop "$UNIT.scope" 2>/dev/null
      return 0
    fi
    sleep 10
  done
}

cycle_stats(){ # $1 out, $2 cgroups csv, $3 alive-predicate (shell string), $4 max seconds
  local OUT="$1" CGS="$2" ALIVE="$3" MAX="$4" t0=$SECONDS w=0 g ws a
  printf 'win\tgroup\tt_start\tt_end\talive_after\n' > "$OUT/windows.tsv"
  while eval "$ALIVE" 2>/dev/null && [ $((SECONDS - t0)) -lt "$MAX" ]; do
    for g in $GORDER; do
      eval "$ALIVE" 2>/dev/null || break
      ws=$EPOCHREALTIME
      taskset -c "$CPUS_HOUSE" "$PERF" stat -a -e "${GRP[$g]}" --for-each-cgroup="$CGS" \
        -- sleep "$WINSEC" 2> "$OUT/group_${g}_w$(printf '%03d' $w).txt"
      a=1; eval "$ALIVE" 2>/dev/null || a=0        # sampled AFTER the window (real signal)
      printf '%03d\t%s\t%s\t%s\t%d\n' "$w" "$g" "$ws" "$EPOCHREALTIME" "$a" >> "$OUT/windows.tsv"
      w=$((w+1))
    done
  done
  echo "$w" > "$OUT/n_windows"
}

write_metadata(){ # $1 out, $2 workload, $3 config-id, $4 run-n, $5 extra-json
  python3 - "$@" <<'PY'
import json, subprocess, sys, time, os
out, wl, cfg, run, extra = sys.argv[1:6]
rev = subprocess.run(["git","-C",os.environ["REPO"],"rev-parse","--short","HEAD"],capture_output=True,text=True).stdout.strip()
json.dump({"workload":wl,"config":cfg,"run":int(run),"model":os.environ["MODEL_ID"],
  "endpoint":os.environ["GLM_ENDPOINT"],"thinking":os.environ["THINKING"],
  "cpus_measured":os.environ["CPUS_MEASURED"],"cpus_house":os.environ["CPUS_HOUSE"],
  "winsec":int(os.environ["WINSEC"]),"repo_rev":rev,"kernel":os.uname().release,
  "ts_start":time.time(),"extra":json.loads(extra or "{}")}, open(f"{out}/metadata.json","w"), indent=1)
PY
}
export REPO MODEL_ID GLM_ENDPOINT THINKING CPUS_MEASURED CPUS_HOUSE WINSEC

episode_ok(){ # $1 out, $2 name
  local nw=$(cat "$1/n_windows" 2>/dev/null || echo 0)
  local sz=$(stat -c%s "$1/rec_scope1.data" 2>/dev/null || echo 0)
  if [ "${nw:-0}" -ge 8 ] && [ "${sz:-0}" -gt 50000 ]; then
    log "EPISODE-OK $2 (windows=$nw rec=${sz}B)"; touch "$1/DONE"; return 0
  fi
  log "EPISODE-FAIL $2 (windows=$nw rec=${sz}B)"; return 1
}

# ---------------- proxy (own scope -> own cgroup, housekeeping CPUs) --------------------------
PROXY_CG=""
start_proxy(){
  [ -s "$KEYFILE" ] || { log "FATAL: $KEYFILE missing"; exit 1; }
  export GLM_API_KEY="$(tr -d '[:space:]' < "$KEYFILE")"
  PROXY_UNIT="glm-proxy-$$"
  systemd-run --user --scope --unit="$PROXY_UNIT" --collect -- \
    taskset -c "$CPUS_HOUSE" "$REPO/agentic/openclaw/.venv_litellm/bin/litellm" \
    --config "$KIT/litellm_glm.yaml" --port "$PROXY_PORT" > "$KIT/proxy.log" 2>&1 &
  local i; for i in $(seq 1 30); do
    curl -sf "localhost:$PROXY_PORT/health/liveliness" >/dev/null 2>&1 && break; sleep 2
  done
  curl -sf "localhost:$PROXY_PORT/health/liveliness" >/dev/null 2>&1 \
    || { log "FATAL: litellm proxy did not start (see $KIT/proxy.log)"; exit 1; }
  local pp=$(pgrep -u "$USER" -f "litellm --config $KIT/litellm_glm.yaml" | head -1)
  PROXY_CG=$(cg_of "$pp")
  log "proxy up (:$PROXY_PORT cgroup=$PROXY_CG)"
}
stop_proxy(){ [ -n "$PROXY_UNIT" ] && systemctl --user stop "$PROXY_UNIT.scope" 2>/dev/null; PROXY_UNIT=""; }

# ---------------- SWE episode ----------------------------------------------------------------
swe_cleanup_sandbox(){ docker ps -q --filter "name=sweb" | xargs -r docker rm -f >/dev/null 2>&1; }

swe_episode(){ # $1 instance, $2 run n
  local INST="$1" N="$2" SHORT="${1%%__*}${SWE_SHORT_SUFFIX:-}"   # suffix: second config of same repo (e.g. django-lite)
  local UNIT="glm-swe-${SHORT}-r${N}"
  local OUT="$DATA/${TIER_PREFIX}_swe_${SHORT}/run_${N}"
  [ -f "$OUT/DONE" ] && { log "skip $SHORT run$N (DONE)"; return 0; }
  mkdir -p "$OUT"; rm -rf "$OUT"/*        # -r: traj/ dir from a failed attempt must go too
  log "================ swe $SHORT run$N ($MODEL_ID) ================"
  local ODIR="runs/${TIER_PREFIX}_live/${INST}_r${N}"
  rm -rf "$REPO/agentic/swe_agent/$ODIR"
  sudo systemctl stop "$UNIT.scope" 2>/dev/null          # active leftover from a prior run
  sudo systemctl reset-failed "$UNIT.scope" 2>/dev/null
  sudo systemd-run --collect --scope --slice="$MSLICE" --unit="$UNIT" -- \
    runuser -u "$USER" -- bash -c "cd '$REPO/agentic/swe_agent' && source .venv/bin/activate && \
      sweagent run-batch --config external/SWE-agent/config/default.yaml \
        --instances.type swe_bench --instances.subset $SWE_SUBSET --instances.split test \
        --instances.filter '$INST' \
        --agent.model.name 'openai/$MODEL_ID' \
        --agent.model.api_base 'http://localhost:$PROXY_PORT/v1' --agent.model.api_key dummy \
        --agent.model.per_instance_cost_limit 0 --agent.model.total_cost_limit 0 \
        --agent.model.temperature $SWE_TEMP \
        --num_workers 1 --output_dir '$ODIR'" > "$OUT/agent.log" 2>&1 &
  # liveness + harness cgroup come from the scope UNIT, never from a pgrep'd PID:
  # the launch chain is sudo->systemd-run->runuser (root-owned) — kill -0 would EPERM-lie,
  # and pgrep|head-1 would select the sudo wrapper in the caller's session cgroup.
  local HARNESS_CG="$MSLICE/$UNIT.scope"
  local ALIVE="systemctl is-active --quiet $UNIT.scope"
  local i; for i in $(seq 1 30); do [ -d "/sys/fs/cgroup/$HARNESS_CG" ] && break; sleep 1; done
  [ -d "/sys/fs/cgroup/$HARNESS_CG" ] || { log "ERROR: harness scope never appeared"; tail -5 "$OUT/agent.log"; return 1; }
  local SB=""; for i in $(seq 1 240); do
    SB=$(docker ps --format '{{.ID}} {{.Names}}' | grep -i sweb | awk '{print $1}' | head -1)
    [ -n "$SB" ] && break; eval "$ALIVE" || break; sleep 1
  done
  [ -n "$SB" ] || { log "ERROR: no sandbox for $SHORT"; sudo systemctl stop "$UNIT.scope" 2>/dev/null; swe_cleanup_sandbox; return 1; }
  local TOOL_CG=$(cg_of_container "$SB")
  local TOOL_SYMFS=$(docker inspect -f '{{.GraphDriver.Data.MergedDir}}' "$SB" 2>/dev/null)
  for i in $(seq 1 180); do grep -aq "STEP 2" "$OUT/agent.log" && break; eval "$ALIVE" || break; sleep 2; done
  grep -aq "STEP 2" "$OUT/agent.log" || { log "ERROR: $SHORT never reached STEP 2"; sudo systemctl stop "$UNIT.scope" 2>/dev/null; swe_cleanup_sandbox; return 1; }
  log "WORK VERIFIED $SHORT run$N (harness=$HARNESS_CG tool=$TOOL_CG)"
  write_metadata "$OUT" swe "$SHORT" "$N" "{\"instance\":\"$INST\",\"subset\":\"$SWE_SUBSET\",\"temperature\":$SWE_TEMP,\"harness_cg\":\"$HARNESS_CG\",\"tool_cg\":\"$TOOL_CG\",\"proxy_cg\":\"$PROXY_CG\"}"
  local CGS="$HARNESS_CG,$TOOL_CG,$PROXY_CG"
  ( loop_guard "$OUT/agent.log" "$UNIT" ) & LG_PID=$!   # inherits house-CPU confinement (user.slice)
  start_pollers "$OUT" "$CGS"; start_records "$OUT" "$CGS"
  cycle_stats "$OUT" "$CGS" "$ALIVE" "$SWE_DRAIN_S"
  kill "$LG_PID" 2>/dev/null
  stop_records "$OUT" "$CGS" "-,${TOOL_SYMFS:--},-"; stop_pollers "$OUT"   # records stop BEFORE any cgroup teardown we control
  sudo systemctl stop "$UNIT.scope" 2>/dev/null
  cp -r "$REPO/agentic/swe_agent/$ODIR" "$OUT/traj" 2>/dev/null
  swe_cleanup_sandbox
  episode_ok "$OUT" "$SHORT-run$N"
}

# ---------------- OC episode -----------------------------------------------------------------
oc_episode(){ # $1 task key, $2 run n
  local T="$1" N="$2"
  declare -A OCT
  OCT[calendar]="tasks/01_Productivity_Flow/01_Productivity_Flow_task_6_calendar_scheduling.md"
  OCT[linkapix]="tasks/02_Code_Intelligence/02_Code_Intelligence_task_9_link_a_pix_color_easy_zh.md"
  OCT[jigsaw-med]="tasks/02_Code_Intelligence/02_Code_Intelligence_task_4_jigsaw_puzzle_medium_zh.md"
  OCT[web-digest]="tasks/01_Productivity_Flow/01_Productivity_Flow_task_1_arxiv_digest.md"
  OCT[pdf-digest]="tasks/01_Productivity_Flow/01_Productivity_Flow_task_10_pdf_digest.md"
  OCT[image-crop]="tasks/05_Creative_Synthesis/05_Creative_Synthesis_task_10_social_poster_multi_crop.md"
  OCT[sam3-debug]="tasks/02_Code_Intelligence/02_Code_Intelligence_task_2_sam3_debug.md"
  OCT[jigsaw-hard]="tasks/02_Code_Intelligence/02_Code_Intelligence_task_5_jigsaw_puzzle_hard_zh.md"
  OCT[connect-dots]="tasks/02_Code_Intelligence/02_Code_Intelligence_task_12_connect_the_dots_hard_zh.md"
  local OUT="$DATA/${TIER_PREFIX}_oc_${T}/run_${N}"
  [ -f "$OUT/DONE" ] && { log "skip oc-$T run$N (DONE)"; return 0; }
  mkdir -p "$OUT"; rm -rf "$OUT"/*
  log "================ oc $T run$N ($MODEL_ID) ================"
  docker ps -aq --filter ancestor=wildclawbench-ubuntu:v1.3 | xargs -r docker rm -f >/dev/null 2>&1
  # setsid + exec chain: AGENT_PID IS run_batch.py's process-group leader, so
  # kill -- -PID reaps the whole tree (bare $! was the wrapper subshell — orphan bug)
  setsid bash -c "cd '$REPO/agentic/openclaw/external/WildClawBench' && . .venv/bin/activate && \
    exec taskset -c '$CPUS_HOUSE' python3 eval/run_batch.py --task '${OCT[$T]}' \
      --models-config '$KIT/my_api_glm.json' --model 'my-openai-proxy/$MODEL_ID' \
      --parallel 1 </dev/null" > "$OUT/agent.log" 2>&1 &
  AGENT_PID=$!
  local ALIVE="kill -0 $AGENT_PID"
  oc_fail(){ kill -9 -- "-$AGENT_PID" 2>/dev/null; wait "$AGENT_PID" 2>/dev/null; AGENT_PID=""
             [ -n "$WATCHER_PID" ] && sudo kill "$WATCHER_PID" 2>/dev/null; WATCHER_PID=""
             docker ps -aq --filter ancestor=wildclawbench-ubuntu:v1.3 | xargs -r docker rm -f >/dev/null 2>&1; }
  local CID=""; local i; for i in $(seq 1 150); do
    CID=$(docker ps -q --filter ancestor=wildclawbench-ubuntu:v1.3 | head -1)
    [ -n "$CID" ] && break; eval "$ALIVE" 2>/dev/null || break; sleep 2
  done
  [ -n "$CID" ] || { log "ERROR: no container for $T"; oc_fail; return 1; }
  local FULL=$(docker inspect -f '{{.Id}}' "$CID")
  local CONT_CG=$(cg_of_container "$CID")
  local CONT_SYMFS=$(docker inspect -f '{{.GraphDriver.Data.MergedDir}}' "$CID" 2>/dev/null)
  if [ "${OC_WATCHER:-lineage}" = "lineage" ]; then
    # rung-2 lineage watcher (netlink fork/exec, name-blind, accepted 2026-07-12):
    # required for node/JS tools; lineage.tsv enables E4 PID-set purity + post-hoc
    # re-attribution of pre-move samples. OC_WATCHER=legacy restores the comm sorter.
    sudo taskset -c "$CPUS_HOUSE" python3 "$KIT/oc_lineage_watcher.py" \
      "/sys/fs/cgroup/$CONT_CG" "$OUT/lineage.tsv" > "$OUT/watcher.log" 2>&1 & WATCHER_PID=$!
  else
    sudo taskset -c "$CPUS_HOUSE" bash "$KIT/oc_cgroup_watcher.sh" "/sys/fs/cgroup/$CONT_CG" \
      > "$OUT/watcher.log" 2>&1 & WATCHER_PID=$!
  fi
  for i in $(seq 1 20); do
    [ -d "/sys/fs/cgroup/$CONT_CG/agent" ] && [ -d "/sys/fs/cgroup/$CONT_CG/toolexec" ] && break; sleep 1
  done
  [ -d "/sys/fs/cgroup/$CONT_CG/toolexec" ] || { log "ERROR: watcher did not create child cgroups for $T"; oc_fail; return 1; }
  local STARTED=0
  for i in $(seq 1 300); do
    grep -q "Waiting for agent to finish" "$OUT/agent.log" 2>/dev/null && { STARTED=1; break; }
    eval "$ALIVE" 2>/dev/null || break; sleep 1
  done
  [ "$STARTED" = 1 ] || { log "ERROR: oc-$T agent never started working"; oc_fail; return 1; }
  log "WORK VERIFIED oc-$T run$N (container=$CONT_CG)"
  write_metadata "$OUT" oc "$T" "$N" "{\"container_cg\":\"$CONT_CG\",\"proxy_cg\":\"$PROXY_CG\",\"watcher\":\"${OC_WATCHER:-lineage}\"}"
  local TP="${OCT[$T]}"; local STEM=$(basename "$TP" .md); local TCAT=$(dirname "$TP"); TCAT=${TCAT#tasks/}
  local RESDIR="$REPO/agentic/openclaw/external/WildClawBench/output/openclaw/$TCAT/$STEM"
  ( oc_loop_guard "$RESDIR" "$AGENT_PID" ) & LG_PID=$!
  local CGS="$CONT_CG/agent,$CONT_CG/toolexec,$PROXY_CG"
  start_pollers "$OUT" "$CGS"; start_records "$OUT" "$CGS"
  cycle_stats "$OUT" "$CGS" "$ALIVE" "$OC_DRAIN_S"
  kill "$LG_PID" 2>/dev/null
  kill -9 -- "-$AGENT_PID" 2>/dev/null; wait "$AGENT_PID" 2>/dev/null; AGENT_PID=""
  # archive the transcript + grader outputs next to the capture (three-way join input)
  local RESRUN=$(ls -td "$RESDIR"/*/ 2>/dev/null | head -1)
  [ -n "$RESRUN" ] && mkdir -p "$OUT/transcript" && \
    cp "${RESRUN}chat.jsonl" "$RESRUN"*.json "$OUT/transcript/" 2>/dev/null
  stop_records "$OUT" "$CGS" "${CONT_SYMFS:--},${CONT_SYMFS:--},-"; stop_pollers "$OUT"
  sudo kill "$WATCHER_PID" 2>/dev/null; WATCHER_PID=""
  docker ps -aq --filter ancestor=wildclawbench-ubuntu:v1.3 | xargs -r docker rm -f >/dev/null 2>&1
  episode_ok "$OUT" "oc-$T-run$N"
}

# ---------------- SWE replay episode (determinism anchor: recorded traj, NO model) ------------
replay_episode(){ # $1 short-name, $2 source live run, $3 dest run n
  local SHORT="$1" SRC="$2" N="$3"
  local TRAJ=$(find "$DATA/${TIER_PREFIX}_swe_${SHORT}/run_${SRC}/traj" -name "*.traj" 2>/dev/null | head -1)
  [ -n "$TRAJ" ] || { log "SKIP replay $SHORT r$N (no traj in live run_$SRC)"; return 0; }
  local OUT="$DATA/${TIER_PREFIX}_replay_swe_${SHORT}/run_${N}"
  [ -f "$OUT/DONE" ] && { log "skip replay $SHORT r$N (DONE)"; return 0; }
  mkdir -p "$OUT"; rm -rf "$OUT"/*
  log "================ replay $SHORT run$N (traj of live run_$SRC, no model) ================"
  local UNIT="glm-rep-${SHORT}-r${N}"
  sudo systemctl stop "$UNIT.scope" 2>/dev/null
  sudo systemctl reset-failed "$UNIT.scope" 2>/dev/null
  docker ps -q --filter "name=sweb" | xargs -r docker rm -f >/dev/null 2>&1
  sudo systemd-run --collect --scope --slice="$MSLICE" --unit="$UNIT" -- \
    runuser -u "$USER" -- bash -c "cd '$REPO/agentic/swe_agent' && source .venv/bin/activate && \
      sweagent run-replay --traj_path '$TRAJ'" > "$OUT/agent.log" 2>&1 &
  local HARNESS_CG="$MSLICE/$UNIT.scope"
  local ALIVE="systemctl is-active --quiet $UNIT.scope"
  local i; for i in $(seq 1 30); do [ -d "/sys/fs/cgroup/$HARNESS_CG" ] && break; sleep 1; done
  [ -d "/sys/fs/cgroup/$HARNESS_CG" ] || { log "ERROR: replay scope never appeared"; tail -3 "$OUT/agent.log"; return 1; }
  local SB=""; for i in $(seq 1 240); do
    SB=$(docker ps --format '{{.ID}} {{.Names}}' | grep -i sweb | awk '{print $1}' | head -1)
    [ -n "$SB" ] && break; eval "$ALIVE" || break; sleep 1
  done
  [ -n "$SB" ] || { log "ERROR: no sandbox for replay $SHORT r$N"; sudo systemctl stop "$UNIT.scope" 2>/dev/null; return 1; }
  local TOOL_CG=$(cg_of_container "$SB")
  local TOOL_SYMFS=$(docker inspect -f '{{.GraphDriver.Data.MergedDir}}' "$SB" 2>/dev/null)
  log "REPLAY RUNNING $SHORT r$N (harness=$HARNESS_CG tool=$TOOL_CG)"
  write_metadata "$OUT" swe_replay "$SHORT" "$N" "{\"source_run\":$SRC,\"traj\":\"$TRAJ\",\"harness_cg\":\"$HARNESS_CG\",\"tool_cg\":\"$TOOL_CG\"}"
  local CGS="$HARNESS_CG,$TOOL_CG"          # no proxy: the model is never called
  start_pollers "$OUT" "$CGS"; start_records "$OUT" "$CGS"
  cycle_stats "$OUT" "$CGS" "$ALIVE" "${REPLAY_DRAIN_S:-1800}"
  stop_records "$OUT" "$CGS" "-,${TOOL_SYMFS:--}"; stop_pollers "$OUT"
  sudo systemctl stop "$UNIT.scope" 2>/dev/null
  swe_cleanup_sandbox
  episode_ok "$OUT" "replay-$SHORT-run$N"
}

stage_replay(){ # SWE determinism anchor: pair-replay each live run + 2 noise replays of run_1's traj
  local CFGS="${1:-astropy scikit-learn sympy django}"
  RAN_WORK=1
  apply_isolation
  for SHORT in $CFGS; do
    for N in 1 2 3; do replay_episode "$SHORT" "$N" "$N"; done   # live run_N <-> replay run_N pairs
    for N in 4 5;   do replay_episode "$SHORT" 1   "$N"; done   # same-traj repeats: pure measurement noise
  done
  restore_isolation
  stage_validate || true
}

# ---------------- stages ---------------------------------------------------------------------
stage_preflight(){
  log "PREFLIGHT"
  local fail=0
  [ -n "$PERF" ] && [ -x "$PERF" ] || { log "FAIL: perf binary"; fail=1; }
  [ "$(cat /proc/sys/kernel/perf_event_paranoid)" = "-1" ] || { log "FAIL: perf_event_paranoid != -1"; fail=1; }
  [ -s "$KEYFILE" ] || { log "FAIL: $KEYFILE missing"; fail=1; }
  local K=$(tr -d '[:space:]' < "$KEYFILE")
  local code=$(curl -s --max-time 20 -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $K" "$GLM_ENDPOINT/models")
  [ "$code" = 200 ] || { log "FAIL: endpoint $GLM_ENDPOINT -> HTTP $code"; fail=1; }
  curl -s --max-time 20 -H "Authorization: Bearer $K" "$GLM_ENDPOINT/models" | grep -q "\"$MODEL_ID\"" \
    || { log "FAIL: $MODEL_ID not in model list"; fail=1; }
  docker info >/dev/null 2>&1 || { log "FAIL: docker"; fail=1; }
  [ -x "$REPO/agentic/openclaw/.venv_litellm/bin/litellm" ] || { log "FAIL: litellm venv"; fail=1; }
  [ -d "$REPO/agentic/swe_agent/.venv" ] || { log "FAIL: swe venv"; fail=1; }
  pgrep -x perf >/dev/null && { log "FAIL: stale perf running"; fail=1; }
  local free=$(df --output=avail -m "$DATA" | tail -1)
  [ "$free" -gt 5000 ] || { log "FAIL: <5G free at $DATA"; fail=1; }
  sudo -n true 2>/dev/null || { log "FAIL: passwordless sudo needed for isolation"; fail=1; }
  if systemctl is-active k3s >/dev/null 2>&1; then
    [ -x /usr/local/bin/k3s-killall.sh ] || { log "FAIL: k3s active but k3s-killall.sh missing (pods would escape the shield)"; fail=1; }
  fi
  [ $fail -eq 0 ] && log "PREFLIGHT OK" || { log "PREFLIGHT FAILED"; exit 1; }
}

stage_dryrun(){
  log "DRYRUN: 8 groups vs busy dummy scope"
  systemd-run --user --scope --unit="glm-dryrun-$$" --collect -- python3 -c "
import numpy as np, time  # glm_dryrun_marker
a=np.random.rand(600,600); x=0; t=time.time()
while time.time()-t<180:
    a@a
    for i in range(200000): x = x+1 if (x*2654435761)&0x10000 else x-1
" >/dev/null 2>&1 &
  sleep 2
  local BP=$(pgrep -f glm_dryrun_marker | head -1)
  [ -n "$BP" ] || { log "DRYRUN FAIL: dummy did not start"; exit 1; }
  local CG=$(cg_of "$BP") fail=0
  for g in $GORDER; do
    "$PERF" stat -a -e "${GRP[$g]}" --for-each-cgroup="$CG" -- sleep 3 2> "$STATE/dry_$g.txt"
    if grep -qE "<not counted>|<not supported>|Bad event|unknown" "$STATE/dry_$g.txt" \
       || grep -qE '\(\s*[0-9]+[.,][0-9]+%\s*\)\s*$' "$STATE/dry_$g.txt"; then
      log "DRYRUN FAIL: group $g (see $STATE/dry_$g.txt)"; fail=1
    else log "  $g OK"; fi
  done
  # kernel-split calibration: prove cycles:u/:k against scopes with KNOWN kernel share
  log "DRYRUN: kernel-split calibration (ground truth)"
  systemd-run --user --scope --unit="glm-dryks-$$" --collect -- python3 -c "
import os, time  # glm_dryks_marker
t = time.time()
while time.time() - t < 60: os.getppid()" >/dev/null 2>&1 &
  sleep 1
  local KP=$(pgrep -f glm_dryks_marker | head -1)
  if [ -n "$KP" ]; then
    local KCG=$(cg_of "$KP")
    # ground truth = cross-subsystem agreement: PMU cycles:u/:k vs the scheduler's own
    # user_usec/system_usec for the SAME scope over the SAME window. (A guessed absolute
    # threshold is not ground truth — a python syscall loop is legitimately ~72% user;
    # measured 28.2/28.0% on consecutive runs, i.e. stable.)
    local B0=$(head -3 "/sys/fs/cgroup/$KCG/cpu.stat" | tr '\n' ' ')
    "$PERF" stat -a -e "${GRP[priv]}" --for-each-cgroup="$CG,$KCG" -- sleep 5 2> "$STATE/dry_kcal.txt"
    local B1=$(head -3 "/sys/fs/cgroup/$KCG/cpu.stat" | tr '\n' ' ')
    if python3 - "$STATE/dry_kcal.txt" "$CG" "$KCG" "$B0" "$B1" <<'PY'
import re, sys
txt = open(sys.argv[1]).read()
def kshare(cg):
    vals = {}
    for ln in txt.splitlines():
        m = re.match(r"^\s+([\d,]+)\s+(cycles:[uk])\s+(\S+)", ln)
        if m and m.group(3) == cg:
            vals[m.group(2)] = float(m.group(1).replace(",", ""))
    tot = vals.get("cycles:u", 0) + vals.get("cycles:k", 0)
    return 100 * vals.get("cycles:k", 0) / tot if tot else -1
comp, sysc = kshare(sys.argv[2]), kshare(sys.argv[3])
b0 = dict(zip(sys.argv[4].split()[::2], map(float, sys.argv[4].split()[1::2])))
b1 = dict(zip(sys.argv[5].split()[::2], map(float, sys.argv[5].split()[1::2])))
du, ds = b1["user_usec"] - b0["user_usec"], b1["system_usec"] - b0["system_usec"]
sched = 100 * ds / (du + ds) if du + ds > 0 else -1
print(f"  compute-scope kernel {comp:.1f}% (expect <5)")
print(f"  syscall-scope: PMU {sysc:.1f}% vs scheduler {sched:.1f}% (agree within 6pp, both >10)")
sys.exit(0 if (0 <= comp < 5 and sysc > 10 and sched > 10 and abs(sysc - sched) < 6) else 1)
PY
    then log "  kernel-split calibration OK (PMU agrees with scheduler accounting)"
    else log "DRYRUN FAIL: kernel-split calibration"; fail=1; fi
    kill "$KP" 2>/dev/null
  else log "DRYRUN WARN: kernel-calibration scope did not start"; fi
  kill "$BP" 2>/dev/null
  [ $fail -eq 0 ] && { log "DRYRUN OK"; touch "$STATE/dryrun_ok"; } || exit 1
}

stage_isolation_test(){
  SKIP_K3S=1
  apply_isolation
  log "ISOTEST: verifying"
  local fail=0
  [ "$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)" = performance ] || { log "ISOTEST FAIL governor"; fail=1; }
  [ "$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)" = 1 ] || { log "ISOTEST FAIL no_turbo"; fail=1; }
  grep -q '\[never\]' /sys/kernel/mm/transparent_hugepage/enabled || { log "ISOTEST FAIL thp"; fail=1; }
  [ "$(cat /proc/sys/kernel/nmi_watchdog)" = 0 ] || { log "ISOTEST FAIL nmi"; fail=1; }
  local EFF=$(docker run --rm busybox cat /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null)
  [ "$EFF" = "$CPUS_MEASURED" ] || { log "ISOTEST FAIL container cpuset ($EFF != $CPUS_MEASURED)"; fail=1; }
  local SYS_EFF=$(cat /sys/fs/cgroup/system.slice/cpuset.cpus.effective)
  [ "$SYS_EFF" = "$CPUS_HOUSE" ] || { log "ISOTEST FAIL system.slice cpuset ($SYS_EFF)"; fail=1; }
  restore_isolation
  SKIP_K3S=0
  local G=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
  [ "$G" = "$(cat "$STATE/governor" 2>/dev/null || echo "$G")" ] || { log "ISOTEST FAIL revert governor"; fail=1; }
  [ $fail -eq 0 ] && log "ISOTEST OK (applied + reverted cleanly)" || { log "ISOTEST FAILED"; exit 1; }
}

stage_smoke(){
  RAN_WORK=1
  start_proxy
  log "SMOKE: chat through proxy"
  local R=$(curl -s --max-time 120 -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: ready\"}],\"max_tokens\":512}" \
    "localhost:$PROXY_PORT/v1/chat/completions")
  echo "$R" | grep -q 'ready' && log "SMOKE chat OK" || { log "SMOKE chat FAIL: ${R:0:300}"; exit 1; }
  # OC uses "api": "openai-completions" (same chat path; z.ai has no /responses endpoint —
  # litellm only bridges Responses for its native adapters, verified 2026-07-08).
  log "SMOKE: tool-call through proxy (fc path for SWE-agent)"
  R=$(curl -s --max-time 120 -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Create file hello.txt containing hi. Use the tool.\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"create_file\",\"description\":\"Create a file\",\"parameters\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"}},\"required\":[\"path\",\"content\"]}}}],\"max_tokens\":1024}" \
    "localhost:$PROXY_PORT/v1/chat/completions")
  echo "$R" | grep -q '"tool_calls"' && log "SMOKE tool-call OK" || { log "SMOKE tool-call FAIL: ${R:0:300}"; exit 1; }
  stop_proxy
  touch "$STATE/smoke_ok"; log "SMOKE OK"
}

stage_campaign(){ # $1 phase: swe|oc
  local PHASE="${1:-swe}"
  [ -f "$STATE/dryrun_ok" ] && [ -f "$STATE/smoke_ok" ] || {
    [ "${FORCE:-0}" = 1 ] || { log "refusing: run dryrun + smoke first (or FORCE=1)"; exit 1; }; }
  RAN_WORK=1
  apply_isolation
  start_proxy
  if [ "$PHASE" = swe ]; then
    for INST in $SWE_INSTANCES; do
      for N in $(seq 1 "$REPEATS"); do swe_episode "$INST" "$N"; done
    done
    log "SWE PHASE COMPLETE — review validator + sanity output, then: $0 campaign oc"
  else
    for T in $OC_TASKS; do
      for N in $(seq 1 "$REPEATS"); do oc_episode "$T" "$N"; done
    done
    log "OC PHASE COMPLETE"
  fi
  stop_proxy
  restore_isolation
  stage_validate || true
}

stage_validate(){ python3 "$KIT/validate_glm_agents.py" "$DATA" "$TIER_PREFIX"; }

case "${1:-all}" in
  preflight)      stage_preflight ;;
  dryrun)         stage_preflight; stage_dryrun ;;
  isolation-test) stage_isolation_test ;;
  smoke)          stage_preflight; stage_smoke ;;
  smoke-swe)      REPEATS=1 SWE_INSTANCES="astropy__astropy-14096" stage_campaign swe ;;
  smoke-django)   REPEATS=1 SWE_INSTANCES="django__django-10097"   stage_campaign swe ;;
  smoke-oc)       REPEATS=1 OC_TASKS="calendar"                    stage_campaign oc ;;
  campaign)       stage_campaign "${2:-swe}" ;;
  replay-anchor)  stage_replay "${2:-}" ;;
  validate)       stage_validate ;;
  all)            stage_preflight; stage_dryrun; stage_smoke; stage_campaign swe ;;
  *) echo "usage: $0 {preflight|dryrun|isolation-test|smoke|smoke-swe|smoke-django|smoke-oc|campaign swe|campaign oc|validate|all}"; exit 1 ;;
esac
