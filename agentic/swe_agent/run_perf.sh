#!/usr/bin/env bash
# Launch a SWE-agent instance AND attach interval perf to its sandbox container
# cgroup -> a timestamped per-phase counter timeline (inference-wait vs tool/pytest).
# cgroup-scoped: captures every process in the sandbox, survives command churn.
# perf binary: the kernel-6.8 build works cross-kernel here; paranoid=-1 => no sudo.
set -uo pipefail
cd "$(dirname "$0")"
# Resolve a WORKING perf binary + the correct TMA group via the shared lib, which VERIFIES perf
# actually counts (not just exits 0). Fixes: bare 'perf' wrapper refuses to run on OEM/HWE kernels
# whose linux-tools build mismatches; and the old slots-vs-legacy guess could pick a form whose
# events don't exist on this CPU (-> all-zero passes).
. "$(dirname "$0")/../common/perf_events.sh"
. "$(dirname "$0")/../common/lib_perf.sh"
PERF="$(perf_bin)" || { echo "[perf-run] FATAL: no working perf binary (set PERF_HOST_BIN)"; exit 1; }
EVENTS="$(tma_group)"
echo "[perf-run] perf=$PERF"
echo "[perf-run] TMA events=$EVENTS"
OUT=runs/perf; mkdir -p "$OUT"
N="${1:-1}"

echo "[perf-run] launching agent (${RUN_SCRIPT:-run_local.sh})..."
bash "${RUN_SCRIPT:-run_local.sh}" "$N" > "$OUT/agent.log" 2>&1 &
AGENT=$!

echo "[perf-run] waiting for swebench sandbox container..."
CID=""
for i in $(seq 1 150); do
  CID=$(docker ps --format '{{.ID}} {{.Names}}' | grep -i sweb | awk '{print $1}' | head -1)
  [[ -n "$CID" ]] && break
  kill -0 "$AGENT" 2>/dev/null || { echo "[perf-run] agent exited before sandbox appeared"; break; }
  sleep 2
done
[[ -z "$CID" ]] && { echo "[perf-run] no sandbox container; aborting"; wait "$AGENT"; exit 1; }

FULL=$(docker inspect -f '{{.Id}}' "$CID")
CG="system.slice/docker-${FULL}.scope"
echo "[perf-run] sandbox cid=$CID"
echo "[perf-run] cgroup=$CG"
echo "$(date +%s.%N) perf_start cgroup=$CG cid=$CID" > "$OUT/markers.txt"

# interval perf on the sandbox cgroup -> timestamped CSV (CPU/tool side)
"$PERF" stat -e "$EVENTS" -G "$CG" -a -I 1000 -x, -o "$OUT/perf_timeline.csv" &
PERFPID=$!

# parallel GPU sampler (epoch,util%,mem MiB) -> GPU/inference side. vLLM is the only
# GPU user during the run, so GPU-util = inference activity. Aligns with perf via the
# perf_start epoch in markers.txt. Together: CPU(tool) vs GPU(inference) time split.
( while sleep 1; do
    printf "%s,%s\n" "$(date +%s.%N)" \
      "$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  done ) > "$OUT/gpu_timeline.csv" 2>/dev/null &
GPUPID=$!

# 4th probe: vLLM inference-SERVER CPU (cores) — the serving-side CPU (tokenize/schedule/
# sample/detok + enforce-eager engine loop) that the other probes miss. epoch,cores CSV.
/usr/bin/python3 vllm_cpu_sampler.py "$OUT/vllm_timeline.csv" &
VLLMPID=$!
echo "[perf-run] vLLM-server CPU probe pid=$VLLMPID"

# vLLM-ENGINE perf-TMA (the DURING-inference CPU) — same EVENTS as the agent so we can
# compare Agent-CPU (tool-exec, outside inference) vs Inference-CPU (vLLM busy-wait). This is
# the piece no cloud Nitro/T4 box could give; needs the host PMU + vLLM on the same machine.
VLLMPERFPID=""
VPIDS=$(pgrep -f "VLLM::EngineCore|EngineCore|vllm.*serve|vllm.entrypoints|from multiprocessing" | paste -sd, -)
if [[ -n "$VPIDS" ]]; then
  echo "$(date +%s.%N) vllm_perf_start pids=$VPIDS" >> "$OUT/markers.txt"
  "$PERF" stat -p "$VPIDS" -e "$EVENTS" -I 1000 -x, -o "$OUT/vllm_perf_timeline.csv" 2>/dev/null &
  VLLMPERFPID=$!
  echo "[perf-run] vLLM-engine perf-TMA pids=$VPIDS"
else
  echo "[perf-run] WARN: no vLLM pids for engine perf-TMA"
fi

# 3rd probe: perf on the sweagent HOST controller process (the "agent brain" CPU).
# This is what falls in the grey zone -> splits grey into host-agent-CPU vs true idle.
HOSTPID=""
APID=$(pgrep -f "sweagent run-batch" | head -1)
if [[ -n "$APID" ]]; then
  echo "[perf-run] host-agent perf on sweagent pid=$APID"
  echo "$(date +%s.%N) host_perf_start pid=$APID" >> "$OUT/markers.txt"
  "$PERF" stat -p "$APID" -e "$EVENTS" -I 1000 -x, -o "$OUT/host_agent_timeline.csv" &
  HOSTPID=$!
else
  echo "[perf-run] WARN: could not find sweagent pid for host probe"
fi

wait "$AGENT"
echo "$(date +%s.%N) agent_done" >> "$OUT/markers.txt"
kill -INT "$PERFPID" ${HOSTPID:+"$HOSTPID"} ${VLLMPERFPID:+"$VLLMPERFPID"} 2>/dev/null; sleep 1; kill "$PERFPID" "$GPUPID" ${HOSTPID:+"$HOSTPID"} "$VLLMPID" ${VLLMPERFPID:+"$VLLMPERFPID"} 2>/dev/null
echo "[perf-run] done -> $OUT/perf_timeline.csv, $OUT/agent.log, trajectory in runs/smoke"
