#!/usr/bin/env bash
# Statistical spin-vs-work profiler for the vLLM EngineCore during inference.
# WHY: perf record is broken on this kernel/perf mismatch, and the validation found the
# EngineCore's high IPC/retiring is largely a CUDA-sync SPIN (cudaEventSynchronize), NOT
# compute. This gdb-samples the hottest EngineCore thread N times and classifies each
# backtrace -> spin% vs real-work%. Run DURING sustained decode (GPU busy).
#
# Usage: vllm_spin_sample.sh [N_samples=40] [interval_s=0.25]
set -uo pipefail
N="${1:-40}"; INT="${2:-0.25}"
EPID=$(ps -eo pid,comm | awk '$2 ~ /EngineCor/ {print $1; exit}')
[ -z "${EPID:-}" ] && { echo "FATAL: no VLLM::EngineCore process found (is vLLM up + serving?)" >&2; exit 1; }
command -v gdb >/dev/null || { echo "FATAL: gdb not installed" >&2; exit 1; }
SUDO=""; [ "$(id -u)" != 0 ] && SUDO="sudo -n"
spin=0; work=0; other=0
echo "sampling EngineCore pid=$EPID  n=$N  interval=${INT}s ..."
for i in $(seq 1 "$N"); do
  HOT=$(top -bH -n1 -p "$EPID" 2>/dev/null | tail -n +7 | sort -k9 -nr | awk 'NR==1{print $1}')
  [ -z "${HOT:-}" ] && { sleep "$INT"; continue; }
  bt=$($SUDO timeout 8 gdb -p "$HOT" -batch -ex bt 2>/dev/null | grep -E '^#' || true)
  if echo "$bt" | grep -qiE "cuEventSynchronize|cudaEventSynchronize|cuStreamSynchronize|cudaStreamSynchronize|cuCtxSynchronize|cuEventQuery"; then
    spin=$((spin+1))
  elif echo "$bt" | grep -qiE "torch|vllm|sample|token|detok|schedul|EvalFrame"; then
    work=$((work+1))
  else
    other=$((other+1))
  fi
  sleep "$INT"
done
tot=$((spin+work+other)); [ "$tot" = 0 ] && tot=1
echo "=== vLLM EngineCore spin-vs-work (n=$tot) ==="
printf "  GPU-sync SPIN  = %3d (%d%%)   <- cudaEventSynchronize etc.\n" "$spin" $((spin*100/tot))
printf "  real WORK      = %3d (%d%%)   <- torch/sample/detok/schedule\n" "$work" $((work*100/tot))
printf "  other/unclass  = %3d (%d%%)\n" "$other" $((other*100/tot))
