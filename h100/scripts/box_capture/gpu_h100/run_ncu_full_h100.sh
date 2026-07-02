#!/usr/bin/env bash
# H100 capture: warp-state TMA + Speed-of-Light + microarch metrics, dominant kernels (-k), per regime.
# Usage: run_ncu_full_h100.sh <launch_count> [regime ...]   (default: 60 prefill decode normal)
set -u
cd "$HOME/gpu_h100"
LC="${1:-60}"; shift || true
REGIMES=("$@"); [ ${#REGIMES[@]} -eq 0 ] && REGIMES=(prefill decode normal)
pkill -9 -f run_regime_h100.py 2>/dev/null; pkill -9 -x ncu 2>/dev/null; sleep 3
PY="$HOME/vllmenv/bin/python3"
# bf16 (unquantized) uses cuBLAS/cutlass GEMM (xmma/sm90/nvjet/cutlass) instead of Marlin; broad net:
KREGEX="regex:xmma|gemm|cutlass|nvjet|sm90|wgmma|flash|fmha|attention|rms_norm|rmsnorm|rotary|act_and_mul|silu|reshape_and_cache"
MV="sm__inst_issued.avg.per_cycle_active,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warps_eligible.avg.per_cycle_active,smsp__thread_inst_executed_per_inst_executed.ratio,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active,l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct,dram__throughput.avg.pct_of_peak_sustained_elapsed,launch__registers_per_thread"
mkdir -p runs/ncu
for R in "${REGIMES[@]}"; do
  echo "=== $(date +%T) capture $R (launch-count $LC) ==="
  sudo env "PATH=$PATH" "HF_HOME=$HOME/.cache/huggingface" \
    ncu --target-processes application-only --nvtx --nvtx-include "GEN/" -k "$KREGEX" \
    --section WarpStateStats --section SpeedOfLight --metrics "$MV" \
    --launch-count "$LC" -f -o "runs/ncu/$R" \
    "$PY" run_regime_h100.py "$R" > "runs/ncu/$R.log" 2>&1
  echo "$R done: $(grep -cE '==PROF== Profiling' runs/ncu/$R.log) profiled kernels"
  "$PY" build_gpu_tma_h100.py "$R" "runs/ncu/$R.ncu-rep" 2>&1 | head -20
done
echo "=== ALL DONE $(date +%T) ==="
