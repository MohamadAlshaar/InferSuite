#!/usr/bin/env bash
# Complete capture per regime: warp-state TMA (WarpStateStats) + Speed-of-Light + the MICROARCH hardware
# measures (IPC, occupancy, eligible warps, SIMT efficiency, FMA/Tensor/LSU/ALU pipe util, L1/L2 hit,
# DRAM bandwidth, registers/thread). Dominant kernels only (-k). One self-contained .ncu-rep per regime.
set -u
cd /home/mohamad/llm-service-kernel-latest/agentic/inference
pkill -9 -f run_regime.py 2>/dev/null; pkill -9 -x ncu 2>/dev/null; sleep 3
KREGEX="regex:Marlin|flash_fwd|rms_norm|rotary_embedding|act_and_mul|reshape_and_cache"
MV="sm__inst_issued.avg.per_cycle_active,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__warps_eligible.avg.per_cycle_active,\
smsp__thread_inst_executed_per_inst_executed.ratio,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,\
sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active,\
l1tex__t_sector_hit_rate.pct,\
lts__t_sector_hit_rate.pct,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
launch__registers_per_thread"
for R in prefill decode normal; do
  echo "=== $(date +%T) full capture: $R ==="
  sudo env "PATH=$PATH" "HF_HOME=/home/mohamad/.cache/huggingface" \
    ncu --target-processes application-only --nvtx --nvtx-include "GEN/" -k "$KREGEX" \
    --section WarpStateStats --section SpeedOfLight --metrics "$MV" \
    --launch-count 60 -f -o "runs/ncu/$R" \
    ../bigcodebench/.venv/bin/python3 run_regime.py "$R" > "runs/ncu/$R.log" 2>&1
  echo "$R done: $(grep -cE '==PROF== Profiling' runs/ncu/$R.log) kernels | metric-errors: $(grep -ciE 'could not|invalid metric|not found' runs/ncu/$R.log)"
done
echo "=== $(date +%T) ALL FULL CAPTURES DONE ==="