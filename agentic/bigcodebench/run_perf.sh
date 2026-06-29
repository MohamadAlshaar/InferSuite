#!/usr/bin/env bash
# Perf harness for BigCodeBench CODE-EXECUTION (the heavy, library-driven tool-exec).
# BigCodeBench evaluate --execution local runs the (ground-truth or generated) solutions +
# their test suites in local python subprocesses (numpy/pandas/scipy/sklearn/...). vLLM is
# IDLE during eval, so a system-wide perf (-a) is dominated by the code execution.
# ONE combined perf instance: TMA(fixed, clean) + GP cache/fp/mlp (multiplexed) -> ratios valid.
#
# Usage: run_perf.sh            (defaults: GT solutions, hard subset, local exec)
#        EVAL_CMD="..." run_perf.sh   (override the bigcodebench.evaluate command)
set -uo pipefail
cd "$(dirname "$0")"
# Resolve a WORKING perf binary + correct TMA group via the shared lib (verifies perf actually
# counts; bare 'perf' wrapper refuses to run on mismatched OEM/HWE kernels, and the slots-vs-legacy
# guess could pick events absent on this CPU -> all-zero passes).
. "$(dirname "$0")/../common/perf_events.sh"
. "$(dirname "$0")/../common/lib_perf.sh"
PERF="$(perf_bin)" || { echo "[bcb-perf] FATAL: no working perf binary (set PERF_HOST_BIN)"; exit 1; }
OUT=runs/perf; mkdir -p "$OUT"
TMA="$(tma_group)"
echo "[bcb-perf] perf=$PERF ; TMA=$TMA"
GP="mem_load_retired.l1_hit,mem_load_retired.l2_hit,mem_load_retired.l3_hit,mem_load_retired.l3_miss,fp_arith_inst_retired.scalar_single,fp_arith_inst_retired.128b_packed_single,fp_arith_inst_retired.256b_packed_single,fp_arith_inst_retired.512b_packed_single,fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.128b_packed_double,fp_arith_inst_retired.256b_packed_double,fp_arith_inst_retired.512b_packed_double,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles,uops_executed.thread,uops_retired.slots"

# default: run the GROUND-TRUTH solutions locally (guaranteed heavy execution, no LLM needed)
# --split must be complete|instruct (NOT a dataset version). GT execution runs
# canonical_solution+test (split-independent) so headline numbers are unaffected; fixed for correctness.
# NOTE: --samples __dummy__.jsonl is REQUIRED even for --check_gt_only: bigcodebench asserts
# samples!=None before the local path bypasses it (samples is never read in GT-only mode;
# get_groundtruth runs the dataset's canonical solutions).
EVAL_CMD="${EVAL_CMD:-python3 -m bigcodebench.evaluate --execution local --check_gt_only --samples __dummy__.jsonl --subset ${BCB_SUBSET:-hard} --split ${BCB_SPLIT:-complete} --parallel ${BCB_PAR:-4}}"

echo "[bcb-perf] eval: $EVAL_CMD"
( . .venv/bin/activate && eval "$EVAL_CMD" </dev/null ) > /tmp/bcb_eval.log 2>&1 &
EV=$!

# wait a moment for the eval to start executing (dataset load + first subprocesses)
sleep 8
echo "$(date +%s.%N) perf_start" > "$OUT/markers.txt"
echo "[bcb-perf] probing (system-wide; vLLM idle during eval)..."
# system-wide perf (paranoid=-1 -> no sudo). TMA fixed-counter stays clean; GP multiplexed.
"$PERF" stat -e "${TMA},${GP}" -a -I 1000 -x, -o "$OUT/bcb_timeline.csv" & PP=$!
# GPU sampler (expect ~idle -> confirms eval is CPU-only)
( while sleep 1; do printf "%s,%s\n" "$(date +%s.%N)" \
    "$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')"; done ) \
    > "$OUT/gpu_timeline.csv" 2>/dev/null & GPP=$!

wait "$EV"
echo "$(date +%s.%N) eval_done" >> "$OUT/markers.txt"
kill -INT "$PP" 2>/dev/null; sleep 1; kill "$PP" "$GPP" 2>/dev/null
echo "[bcb-perf] done -> $OUT  (eval log: /tmp/bcb_eval.log)"
