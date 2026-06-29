#!/usr/bin/env bash
# GPU "TMA" (warp-stall breakdown) + Speed-of-Light + roofline, the deferred-ncu plan:
# offline vLLM LLM API, TP=1, enforce_eager -> nsys (top kernels) -> ncu (decode kernels).
# Run ON the L40S box after download_models.sh. ncu replays kernels = slow but bounded here.
set -uo pipefail
cd /opt/agentic; . venv/bin/activate
export MODEL="${MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct-AWQ}"
OUT=/opt/agentic/ncu_out; mkdir -p "$OUT"

cat > offline_gen.py <<'PY'
import os
from vllm import LLM, SamplingParams
m = os.environ.get("MODEL")
llm = LLM(model=m, quantization="awq_marlin", enforce_eager=True,
          tensor_parallel_size=1, gpu_memory_utilization=0.92, max_model_len=4096)
# enough new tokens to exercise DECODE kernels (the steady-state LLM-inference regime)
llm.generate(["Explain CPU cache coherence in depth."],
             SamplingParams(max_tokens=64, temperature=0))
PY

echo "[1/3] nsys: hottest kernels ..."
nsys profile -o "$OUT/trace" --force-overwrite true python offline_gen.py >/dev/null 2>&1 || true
nsys stats --report cuda_gpu_kern_sum --format csv "$OUT/trace.nsys-rep" > "$OUT/kernels.csv" 2>/dev/null || true
echo "  top kernels:"; head -12 "$OUT/kernels.csv" 2>/dev/null

echo "[2/3] ncu: roofline + warp-stall on decode kernels (skip prefill) ..."
M="sm__throughput.avg.pct_of_peak_sustained_elapsed"
M="$M,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"
M="$M,sm__warps_active.avg.pct_of_peak_sustained_active"
for s in long_scoreboard short_scoreboard wait barrier membar lg_throttle mio_throttle \
         math_pipe_throttle tex_throttle not_selected no_instructions imc_miss drain dispatch_stall; do
  M="$M,smsp__average_warps_issue_stalled_${s}_per_issue_active.ratio"
done
NCU="$(command -v ncu || echo /usr/local/bin/ncu)"
sudo "$NCU" --target-processes all --launch-skip 40 --launch-count 30 \
   --metrics "$M" --csv python offline_gen.py > "$OUT/ncu_metrics.csv" 2>"$OUT/ncu.err" \
   || { echo "  ncu failed — see $OUT/ncu.err (perm? reboot to apply NVreg flag, or keep sudo)"; }

echo "[3/3] parse -> GPU stall-reason bar + SoL/roofline"
python /opt/agentic/parse_ncu_tma.py "$OUT/ncu_metrics.csv" || true
