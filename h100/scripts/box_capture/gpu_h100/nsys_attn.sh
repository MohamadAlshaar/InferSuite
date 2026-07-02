#!/usr/bin/env bash
# nsys TRACE (not replay) each regime -> per-kernel wall-time -> the FA3 attention time-share that ncu
# cannot get. Whole-trace (GEN prefill dominates the tiny warmup). Requires GPU free (run after ncu).
set -u
cd "$HOME/gpu_h100"; PY="$HOME/vllmenv/bin/python3"
pkill -9 -f run_regime_h100 2>/dev/null; sleep 2
mkdir -p runs/nsys
for R in prefill decode normal; do
  echo "=== $(date +%T) nsys $R ==="
  sudo env "PATH=$PATH" "HF_HOME=$HOME/.cache/huggingface" \
    nsys profile --trace=cuda,nvtx --force-overwrite true -o "runs/nsys/$R" \
    "$PY" run_regime_h100.py "$R" > "runs/nsys/$R.log" 2>&1
  nsys stats --report cuda_gpu_kern_sum --format csv --force-export true "runs/nsys/$R.nsys-rep" 2>/dev/null > "runs/nsys/${R}_kern.csv"
  echo "$R: $(grep -c "," "runs/nsys/${R}_kern.csv") kernel rows"
done
echo NSYS_ATTN_DONE
