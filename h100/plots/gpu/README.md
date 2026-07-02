# GPU top-down (ncu) — H100 · Qwen2.5-32B-Instruct (bf16) · FlashAttention-3

Same methodology and **same workload** (`agentic/inference/prompts.json`) as the local A2000 `inf_thesis_plots/gpu`
reference, re-captured on the co-located **H100 PCIe** with the **32B bf16** model.

## Setup
- vLLM 0.24 offline, `enforce_eager=True` (kernels launch individually → profileable), **`enable_prefix_caching=False`**
  + distinct warmup prompt (so the measured prefill is a real prefill, not a prefix-cache hit), NVTX-fenced `GEN` range.
- Attention backend = **FlashAttention version 3** (Hopper).
- **3 regimes** (identical selection to the A2000 run): **prefill** = largest ≤16K-token prompt (14,280 tok), out=1;
  **decode** = `"Hello"`, out=256 (pure M=1); **normal** = a random 8–16K-token agent prompt, out=64 (the measured agent turn).
- Capture: `ncu --section WarpStateStats --section SpeedOfLight --metrics <IPC, occupancy, tensor/FMA/ALU pipe,
  L1/L2 hit, DRAM BW, registers>`, dominant kernels via `-k`, `--target-processes application-only`, NVTX `GEN/`.
  Data: `h100/data_gpu/gpu_tma.json`. Scripts: `agentic/inference` originals + box `~/gpu_h100/run_regime_h100.py`,
  `run_ncu_full_h100.sh`, `build_gpu_tma_h100.py`; figures by `h100/scripts/plot_gpu.py`.

## ⚠️ FlashAttention-3 microarch is NOT measurable by ncu (but we still show its dominance)
The FA3 forward kernel is a **CUTLASS sm90 warp-specialized kernel** (`cutlass::device_kernel<flash::enable_sm90_or_later<…>>`)
that uses producer/consumer warp specialization + TMA + wgmma. **ncu cannot replay it for the warp-state / Speed-of-Light
counters**, so it is absent from the ncu top-down and from the micro-arch heatmap (`gpu_04`). That is a real tool limitation,
not a modelling choice.

We still measure **how much it dominates**: kernel wall-time from **torch.profiler (CUPTI chrome trace)** — which *traces*
rather than *replays*, so it times FA3 fine. Injected into the kernel-time figure (`gpu_03a`):
- **prefill: Attention = 16 % of GPU time** (GEMM 78 %) — long context → FA over 14 K tokens is a real chunk
- **agent prompts: 11 %** (GEMM 84 %)
- **decode: 2 %** (GEMM 94 %) — M=1 → attention is one query row, tiny
- Attention's share **scales with context length** (16 % → 11 % → 2 %), exactly as expected.
So: FA3's *time-share* is shown; only its *internal micro-arch* (warp-state stalls, pipe util, IPC) is N/A.

## Findings (figures `gpu_01 … gpu_06`)
- **`gpu_01` Speed-of-Light**: prefill is **compute-bound** (SM 85 % / tensor 97 %), decode is **memory-bound**
  (mem 76 % / compute 5 %), a real agent turn is **prefill-dominated / compute-bound** (compute 93 %).
- **`gpu_03a` kernel time**: **GEMM dominates every regime** (78–94 %); attention is the #2 slice in prefill (16 %).
- **`gpu_03b` / `gpu_04`**: the **same bf16 GEMM flips bottleneck** — compute-bound in prefill (tensor 97 %, DRAM 29 %)
  → memory-bound in decode (tensor 4 %, DRAM 83 %). Low occupancy (15–42 %) → back-end / latency-bound, not lane-divergent
  (SIMT 100 %).
- **`gpu_02` / `gpu_05` / `gpu_06`**: warp-scheduler top-down + Intel-style buckets + the 2-D (issue × lane) view.

## Caveats
- Prefill's ncu capture was short (7 kernels) and **missed the Activation kernel's micro-arch**, so the `Prefill · Activation`
  row is dropped from `gpu_04` (decode's Activation has full data). Class-level time-shares (`gpu_03a`) are unaffected (from CUPTI).
- Micro-arch heatmap excludes attention by design (FA3 unprofileable — see above).
