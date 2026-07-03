# GPU characterization — vLLM inference (Qwen2.5-32B-AWQ on L40S)

The GPU side of the agentic-workload study. **All three benchmarks (BigCodeBench,
SWE-agent, OpenClaw) drive the same Qwen2.5-32B-Instruct-AWQ model**, so the GPU
*kernels are identical across them* — profiling each separately yields near-identical
rooflines. The meaningful per-benchmark axis is the **prefill/decode balance**, captured
here as two profiles.

## Method (nsys → top kernels → ncu)
- Offline vLLM (`LLM` API, TP=1, `--enforce-eager`, FLASH_ATTN, in-process engine) on an L40S.
- **nsys** (`cuda_gpu_kern_sum`) → rank kernels by total GPU time → `gpu_kernel_time_donut.png`.
- **ncu** (`--set roofline` + `WarpStateStats` + `SpeedOfLight`), profiling within an NVTX
  `GEN/` range (fences out model-load kernels), `sudo` (counter permission), targeting a
  steady-decode window (`--launch-skip 660`) and an early-prefill window (`--launch-skip 0`).

## Figures
- **`gpu_kernel_time_donut.png`** — where GPU time goes: **~69% marlin AWQ-GEMM (W4A16)**,
  ~17% elementwise/norm/reduce, ~2% SwiGLU, attention+RoPE <2%.
- **`gpu_speedoflight.png`** — Compute(SM)% / DRAM% / Tensor-core% per kernel, decode vs prefill.
- **`gpu_warpstall_tma.png`** — GPU "TMA": warp-stall reasons grouped (Memory / Compute-pipe /
  Shared-L1 / Issue-latency / Frontend), 100%-stacked — the GPU analog of the CPU TMA.

## Result (marlin AWQ-GEMM, the 69% kernel)

| regime | SM% | DRAM% | Tensor% | dominant warp-stall | bound by |
|---|---|---|---|---|---|
| **Decode** (batch=1, serving) | 19 | **75** | 12 | `long_scoreboard` (memory) | **memory (DRAM) bandwidth** |
| **Prefill** (large batch) | **72** | 20 | **72** | `math_pipe_throttle` (tensor) | **compute (tensor cores)** |

**Decode is DRAM-bandwidth-bound** (batch=1 streams 4-bit weights, almost no compute per byte);
**prefill is compute/tensor-bound** (large GEMM saturates the tensor cores). This is the
canonical LLM-inference roofline, measured here on the L40S.

## Mapping to the three benchmarks
- **BigCodeBench** (long code prompts) is the most **prefill-heavy** → closest to the
  compute/tensor-bound point.
- **SWE-agent / OpenClaw** (short prompts, long generation) are **decode-heavy** → memory-bound.
- All are **GPU-inference-dominated at the wall-clock level** (85–90%, see the per-workload
  time donuts); the CPU TMA (FE/BE-bound interpreters, 0% AVX for agents; numpy AVX-512 for
  BCB tool-exec) characterizes the remaining 10–15%.

## Raw data (`data/`)
`ncu_decode_gen.ncu-rep`, `ncu_prefill_gen.ncu-rep` (open in Nsight Compute),
`nsys_decode.nsys-rep`, `nsys_run.log` (top-kernel summary),
`decode_raw.csv` / `prefill_raw.csv` (exported per-kernel metrics).
