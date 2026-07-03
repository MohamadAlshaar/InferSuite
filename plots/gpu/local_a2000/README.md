# GPU top-down (TMA-style) figures — methodology, findings & the bug we caught

Inference regimes on an **RTX A2000** (Ampere), Qwen2.5-7B-Instruct-AWQ via vLLM, `enforce_eager`,
**FlashAttention-2**, NVTX-fenced `GEN` range, profiled with **Nsight Compute**
(`--section WarpStateStats --section SpeedOfLight`, in-process engine, `--target-processes
application-only`). Top-down built by `agentic/inference/build_gpu_tma.py`; data in
`agentic/inference/runs/ncu/gpu_tma.json`.

## ⚠️ A capture bug was found and fixed (an earlier version of these figures was wrong)

The first pass had `enable_prefix_caching=True` (vLLM default) **and** warmed up with the *same*
prompt it then measured. The measured "prefill" was therefore a 100% prefix-cache hit — only a
degenerate single-token step over cached KV. That inverted every conclusion (it made prefill look
*latency-bound* and *attention-dominated*). Fix: `enable_prefix_caching=False` + a distinct throwaway
warmup prompt, so the measured prompt is a **real prefill**. Validated three ways (attention grid,
GEMM SoL signature, duration math). The numbers below are the **corrected** captures.

## What a GPU top-down should be (NVIDIA Nsight Compute methodology)

The right denominator on a GPU is the **warp scheduler's issue slots**. Nsight prescribes a strict
order, which we follow:
1. **Speed-of-Light** — compute (SM) vs memory throughput as **% of peak** → compute-bound /
   memory-bound / latency-bound. This is the % -of-peak form of the roofline (ncu computes it against
   the correct mixed-precision / tensor-core ceilings; we lack FLOP/byte counters for an
   arithmetic-intensity roofline).
2. **Issue efficiency** — *"only look at stalls if the scheduler fails to issue every cycle"*; we
   report issued instructions per active scheduler cycle.
3. **Warp-state stall breakdown** — of the issue slots, what fraction issued vs why it stalled.
   `not_selected` = latency *hidden* (shown separately, not a bottleneck). Reasons roll up to:
   memory (long/short scoreboard, throttles), math-pipe (compute), execution-latency (`wait`),
   sync (barrier/membar), fetch/branch.

**We profile the *dominant* kernels, not all of them.** A `-k` filter captures only the meaningful
kernels (AWQ GEMM, attention, RMSNorm, RoPE, activation, KV-write) — ~98% of GPU time — and the result
is duration-weighted. The trivial elementwise kernels are excluded (negligible time). Decode is a pure
1-token-prompt run so every kernel is M=1 (no launch-skip fragility).

## Headline findings (corrected)

| regime | SoL compute / memory | dominant kernel | bottleneck |
|---|---|---|---|
| **Prefill-focused** | **81% / 28%** | AWQ GEMM 79% (86% compute) | **compute-bound** (math pipe) |
| **Decode-focused** | 43% / **61%** | AWQ GEMV 90% (67% memory) | **memory-bound** (weight bandwidth) |
| **Prompts** (real turn) | 72% / 36% | AWQ GEMM 82% | prefill-leaning blend |

- The **AWQ GEMM (Marlin) dominates every regime** (79–90% of GPU time) — *not* attention.
- The **same kernel flips bottleneck**: compute-bound in prefill (86% SM), memory-bound as the M=1
  GEMV in decode (67% memory, weight-streaming) — the classic prefill/decode roofline split.
- Prefill is **math-pipe-throttle** bound (45% of issue cycles); decode issues more (0.47 vs 0.21
  instr/cyc) but waits on memory.
- A real agent turn (**Prompts**) is **prefill-dominated**: at a 13.5K-token prompt the prefill is
  86% / 75% / 60% of GPU time for 64 / 128 / 256 output tokens (decode ≈ 40 tok/s). Figures use the
  128-token blend (75% prefill / 25% decode).

## The seven figures

| Figure | Shows | Audience |
|---|---|---|
| **01 Speed-of-Light** | regimes on compute% × memory% of peak | both — bottleneck class + headroom |
| **02 Warp-scheduler top-down** | issue slots: issued + stall classes | hardware (which pipe) + algorithm (which dependency) |
| **03a Kernel-time composition** | GPU-time by kernel class | algorithm — the GEMM is the hotspot everywhere |
| **03b Kernel bottleneck flip** | AWQ GEMM compute%/memory% in prefill vs decode | algorithm — fuse/batch decode (weight-bound); prefill already compute-bound |
| **04 Microarch signature heatmap** | dominant kernels × the **hardware** microarch measures (NOT the warp-state top-down), grouped: *Throughput/latency-hiding* (IPC, achieved occupancy, eligible warps) · *Compute pipes* (Tensor / FMA / ALU cycles-active) · *SIMT* (active lanes/warp) · *Memory hierarchy* (L1 hit, L2 hit, DRAM BW) · *Resources* (registers/thread). Per-column min–max colour, true values printed | both — the hardware fingerprint and the GPU analog of the CPU IPC/ILP/MLP/cache/AVX signature; shows the GEMM's tensor-pipe + IPC flip, attention's cache-hit behaviour, decode's DRAM-bandwidth pressure |
| **05 GPU top-down in Intel-style buckets** | warp-state re-cast into **Retiring / Front-end / Back-end{Core, Memory} / Covered**, with **Sync** as a GPU-specific side bucket; back-end % and heavy-op (tensor) annotated | both — the CPU-TMA-parallel view; shows the GPU is back-end-dominated (75/49/68%), prefill Core/tensor-bound vs decode more Memory; **Bad-spec = N/A** (no GPU speculation) and **Covered** (latency-hidden) has no CPU analog |
| **06 2-D throughput grid** | issue-efficiency (temporal) × lane-efficiency (spatial), regimes plotted as points | conceptual — GPU throughput is 2-D (the CPU is 1-D); useful work = issue-eff × lane-eff (**lane-weighted Retiring**); our kernels sit on the **lane ceiling** (divergence-free) so all loss is temporal |

### How the Intel-TMA buckets map to the GPU (research-grounded; see the slot-mapping synthesis)
A faithful "GPU TMA" is **not** a 4-bucket clone. The key results (grounded in DrGPU/GSI/GPA + the SIMT literature):
- **Retiring → `selected`/issued**, but *interpretation inverts* (low issue is fine when latency is hidden) and it should be **lane-weighted** (`LWR = Issued% × lanes/32`).
- **Heavy-ops → tensor-core MMA** — the *cleanest* L2 analog (one issue, multi-cycle macro-op; low IPC is *good*).
- **Bad-speculation → N/A** — GPUs don't speculate. Its role splits into **SIMT divergence** (spatial, wasted lanes ← branch-mispredict) + **instruction replays** (temporal, wasted slots ← machine-clears); both ≈0 here.
- **Front-end → `no_instruction`** (just I-cache; *fetch-bandwidth has no analog* — dropped).
- **Back-end → Core** (`math_pipe_throttle` + `wait`) **+ Memory** (`long/short_scoreboard` + throttles + DRAM) — clean, and the GPU memory side is *richer* (adds shared-mem + LSU-throttle leaves).
- **Covered (`not_selected`)** — GPU-only, **no CPU analog**; small here (occ 16–18%) → stalls are uncovered → *genuinely* back-end-bound.
- **2-D model**: throughput = issue-slots (temporal) × SIMT-lanes (spatial); the CPU TMA collapses to one axis.

### Microarch metrics captured (the GPU analog of the CPU signature)
Per the GPU-characterisation literature (esp. *Measuring GPU utilization one level deeper*, arXiv 2501.16909):
`sm__inst_issued.avg.per_cycle_active` (IPC, max 4/SM), `sm__warps_active…pct_of_peak` (achieved occupancy),
`smsp__warps_eligible…per_cycle_active` (latency-hiding headroom), `smsp__thread_inst_executed_per_inst_executed.ratio`
(SIMT lane efficiency ÷32), `sm__pipe_{tensor,fma,alu}_cycles_active…pct_of_peak` (which functional unit is hot —
**cycles-active, not the inst-executed rate**: HMMA tensor ops are multi-cycle, so the issue-rate variant understates
tensor-core busy-ness ~4× and disagrees with the SoL Compute%; cycles-active matches SoL exactly, e.g. prefill GEMM 86%),
`l1tex__t_sector_hit_rate.pct` / `lts__t_sector_hit_rate.pct` (L1/L2 reuse), `dram__throughput…pct_of_peak`
(memory-wall pressure), `launch__registers_per_thread` (occupancy limiter). Diagnostic rule: compute-bound = high
IPC (→4) + high pipe util + low DRAM BW; memory-bound = low IPC + high DRAM BW + cache misses; occupancy/latency-bound
= low occupancy + few eligible warps. *(Captured via `--metrics` alongside the WarpState + SoL sections — ncu only
records the counters you request, hence the re-capture.)*

## Caveats the thesis must state
- **batch=1 / single request + `enforce_eager`**: a latency-bound operating point, not throughput
  batched serving. Per-kernel HW counters are valid (ncu profiles each kernel in isolation), but the
  issue rate and time-split are batch-1 specific.
- **RTX A2000 is a PROXY** (26 SM Ampere consumer card), not the thesis target hardware. Exact %s and
  the prefill/decode time-split will not transfer to H100; only the qualitative direction (prefill
  compute-bound, decode memory-bound) is portable.
- **Shallow decode KV** (~1–256 tokens here vs agentic 16–32K): deeper KV raises the decode attention's
  memory-latency share. The decode GEMV (the dominant decode cost) is KV-independent and captured correctly.
- **Dominant-kernel coverage**: the `-k` filter omits trivial elementwise kernels (~2% of GPU time).

## Sources
- NVIDIA, *Nsight Compute Profiling Guide* (Speed-of-Light, Warp State Statistics, scheduler issue
  efficiency, Roofline): https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
- NVIDIA Technical Blog, *Accelerating HPC Applications with Nsight Compute Roofline Analysis*:
  https://developer.nvidia.com/blog/accelerating-hpc-applications-with-nsight-compute-roofline-analysis/
- NERSC, *Hierarchical Roofline analysis for GPUs* / Roofline docs: https://docs.nersc.gov/tools/performance/roofline/
- Williams, Waterman, Patterson, *Roofline: an insightful visual performance model* (CACM 2009).
