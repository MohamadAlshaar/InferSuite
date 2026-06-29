# CPU activity DURING inference — micro evidence

Backs the two claims that the GPU roofline only *implies*: (1) inference dominates wall-clock,
and (2) the CPU is near-idle while the GPU generates. These are **direct CPU measurements**, not
inferences from the GPU side.

## Figures
- **`agent_cpu_timeline.png`** — agent/tool CPU cores over wall-clock (fresh us-west-2 OpenClaw
  runs on c7i.metal, per-second `perf` cycles ÷ 3 GHz). The CPU sits at **median ~0.01 cores**
  (idle) and only spikes to ~2.3 cores in brief **tool-exec bursts** covering **11–23%** of
  wall-clock. The rest of the time (the GPU-generation windows) the CPU does nothing.
- **`vllm_cpu_timeline.png`** — the vLLM *server* CPU cores over time (local runs, `vllm_cpu_sampler.py`
  sampling /proc deltas of the engine processes). Steady **~1 core** during generation
  (OpenClaw 1.06, SWE 0.85, BCB 1.04 median) — the EngineCore loop (tokenize / schedule / sample /
  detok + eager kernel dispatch). Not compute-heavy.
- **`cpu_during_inference_summary.png`** — one-glance summary: agent idle 0.01 cores · tool-exec
  peak ~2.3 cores · vLLM ~1 core.

## What this establishes
During inference there are two CPUs and **neither is the bottleneck**: the **agent CPU is flat-idle**
(work happens in short tool bursts *outside* the generation windows), and the **vLLM server CPU is a
steady ~1 core**. This is the CPU-side confirmation of the time-donut split and is consistent with
the GPU roofline (decode is DRAM-bound: the GPU itself is ~80% idle on compute, so the CPU has even
less to do).

## Provenance / caveats (read before citing)
- **Agent timelines** = the finalized us-west-2 runs (`aws_agents/openclaw/data/oc_*_timeline.csv`).
- **vLLM timelines** = the *earlier local* runs (`{openclaw,swe_agent,bigcodebench}/runs/perf/vllm_timeline.csv`).
  The us-west-2 GPU box was not CPU-sampled, so the ~1-core figure is from the local captures; it is
  consistent across all three workloads.
- The earliest rows of each vLLM trace read ~0.02 cores — that's **engine startup**, not steady state
  (and the same artifact as the historical "vLLM looks idle" undercount bug). Medians exclude it.
- Cores ≈ `cycles/s ÷ 3 GHz` (nominal); absolute cores are approximate, the **idle-vs-burst ratio**
  is the robust result.
- vLLM ran `enforce_eager` (required for the ncu kernel capture); production CUDA-graphs would lower
  the ~1-core dispatch cost further, so this is the *pessimistic* CPU case.
