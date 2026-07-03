# H100 co-located agentic run — CPU orchestration characterization

Separate results tree for the **fully co-located H100** experiments (vLLM + agents + tools on one
box). Keeps this data cleanly apart from the bare-metal `agentic/CANONICAL/` (7B, A2000, with TMA)
and the local `agentic/inference/plots/` trees.

## Goal
Characterize **what the CPU orchestrates DURING inference** for the agentic workloads (BCB, SWE-agent,
OpenClaw) at **32B on production H100 hardware** — de-emphasizing TMA (unavailable on this VM), and
focusing on: general microarch (IPC, cache-MPKI, branch-MPKI, FP), **function attribution** (native
C/C-API via `perf record`, Python via py-spy), **core-seconds** (cost), and **unmasking** the
orchestration hidden under the `cuEventSync` busy-wait spin.

## Hardware / environment (verified 2026-07-01)
- Lambda **H100 PCIe 80GB**, driver 580, CUDA 12.8; Xeon 8480+ (SPR), **26 vCPU KVM guest**, 891 GB free.
- vLLM 0.24.0 (`~/vllmenv`, py3.10); BCB harness venv `~/bcb/.venv` (bigcodebench 0.2.5 + task libs).
- **No Intel TMA topdown** on the VM (`topdown-*` PMU absent) — hence the function-attribution focus.
- **Basic PMCs work** (cycles/instructions/cache/branch/`fp_arith.*` incl. 512b packed-double).

## Models (provenance)
- **BCB, SWE-agent → Qwen2.5-Coder-32B-Instruct** (downloaded to HF cache).
- **OpenClaw → Qwen2.5-32B-Instruct**.
- Production serving config (cudagraphs on, prefix caching on, `--max-num-seqs 16`) — NOT the ncu
  `enforce_eager` profiling config.

## Measurement rigor (the three checks)
1. **No multiplexing / no silent zeroing.** The KVM vPMU exposes ~**6 GP counters** and *silently
   zeros* events beyond that (no `[NN%]` multiplex tag — verified: `cycles`→0 at 10 events). So:
   - **core** pass = `cycles, instructions, cache-references, cache-misses, branch-instructions,
     branch-misses` (4 GP + 2 fixed) + `task-clock` (software).
   - **fp** pass = `cycles, instructions, fp_arith.{scalar,128b,256b,512b}_double` — a **separate run**
     (adding 4 FP GP to core would oversubscribe → zeroing).
   - Attribution uses **`perf record -e task-clock`** (software event, **0 PMCs**) so it never contends
     with the counters. py-spy uses ptrace (0 PMCs).
   - `validate_run.sh` flags any zeroed `cycles` row or `pct<99` (multiplexed) row.
2. **Agent completion.** `validate_run.sh` checks solved/turns, `RUN_START/RUN_END`, matched
   `toolexec_start/end` marker pairs, and greps for `chat-err`/`Traceback`/`Connection refused`.
3. **Data integrity.** Every counter non-zero, no `<not counted>/<not supported>`, `perf record` +
   py-spy outputs non-empty.

## Conditions
Each app is captured under two serve conditions (the **unmask**):
- **spin** (baseline, `CUDA_BLOCKING_SYNC=0`) — host thread busy-waits; orchestration is masked.
- **evblock** (`CUDA_BLOCKING_SYNC=1`, `evblock.so` LD_PRELOAD) — host thread BLOCKS on sync; the
  residual CPU is the *true* orchestration.

## Layout
- `scripts/` — `serve_h100.sh`, `capture_orchestration.sh` (MODE=core|fp), `validate_run.sh`,
  `agentic_bcb.py`, `cudasync/` (evblock shim + sitecustomize), plot scripts.
- `data/`  — pulled captures per run: `engine_timeline.csv`, `engine_fp_timeline.csv`,
  `sys_timeline.csv`, `engine_perf.data`, `engine_pyspy.folded`, `markers.txt`, `agent.log`.
- `plots/` — figures.

## Results — BigCodeBench (Coder-32B, 12 tasks × 3 turns)  →  `plots/bcb/`

**Provenance note:** the prior CANONICAL "API run" driver = **Claude Sonnet 4.6** (cloud), NOT 7B. With
Claude the LLM runs in Anthropic's datacenter, so the local box had **no serving CPU**; this H100 run is
the first time BCB's during-inference orchestration CPU is measurable.

**Agent behavior** (`bcb_05_tasks_loops.png`): **6/12 solved, 26 loops** (= 26 tool-exec runs). Solved
tasks resolve in 1–2 turns; unsolved burn the full 3. (Claude Sonnet on full BCB-Hard ≈ 98/123 / 80%.)

**Time-split:** 91.2% inference / 8.8% tool (vs Claude-API 84/16) — both inference-dominated.

**During-inference engine microarch** (whole-run engine aggregate — NB the IPC here is the *whole-run*
value, diluted by tool-idle gaps + the py-spy attribution overhead in this live core capture; the clean
during-*generation* IPC is **2.97**, from the replay-basis fp/mem/stall groups, matching SWE and OpenClaw):
| condition | solved | IPC (whole-run) | cache-MPKI | branch-MPKI | core-sec |
|-----------|--------|-----|------------|-------------|----------|
| spin (baseline)   | 6/12 | 1.16 | 4.2 | 4.3 | 483.5 |
| evblock (blocked) | 6/12 | 0.78 | 7.5 | 7.4 | 468.1 |

**Unmask** (`bcb_02_coresec.png`): evblock reclaims only **~3%** (484→468 core-sec) → almost no reclaimable
busy-wait on vLLM 0.24 V1/H100 (contrast: **76%** on the old 7B/A2000 — very version/config-dependent).
The during-inference CPU is genuine orchestration + thread-scheduling churn, not a spin.

**What the CPU orchestrates** (Python, py-spy, `bcb_03`): output-bookkeeping `update_async_output_token_ids`
**57%**, sampler 11%, model/gpu-exec 11%, input-prep 8%, scheduler-policy ~2%.

**Native** (perf task-clock): ⚠️ the `bcb_04` figure showing "80% context-switch/scheduling" is
**superseded/retracted** — that capture ran a concurrent py-spy ptrace on `EngineCore` that faked the
scheduling. The clean re-capture (`bcb_during_record_clean`, no py-spy) is **91% CUDA GPU busy-wait**
(`libcuda` + `[vdso]` spin), i.e. the engine is spinning on the GPU, not scheduling. See
`grand_during_attribution.png`.

**FP** (`bcb_06_fp.png`): **2.23M scalar-double ops** over the whole run, **0% vectorized / 0% AVX-512** —
the engine does essentially no floating point (pure integer/control orchestration; the scalar trace is
sampling math). Sharp contrast to BCB *tool-exec* (~21% vectorized, real AVX-512 from numpy).

**One-liner:** during inference the self-hosted engine spends ~1.6 cores mostly **busy-waiting on the GPU**
(~91% `cudaEventSynchronize` spin) wrapped around a thin per-token output-bookkeeping+sample loop — a sharp
contrast to the tool-exec CPU *outside* inference (real Python compute). This serving-CPU view has no
counterpart in the Claude-API run.

## Results — SWE-agent (Coder-32B, 3 SWE-bench Verified tasks)  →  `plots/swe/`

Same 3 tasks as the CANONICAL run: **astropy-14096, scikit-learn-25232, sympy-14248**. Two views on the
same box: (a) DURING inference = replay each recorded trajectory's request stream to the engine under the
same 4-group PMC-safe basis as BCB (`traj_replay_engine.py` + `run_swe_during.sh`); (b) OUTSIDE inference =
run each instance's pytest suite inside its SWE-bench Docker image under cgroup-scoped perf, one counter
group at a time (`swe_toolexec.sh`). All 12 tool CSVs validated **no-mux / no-zero** (min active pct 100).

### The central two-view result (`swe_twoview_heatmap.png`)
**DURING-inference orchestration is one app-invariant signature; OUTSIDE, each task's tool work is its own.**

| row | IPC | L1 hit% | L2 hit% | cache-MPKI | MLP | vec-FP% | MFLOP |
|-----|-----|---------|---------|------------|-----|---------|-------|
| **engine · DURING inf** | **2.97** | 98 | 1.68 | 0.25 | 1.27 | **0.0** | 0.7 |
| astropy · tool  | 2.07 | 98  | 1.80 | 0.71 | 1.51 | 13   | 4.5  |
| scikit-learn · tool | **1.36** | 100 | 0.04 | 0.01 | 2.73 | **88** | **4096** |
| sympy · tool | 2.19 | 97 | 2.48 | 0.39 | 1.42 | **0.05** | 0.5 |

- The **during-inference** row is the *same* tight, L1-resident, high-IPC (2.97), zero-FP orchestration as
  BCB (IPC 2.97) — the engine's per-cycle character does not depend on which agent drives it.
- **Outside**, the tool-exec signature spans the full spectrum: **scikit-learn** is AVX-512-dominated dense
  linalg (88% vectorized, 50% of FP ops are 512-bit, ~1000× the MFLOP; IPC drops to 1.36 because the
  bottleneck is the vector FMA pipeline, not memory — L1 100%, mem-bound ~0). **sympy** is pure scalar /
  branch-heavy symbolic math (0% FP). **astropy** sits between (numpy on small arrays).

### DURING inference — app-invariant, and it's a GPU busy-wait (`grand_during_attribution.png`)
The during-inference engine CPU is app-invariant in **both** the per-cycle signature (IPC 2.97, L1 98%, 0 FP)
**and** the attribution: **~65-91% is the host thread busy-waiting on the GPU** (default `cudaEventSynchronize`
spin → `libcuda` + `[vdso]` clock-poll), doing no real work. Measured clean (py-spy-free, cgroup-scoped):
code-gen **91%**, SWE-agent **65%**, OpenClaw **89%** busy-wait.

> ⚠️ **Retraction.** An earlier version claimed a *context-length gradient* (BCB "79% thread-scheduling" →
> SWE/OpenClaw busy-wait). That was a **measurement artifact**: BCB's old core record ran a concurrent
> **py-spy ptrace on `EngineCore`**, which kept stopping the thread and faked "80% scheduling" (and 7×-low
> cycles / IPC 1.2). The clean BCB re-capture (no py-spy) is **91% busy-wait**, like the others. The
> `bcb_02`/`swe_02` per-benchmark donuts are superseded by `grand_during_attribution.png`. Confirmed against
> NVIDIA docs: default event sync busy-waits unless `cudaEventBlockingSync` is set.

### Time allocation — inference (GPU) vs tool (CPU), per task (`swe_time_allocation.png`)
Whole-agent-loop split: tool CPU = Σ trajectory `execution_time` (measured); inference GPU = forced-decode
replay wall-time (all turns run, 0 failures; contexts truncated keep-recent to fit `max_model_len`).

| task | GPU inference | CPU tool | split | total |
|------|---------------|----------|-------|-------|
| astropy | 244 s | 29 s | **89% GPU / 11% CPU** | 273 s |
| scikit-learn | 60 s | 144 s | **29% GPU / 71% CPU** | 204 s |
| sympy | 600 s | 29 s | **95% GPU / 5% CPU** | 628 s |

SWE-agent tasks are normally inference(GPU)-dominated (astropy 89%, sympy 95%), but **numerically-heavy
tool work flips it**: scikit-learn's AVX-512 imputation test suite makes the loop **71% CPU**. Same effect
as the CANONICAL `06_time_allocation`, now measured on the co-located H100 — and it lines up with the
tool-exec heatmap (scikit-learn is the 88%-vectorized / 4096-MFLOP row).

### Per-task figures
- `swe_signature_heatmap.png` — the 3 tasks' tool-exec signature side by side (fig-04 house style).
- `swe_tool_{astropy,scikit-learn,sympy}.png` — individual signature bars; `swe_tool_compare.png` — IPC/vec/MFLOP bars.

## Results — OpenClaw / WildClawBench (Instruct-32B, 4 tasks)  →  `plots/oc/`

Same 4 tasks as the CANONICAL run: **calendar, pdf-digest, web-digest (arxiv), image-crop (social-poster)**.
OpenClaw is a general **computer-use** agent where **one container = agent + all tools** (bash, Python,
Playwright/Chromium browser, PDF/image libs), so the OUTSIDE-inference CPU is the whole container cgroup.

**Provenance note:** the CANONICAL OpenClaw driver was **Claude-Sonnet-via-proxy** (`my_api.json` model id
`claude-sonnet-4-6`), so — as with BCB — this self-hosted Instruct-32B run is the *first* time OpenClaw's
serving CPU is on-box. Wiring fixes required for self-hosted vLLM: **`--enable-auto-tool-choice
--tool-call-parser hermes`** (the agent uses `tool_choice:auto`; without it → instant 400/quit),
**`--max-model-len 32768`** (the agent system-prompt alone overflows 16 K), and transferring the per-task
**workspace inputs**. Capture = live per-group runs (browser agents aren't deterministic → can't
record-replay; matches CANONICAL live-repeated method), container-cgroup-scoped, capped at 240 s/group.
All 16 tool CSVs validated **no-mux / no-zero** (min active pct 100).

### OpenClaw tool-exec is a distinct, compute-light corner (`grand_toolexec_heatmap.png`, `oc_signature_heatmap.png`)
| task | IPC | branch-MPKI | cache-MPKI | MLP | vec-FP% | MFLOP |
|------|-----|-------------|------------|-----|---------|-------|
| calendar    | 1.78 | 3.96 | 3.63 | 1.57 | **0.0** | 0.43 |
| pdf-digest  | 1.80 | 5.31 | 4.58 | 1.58 | **0.0** | 0.92 |
| web-digest  | 1.65 | 2.98 | 2.77 | 1.56 | **0.0** | 0.35 |
| image-crop  | 1.54 | 2.65 | 2.75 | 1.64 | **0.0** | 0.45 |

**Uniform signature regardless of nominal modality:** every OpenClaw task is branch-heavy (MPKI 2.6–5.3),
cache-miss-heavy (2.7–4.6), moderate-IPC (1.5–1.8), and **zero floating-point** (<1 MFLOP, 0% vectorized) —
even image-crop. The general computer-use agent's tool CPU is **Python/JS orchestration glue**, not numeric
kernels. This is the opposite end of the spectrum from the code agents that call optimized libraries:

- **grand cross-workload heatmap** (`grand_toolexec_heatmap.png`) places all self-hosted H100 tool-exec rows
  on one figure — **scikit-learn (SWE)** at the AVX-512-BLAS extreme (88% vec, 4096 MFLOP), the **OpenClaw**
  cluster at the compute-light / branch-bound extreme (0% FP), with **astropy/sympy/BCB-code-gen** between.
- The distinguishing axis is **does the tool work invoke optimized numeric kernels** (SWE-scikit → yes;
  OpenClaw general agent → no). CPU character is set by the *tool payload*, not by "it's an agent."

**Honesty caveat (32B-capability artifact).** The 0-FP result is partly because the self-hosted **32B is a
weak agent** on these hard tasks (scores ≈0) and often does not reach the *compute-heavy* tool step. The
2026-06-26 Claude-Sonnet CANONICAL run of the *same* arxiv-digest task showed **57% AVX / 507 MFLOP** (Claude
completed the numpy PDF-digest); here the 32B does orchestration + light ops, so the measured CPU is
agent-orchestration-dominated → 0 FP. So "OpenClaw tool CPU is compute-light" holds *for these 32B-driven
runs*; a stronger driver that finishes the numeric steps would surface FP. Valid for characterizing *the
work the CPU actually did*, not a claim that these tasks are inherently FP-free.

### DURING inference — engine app-invariance extends to OpenClaw (`plots/oc/`)
Engine (vLLM) captured during a live calendar run, cgroup-scoped to the engine's login-session (no-mux):
**IPC 2.97** — identical to BCB and SWE-agent. The during-inference orchestration signature is
driver-independent across all three agentic workloads (code-gen, code-repair, and general computer-use).

## Cross-workload summary figures  →  `plots/`

- **`plots/grand_timesplit.png`** — CPU(tool)-vs-GPU(inference) time allocation for **all 8 self-hosted
  tasks** (BCB code-gen, SWE ×3, OpenClaw ×4), 06-style donuts. Data sources: BCB from `markers.txt`
  (wall vs Σ toolexec pairs → 8.5% CPU); SWE from forced-decode replay wall vs Σ `execution_time`
  (astropy 11%, scikit-learn **71%**, sympy 5% CPU); OpenClaw from `chat.jsonl` timestamp decomposition
  (assistant gap = inference, toolResult gap = tool → 0.5–4.6% CPU). Story: agentic loops are
  inference(GPU)-dominated **except** when the tool payload hits optimized numeric kernels — only
  scikit-learn's AVX-512 test suite flips the loop to CPU-bound (71%). (Caveat: OpenClaw's low CPU% is
  partly the 32B not reaching heavy tool steps — see the OpenClaw honesty caveat above.)
- **`plots/oc/grand_toolexec_heatmap.png`** — the OUTSIDE tool-exec micro-arch signature for all 8 tasks
  on one heatmap (scalar-symbolic → AVX-512-BLAS spectrum).
- **`plots/grand_tool_attribution.png`** — the OUTSIDE tool-exec **recorded** CPU (`perf record`) by
  software component, per task, as role donuts. BCB/SWE = deterministic **replay** (re-run the recorded
  programs / pytest suites — no model); OpenClaw = **live** (browser agent, Instruct-32B). Symbols for the
  Docker workloads resolved via `--symfs=/proc/<container-pid>/root`; DSO-level view is `perf_dso.txt`.
  The **tool payload sets the CPU character**: BCB **77% BLAS** · astropy **59% cc1 compiler** (rebuilds C
  extensions — a fact `perf stat` couldn't show) · scikit-learn **56% BLAS** · sympy **92% CPython
  interpreter** · OpenClaw calendar/pdf/web **80-84% Node.js/V8** (the agent runtime *is* the CPU; the tools
  are quick) · image-crop 40% Python / 35% Node. Scripts: `bcb_tool_record.sh`, `swe_tool_record.sh`,
  `oc_tool_record.sh`, `plot_grand_tool_attribution.py`.
- **`plots/grand_during_attribution.png`** — the DURING-inference **recorded** engine CPU (`perf record`,
  task-clock) as role donuts per benchmark, bucketed by `NATIVE_ROLES`. **App-invariant: ~65-91% is the host
  thread busy-waiting on the GPU** (default `cudaEventSynchronize` spin → `libcuda` + `[vdso]` clock-poll) —
  code-gen **91%**, SWE **65%**, OpenClaw **89%** — doing no real work. All captured clean (py-spy-free,
  cgroup `-a -G`). ⚠️ Supersedes an earlier "context-length gradient" claim that was a py-spy-ptrace artifact
  on BCB's core capture (see the retraction in the SWE section). Matches NVIDIA docs (default event sync
  busy-waits) and *"Characterizing CPU-Induced Slowdowns in Multi-GPU LLM Inference"* (arXiv 2603.22774).
