# agentic/common — shared CPU-measurement library

One canonical implementation of the measurement methodology so all three workloads
(SWE-agent/SWE-bench, BigCodeBench, OpenClaw) are at **parity by construction** instead
of drifting per-harness copies. Built after the 2026-06-20 validation that found the old
copies disagreed and carried silent bugs (see memory `project_validation_findings`).

## Canonical methodology — 6 passes
| pass | events (`perf_events.sh`) | metrics | scope |
|------|---------------------------|---------|-------|
| live | PG_TMA (+ GPU/cpu samplers, markers) | time donut, core-seconds, TMA-L1 | cgroup `-G` (or `-a` if vLLM idle) |
| cache | PG_CACHE | AMAT, LLC-MPKI | cgroup / replay |
| fp | PG_FP | FLOPs, vectorized%, AVX-512% | cgroup / replay |
| mlp | PG_MLP | MLP, ILP (SMT-correct) | cgroup / replay |
| imc | PG_IMC (CHA, node-wide) | DRAM GB | `-a` only (uncore can't be cgroup-scoped) |
| toplev | pmu-tools `toplev -l2` | TMA-L2 tree | `-a` |

**Determinism decides execution, NOT the pass set:**
- SWE-bench / BigCodeBench = deterministic → record once, **replay** each deep pass (no GPU).
- OpenClaw = non-deterministic (browser/exec) → **live, repeated** runs, one pass per run, averaged.

## Fixes baked in (each was a real bug in the old copies)
- **FP packed-double** included → old set was blind, falsely reported "0% AVX". FLOPs reported as a bracket `[no-FMA .. all-FMA×2]`.
- **IMC = uncore_cha** (`cas_count` does NOT exist on EKS c7i.metal); bytes = count×64.
- **ILP = uops_executed.thread** (SMT-correct), not `.core` (up to 2× inflated).
- **HARD-FAIL on zero/empty** perf output (`assert_perf_ok`, parser exits 1) — no more silent IPC 0 read as real.
- **Separate TMA pass** (fixed counters, 100%) — old combined TMA+GP instance multiplexed to ~73%.
- **Effective frequency** from `scaling_cur_freq`, never hard-coded 4.6 GHz.
- **stdin closed** on workloads (no Y/N prompt hang); **SIGINT+wait** flush (not fixed sleep); no nested `&` orphaning.
- **TMA-sum assertion** (`retiring+fe+badspec+be ≈ slots`) catches multiplexed/garbage runs.

## vLLM "CPU during inference" — IMPORTANT
The EngineCore's high IPC/retiring is largely a **CUDA-sync SPIN** (`cudaEventSynchronize`),
not compute (validated live via gdb). Always run `vllm_spin_sample.sh` during decode to
report the **spin% vs real-work%** split, and label vLLM core-seconds as *spinning, not
computing*. This is regime-dependent (enforce_eager + low batch maximizes spin).

## Files
- `perf_events.sh` — the event groups (source it).
- `lib_perf.sh` — perf binary resolution, sudo, paranoid/mlock enable, **effective freq**, `assert_perf_ok` (hard-fail), `perf_aggregate`, `cgroup_of_pid`.
- `parse_perf.py` — merges N perf files → all metrics (system python3). `--freq-hz`, `--json`, `--label`.
- `vllm_spin_sample.sh` — gdb-sampled spin-vs-work split (perf record is broken on this kernel/perf combo).

## Platform notes
- Local box + EKS c7i.metal = Sapphire Rapids → events validated, transfer cleanly.
- EKS p5 GPU node is Nitro → **no CPU PMU** (vLLM TMA only on bare-metal); ncu works there.
- `perf stat` works on kernel 6.17 + perf 6.8; **`perf record` does not** (ABI) → use gdb/strace for code attribution.
- vLLM uid 2000 → `perf -p` needs `sudo` + `perf_event_paranoid=-1`.
