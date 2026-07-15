# SWE_clean figure set — manifest (created 2026-07-15; the HARDENED rerun campaign)

Data: `local_agents/SWE_clean/data/` — one live episode per task under the hardened
environment, plus method-probe episodes and replay anchors. Model: GLM-5.2 (z.ai, thinking
on), temp 0.6, litellm proxy, SWE-agent.

| task | instance | turns | wall | outcome |
|---|---|---|---|---|
| django | django__django-16560 (Verified) | see below | — | RESOLVED on attempt 3-of-3-banked (run_3, swebench-verified). Selection rule: first resolved episode featured; run_1 + run_2 (both submitted-but-unresolved, measurement-valid) kept banked and labeled; balance-contaminated attempt parked. |
| sympy | sympy__sympy-14248 (Verified) | 389 | 81 min | RESOLVED (swebench-verified) |
| babel | babel__babel-15445 (Multilingual) | 92 | 7.3 min | RESOLVED (swebench-verified; attempt 1 rejected: E7 loop ×13, parked) |
| fmt | fmtlib__fmt-3248 (Multilingual) | 112 | 16 min | RESOLVED (swebench-verified) |

Probe episodes (method validation, NOT figure sources): `glm_swe_babel/run_2/4/5` — each
dedicates the whole episode to ONE drill group (`GORDER_OVERRIDE`); `run_3` rejected (loop).
Replay anchors: `glm_replay_swe_{django,fmtlib}/run_1` (live↔replay PASS: dso 98/99%,
cost 5%/0%).

Regenerate: `PLOT_SPEC=local_agents/SWE_clean/plot_spec.json python3
local_agents/scripts/glm/plot_glm_results.py` + `plot_call_structure.py` +
`plot_harness_scaling.py`. Audit: `PLOT_SPEC=... python3 audit_plots.py` (independent
recompute vs `values_dump.json`). Validate: `python3 validate_glm_agents.py
local_agents/SWE_clean/data glm`. Curated shortlist: `plots/thesis_ready/`.

## The hardened environment (differences vs SWE_long)

- Boot: `nohz_full=2-11,14-23 rcu_nocbs=2-11,14-23` (tickless measured cores, RCU offloaded).
  `isolcpus` was tested and REJECTED: it removes the cores from scheduler balancing — a
  20-way pool stacked on ONE core (measured) — which would invalidate tool parallelism.
- Per-launch ISO-PROOF gate: slices/knobs asserted AND the measured partition sampled silent
  (<2% busy over 1.5 s) before any capture; `kptr_restrict=0` pinned in preflight.
- Rotation: 8 zero-multiplex groups (`fpbr cache mlp fe fe_lat core_ports dram_bw priv`),
  **shuffled every cycle** (kills systematic-sampling phase-aliasing; realized order recorded
  in windows.tsv), WINSEC=5 s.
- TMA L1/L2 runs **continuously** per episode (`tma_cont.csv`, PERF_METRICS + fixed slots —
  zero GP counters, 100% coverage; E10 gate).
- Environment A/B vs the soft-isolated SWE_long: replays of banked trajectories reproduce at
  97–100% signature and ±3% cost. nohz_full note: the scheduler's user/system SPLIT on
  tickless cores is vtime-accounted and no longer a precision cross-check (E6 informational);
  the PMU (`cycles:k`) is the OS-share source. Context tracking adds real per-syscall kernel
  cycles (pathological getppid loop: +8pp) — on real workloads the replay A/B measured +0.7pp.

## Units, vocabulary, burst definitions

Identical to `../../SWE_long/plots/MANIFEST.md` (locked): CPU usage (cores) =
core-seconds/second; amounts in core-seconds (exact cgroup deltas); OS share (never
"kernel"); core = logical CPU, partition = 20. Bursts: tool >0.005 / harness >0.02 cores,
gaps <0.4 s merged, ≤0.001 core-s dust dropped, heavy >0.3 peak; exact usec integration.
- **Occupancy, not "effective utilization"**: this metric counts ALL scheduler-accounted
  CPU time including spin/busy-wait — deliberately unlike Intel VTune's "Average Logical Core
  Utilization" (which excludes spin and overhead). A spinning core occupies a core-second like
  any other (unavailable to co-tenants, drawing power); the spin-vs-useful gap is itself a
  thesis result (the vLLM phantom-CPU), so it must remain visible in the measure.

## Ratio construction (CORRECTED 2026-07-15 — the dilution fix)

Cross-group ratios (branch/L1I/L1D/LLC MPKI, L3 drill children) use CO-COUNTED denominators:
instructions/cycles summed over exactly the windows where the numerator event was counted.
The previous construction divided by the ALL-groups sum and understated these ratios ~7.9×
(caught by the dedicated-group probes; also affected the SWE_long/OC_long and archived
certified figures — those sets regenerated 2026-07-15). Same-group ratios (IPC, DSB, AMAT,
MLP, OS share, packed-FP, TMA) were never affected.

## Method validation (windowed rotation vs dedicated-continuous probes)

- TMA census vs windowed TMA (identical trajectory, sympy replay): ±1–3pp.
- fe_lat children: 2–18% · core_ports: 3–10% · dram_bw: 25–36% (across DIFFERENT live
  episodes — DRAM occupancy carries real per-episode variance; treat as CI).
- Per-episode split-half wIPC (busy windows): harness 0.6–4%; tool 2.5–20% (fmt's 20% =
  explore→compile compositional drift — report as that episode's CI, instrument control =
  harness on the same run).
- Coverage gates: ≥7 busy windows/group hard (SMARTS-derived at measured CV≤4%), 10 = target;
  babel harness carries 7–8-busy-window warnings on three groups (wider CI, documented here).
- E9 heavy-burst census: every group saw bursts in every kept episode (shuffle working).

## Figures

Story: **glm_time_split** (wall split; tool = any activity >0.005), **glm_cpu_work**
(exact 3-fence core-second donuts + tables), **glm_timeline** (10 Hz raster),
**glm_timeline_cumulative** (tool+harness only BY CHOICE — the donut carries the 3-fence
total; the 2–6% gap is the litellm share, explained in prose), **glm_tool_calls**,
**glm_call_structure** (per-burst core-seconds), **glm_calls_vs_bursts**,
**glm_internal_tools**, **glm_harness_scaling** (cross-campaign power law: harness core-s ∝
turns^2.69, R²=0.998, n=12 episodes / 4 repos of 68–531 KLOC — orchestration cost is set by
conversation length, not codebase; django vs sympy = same code size, 22× work at 3.1× turns).

Microarchitecture: **glm_signature** (corrected MPKIs; tool L1I MPKI 8–22 = the
instruction-footprint wall), **glm_tma_l1** (single panel; uop-delivery panel retired —
DSB share lives in the signature, DSB/MITE split in the tree), **glm_tma_l2**,
**glm_tma_l3l4** (exact measured children as % of fence cycles; unmeasured siblings omitted
so panels do NOT sum to parents; ports panel = raw width profile, not parent-nested),
**glm_tma_tree** (L1→L4 in one stacked bar; * = labeled proxy splits: FE-bw by uop-delivery
shares, memory levels by load-latency weights — indicative, not measured stalls),
**glm_hw_threads**, **glm_harness_anatomy**.

`thesis_ready/` = the curated 12 (user-selected; excludes call_structure, hw_threads).
