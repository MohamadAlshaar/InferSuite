# OC_clean figure set — manifest (created 2026-07-15; hardened OpenClaw campaign)

Data: `local_agents/OC_clean/data/` — OpenClaw × GLM-5.2 under the hardened environment
(same contract as SWE_clean: nohz_full+rcu_nocbs boot, ISO-PROOF quiet-partition gate per
launch, shuffled 8-group rotation @ WINSEC=5, continuous TMA census, lineage watcher).

| task | featured run | wall | WildClawBench score | attempts |
|---|---|---|---|---|
| jigsaw-med | run_1 | ~26 min | **0.74** | 2 (first parked: 20-min drain cut its grading phase → OC_DRAIN_S=1800 for all OC_clean episodes) |
| pdf-digest | run_1 | ~16 min | **0.76** | 1 |
| sam3-debug | run_1 | ~21 min | **0.00** | 5 — ALL attempts task-failed (runs 1,2,3,5 banked [OK]; one TERM-killed + one window-distribution attempt parked); first featured. The agent does the full torch debugging work every time (642-940 tool core-s) but never lands the graded fix — a GLM-5.2 capability boundary on this task, not a capture issue |
| connect-dots | run_2 | ~26 min | **0.15** | 2 — better of two featured (0.00 / 0.15); task never resolved in any campaign |
| scp-crawl (added) | run_2 | ~6 min | **0.96** | 2 — RESOLVED both attempts (0.95/0.96); run_1 had an E9 sparse-regime roll, run_2 featured. Heavier web sibling of arxiv-digest (sustained crawl+parse) |
| web-digest (added) | run_1 | ~6 min | **0.85** | RESOLVED first valid attempt (one credit-starved attempt parked; extra run_2 banked). The 4th resolved task |
| image-crop (added) | run_1 | ~7 min | **0.00** | task-failed on a full-conversation retry (first attempt credit-starved, parked) |
| paper-poster (added) | run_1 | ~7 min | **0.00** | 2 — both attempts task-failed (run_2 parked: window-distribution). Heavier image sibling of multi-crop (PDF/LaTeX/imagemagick composition) |

**FIGURES feature the FOUR RESOLVED tasks only (user decision 2026-07-15): jigsaw 0.74,
pdf-digest 0.76, web-digest 0.85, scp-crawl 0.96.** The failed tasks (sam3-debug 0.00 x5,
connect-dots 0.15 x2, paper-poster 0.00 x2, image-crop 0.00) remain fully banked and
validated in `data/` — their attempt records are in the table above; the thesis may cite
them (esp. sam3's 642-1940 core-s tool fences) without featuring them in figure panels.

Raw data of the failed tasks was DELETED 2026-07-15 (user decision, disk cleanup) —
scores and attempt history remain recorded in the table above; only the four resolved
tasks' episode data is retained under `data/`.

Selection rule: highest-scoring valid episode featured; every attempt banked; parked dirs
(`rejected_*`) are capture-invalid episodes (drain-cut grading), never counted.
All featured runs: validator [OK] — E1–E3/E6 proofs, E4 lineage purity corrected 100% /
contamination 0.00%, E9 census, E10 continuous-TMA coverage 100%.

Regenerate: `PLOT_SPEC=local_agents/OC_clean/plot_spec.json python3
local_agents/scripts/glm/plot_glm_results.py` + `plot_call_structure.py`.
Audit: same PLOT_SPEC + `audit_plots.py` (independent recompute vs values_dump.json).
Validate: `python3 validate_glm_agents.py local_agents/OC_clean/data glm`.

## Units, vocabulary, ratio construction, method validation

Identical to `../../SWE_clean/plots/MANIFEST.md` (locked): burst vocabulary
(0.005/0.02/0.3, 0.4 s gap, dust 0.001, exact usec integration); CO-COUNTED denominators
for all cross-group ratios (the 2026-07-15 dilution fix); windowed-rotation validity per the
probe calibrations; E6 informational under nohz_full (PMU is the OS-share source).
- **Occupancy, not "effective utilization"**: this metric counts ALL scheduler-accounted
  CPU time including spin/busy-wait — deliberately unlike Intel VTune's "Average Logical Core
  Utilization" (which excludes spin and overhead). A spinning core occupies a core-second like
  any other (unavailable to co-tenants, drawing power); the spin-vs-useful gap is itself a
  thesis result (the vLLM phantom-CPU), so it must remain visible in the measure.

OC-specific notes:
- **Fences**: scope1 = `<container>/agent` (node/V8 gateway = "harness"), scope2 =
  `<container>/toolexec` (spawned tools), scope3 = litellm.
- **Sparse tool fences**: OC light tasks run tools for seconds, not minutes — per-group tool
  windows are structurally few (1–5 busy). The validator reports these as documented-CI warns
  (a per-group floor cannot be satisfied by ANY retry of such a task); per-group tool L3
  ratios for light OC tasks are low-precision — lean on the cpu_work totals, burst structure,
  and the continuous TMA census instead.
- glm_harness_anatomy is SWE-only by design (CPython leaf classifier; the OC gateway is
  node/V8) — the plotter skips it for OC sets.
- Tool-call totals come from `transcript/chat.jsonl` (OC has no sweagent trajectory).


- **E9 sparse regime**: the heavy-burst census is a HARD distribution gate only when the
  episode has >=4x n_groups heavy bursts (balls-in-bins: below that an empty group is
  EXPECTED, not evidence of failure — scp-crawl missed a different group on each of two
  rolls). Sparse episodes report unexposed groups as CI.
- **V8 JIT drift**: OC gateway harness split-half up to ~21% is real node/V8 tier-up drift
  (identical signature on independent episodes, E2/E3 instrument proofs clean) — reported as
  CI; the hard split-half gate applies to the CPython SWE harness only.

## Figures

Same set as SWE_clean minus harness-anatomy: time_split, cpu_work, timeline,
timeline_cumulative (tool+gateway; litellm gap explained in prose), tool_calls,
call_structure, signature, hw_threads, tma_l1, tma_l2, tma_l3l4, tma_tree (starred proxies),
plus extra/glm_software_kernel. First OC campaign with the L3/L4 drill groups captured.
