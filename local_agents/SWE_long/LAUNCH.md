# SWE_long — long-horizon SWE-agent campaign (GLM-5.2)

Started 2026-07-11. Extends the certified 40-min GLM campaign to 90-min episodes and
non-Python languages. Same kit (`../scripts/glm/run_glm_campaign.sh`), same isolation,
same capture contract; only env overrides differ. Data root: `local_agents/SWE_long/data`
(via `DATA_ROOT`), plots later in `local_agents/SWE_long/plots`.

## Design (agreed 2026-07-11)

| # | Task | Language | Rationale |
|---|------|----------|-----------|
| 1 | `django__django-16560`  | Python | REPLACED 10097 (loop-collapsed 3-for-3). Verified difficulty "1-4 hours", 2-file fix, 2023 testbed (modern python). Smoke #3 PASSED: E7 clean (144/156 unique, max run 1), 64 edits + 25 test runs, SUBMITTED at 16 min — but UNRESOLVED on hidden tests. Insight: episode length is confidence-capped, not difficulty-capped (agent sees only existing tests). **TASK 1 DONE 2026-07-11: run_1 = 12 min, 134 steps, ALL GATES OK (E7 130/135 uniq), RESOLVED. Replay anchor x3 PASS: same-traj tool wIPC disp 0% [1.434/1.429/1.427], live<->replay DSO-match 95%, br_ki 1%. Isolation certificate BANKED.** |
| 2 | `sympy__sympy-13878`    | Python | ">4 hours" length probe: full 60-min capped episode, 775 tool CPU-s, replay-anchored — BANKED but demoted from figures. **FIGURES now use `sympy-14248` (`glm_swe_sympy-light`, 2026-07-12): 61 min, 434 steps, 104 edits, E7 clean, RESOLVED** — same instance as certified temp-0 runs = direct temp ablation (0.0: 188-311 steps/31-41 min; 0.6: 434 steps/61 min, more exploration, resolved). |
| 3 | `babel__babel-15445`    | JavaScript | FINAL (15649 looped 654x for 3h -> deleted; then guard false-trip on wrapped "cat" first-lines -> guard fixed to full-block compare). **DONE: 15 min, 174 steps, 97% unique, RESOLVED.** 33% of 15649's size per user call. |
| 4 | `fmtlib__fmt-3248`      | C++ | FINAL (3750 -> 3901 (killed at ~3 min per user) -> 3248, 592B = babel-sized). **DONE: 25-min episode (34-min capture), 169 steps, 91% unique, 56 edits + 25 compile/test cycles, RESOLVED, in-band.** |

**CAMPAIGN COMPLETE 2026-07-11 night: 4/4 banked** (3 resolved + 1 capped-hour), all E-gates
green, replay anchors on tasks 1-2 (0-1% dispersion, 95-100% DSO-match), live loop guard
operational (full-block compare). Figures: `plots/` 4-panel certified style + internal-tools.
REMAINING: audit_plots.py adaptation for SWE_long; certified-data loop audit (user decision);
OC_long after eBPF rung 2 (streaming off); rotate GLM key.

- Caps: `SWE_DRAIN_S=5400`, `REPLAY_DRAIN_S=5400`. `REPEATS=1` (single live episode per task).
- Isolation proof: replay x2 on task 1 (Python spawn pattern) and task 4 (C++ `make -j`
  spawn pattern) instead of 3 live repeats — replay-vs-replay = measurement stability,
  live-vs-replay tool fence = contamination.
- Workflow: task-by-task; after each episode -> validator + review gate BEFORE the next task.
- Model: glm-5.2 via z.ai (litellm proxy), thinking enabled — identical to certified runs.
- Tasks 3-4 need: Multilingual dataset wiring in the runner + PRE-BUILT env images
  (episode fails if sandbox not up in 240 s). Do NOT start them before both are done.

## Incident 2026-07-11: temp-0 degenerate loop (first task-1 attempt DELETED)
- First django 90-min run: all E-gates PASS (measurement valid; harness medIPC 2.903 dead-center
  in certified band) but the WORKLOAD was degenerate: real work for 26 steps (~4 min), then the
  SAME grep repeated 554x for ~85 min. No submission -> unresolved. Data + replays deleted.
- Root cause: sweagent default `temperature: 0.0` (kit inherited it). Same failure mode as the
  two prior Qwen incidents (see memory feedback_temp0_degenerate_agent).
- CERTIFIED data scan (trajectory action-uniqueness): loops exist there too (also temp 0.0):
  astropy r2 (379-step loop), django r2/r3 (375/411), django-lite r1 (alternating, 5% unique),
  django-lite2 r1 (144), scikit r2 (311), sympy r3 (377); django r1 traj = 0 bytes (truncated).
  Clean: astropy r1/r3, scikit r1/r3, sympy r1/r2. AUDIT PENDING (user decision): check whether
  median-run selection in certified figures landed on loop runs.
- Framing: certified campaign = STOCK SWE-agent config (temp 0.0 is its shipped default) —
  loops are stock behavior, label as such. SWE_long = production-like sampling (temp 0.6,
  explicit, recorded in metadata). One knob at a time: penalties only if 0.6 still loops.
- New: `SWE_TEMP` env (default 0.0 = certified behavior) + validator gate E7 action-uniqueness
  (FAIL when >=10 consecutive identical actions). Procedure: 20-min temp-0.6 smoke -> E7 check
  -> delete smoke -> full 90-min run.
- Smoke #2 (temp 0.6 + fixed submit): E7 FAIL x223 — GENUINE model loop this time (git diff
  between two commits of test files -> "0 lines" -> repeated 223x from step 39; never reached
  submit). GLM-5.2 self-conditions on repetition even at 0.6. django-10097 is 3-for-3
  loop-collapsed (temp0 grep x554 / submit-tool retry x146 / git-diff x223). z.ai probe:
  presence_penalty + frequency_penalty both HTTP 200 (may be silently ignored — unproven).
  KEY INSIGHT: clean certified episodes on these Verified instances were SHORT (astropy 7-15
  min, scikit clean runs 7-20 min) — every "long" certified episode was a loop artifact. A
  90-min GENUINE-work run needs harder instances, not just a raised cap. DECISION PENDING.
- Smoke #1 at temp 0.6 (E7 FAIL x63, but NOT sampling): agent did REAL work (reproduce.py,
  2 str_replace edits to validators.py, ran test suite, submit at step 41) — then sweagent's
  `review_on_submit_m/bin/submit` crashed with SyntaxError (f-string; django-10097 testbed
  python < 3.6) and the agent rationally retried submit 146x. FIX: local patch in
  external/SWE-agent (f-string -> .format; checkout also carries pre-existing run_replay.py
  patch). Re-smoke after fix. NOTE: temp 0.6 exploration looked healthy (84% unique pre-submit)
  — and certified django "treadmill" durations are partly LOOP artifacts, so if a working agent
  solves django-10097 quickly, task 1 needs a genuinely harder instance (decision after re-smoke).

## Pre-registered hypotheses
- Harness signature invariant across languages (IPC 2.45-2.89, uop-cache 69-75%, 1-core ceiling).
- django burst-growth curve continues past 40 min, then flattens when harness context
  management kicks in (treadmill saturation) — either outcome is a finding.
- JS tool fence: V8/JIT species — L1I/MITE pressure, higher bad-spec, ~zero packed FP.
- C++ tool fence: cc1plus-dominant compile bursts + high-IPC native test binaries.

## OC_long (later, after eBPF watcher rung 2)
- 4 long tasks: jigsaw-hard, python data-pipeline, node/JS project, C++ project (tests-pass graders).
- STREAMING: ON (reversed 2026-07-12). Verified NO stock knob exists for model-API streaming
  (all documented OpenClaw streaming config is channel-side: blockStreaming*, channels.*.streaming).
  Un-streaming shim option declined by user -> keep streaming = certified comparability preserved
  (proxy shares directly comparable to certified calendar 78%).
- JS OC task REQUIRES the eBPF lineage watcher: ground-truth test (2026-07-11, scratchpad
  octest) measured 100% node-tool misattribution + comm-E4 blind (87% "pure" vs 15% true),
  storm leakage ~68% pinned / 78% unpinned; tool-side 0.00% contamination in all cases.
