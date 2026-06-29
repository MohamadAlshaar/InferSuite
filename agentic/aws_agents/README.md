# Cross-workload CPU characterization — CORRECTED (2026-06-23)

## Agents verified working (the priority)
- BCB agent: solved 4/6 heavy numpy/sklearn tasks (real code-gen + test exec), temp 0.4.
- SWE agent: 89 steps, 211 tool calls, submitted a patch (deep genuine tool use, not degenerate), temp 0.4.
- claw-eval / OpenClaw: verified earlier (temp 0.6 fix was required — temp 0 made the 32B degenerate).

## Time split (wall-clock): GPU-inference-dominated
| run | window | GPU (generation) | CPU (tool-exec) |
|---|---|---|---|
| SWE (89 steps) | 617s | 536s = 87% | 79s = 13% (34s compile + 45s python) |
| BCB agentic | ~265s | ~86% | ~14% |
| claw-eval / OpenClaw | — | 84-95% | 5-16% |
The agent is mostly IDLE waiting on the 32B to generate; CPU lights up in brief tool bursts.

## CPU character (what the CPU does when it runs) — FP-CORRECT
| workload | core-s | TMA bound | AVX (FLOP-wtd) | float64? | what it is |
|---|---|---|---|---|---|
| BCB GT-exec (148 canonical) | 290 | Backend | **25% AVX-512** | 100% float64 | heavy numpy/BLAS (the compute payload) |
| SWE agent (this instance) | 37 | Retiring+FE | **0%** (genuine) | n/a (2.1M FP total) | scikit-learn COMPILE + light python |
| OpenClaw tools | 4 | FE+Backend | (re-measure) | — | node.js/V8 + curl/python |
| claw-eval multimodal | 1.2 | Backend | (re-measure) | — | chromium render |

## Two corrections vs the earlier write-up
1. **SWE was mislabeled "numpy/BLAS test-exec".** It's actually a scikit-learn REBUILD (Cython/gcc
   compile, integer, 0 FP) + light python, 87% idle. The bug is an API-signature fix (store_cv_values),
   so the numerical test suite never ran. **0% AVX is GENUINE** (confirmed with packed_double counters:
   2.1M total FP ops, 0 packed-double). Not a measurement artifact.
2. **The AVX bug was only in the live-agentic wrappers, NOT the GT-exec.** The GT full-suite always used
   PG_FP (with packed_double). Re-measured GT-exec: 100% float64, 6% packed BY INSTRUCTION but **25%
   AVX-512 BY FLOP** (lane-weighted) — reconciles with the earlier 26%. So BCB GT AVX was correct all along.
   The FP32-biased wrappers (SWE/claw/openclaw) are now fixed (full PG_FP).

## Note on the AVX metric
- BY INSTRUCTION count: % of FP instructions that are packed (numpy GT = 6%, mostly scalar).
- BY FLOP (lane-weighted): % of numeric WORK that's vectorized (numpy GT = 25% AVX-512). <- the meaningful one.

## OpenClaw (re-setup + FP-correct, FINAL 2026-06-23)
Re-cloned on fresh box. Fixes re-applied: docker_utils BRAVE/proxy/add-host, my_api.json openai-completions
+ qwen2.5-32b + new IP, vLLM tool flags + 32k, auto-approve (tools.exec.ask off), execution nudge.
FINAL set = **3 substantive runs** (plots: openclaw_time_donuts.png, openclaw_tma.png; details in openclaw/README.md):
  | task | wall | core-s | IPC | TMA Ret/FE/BS/BE |
  |---|---|---|---|---|
  | arxiv digest (Productivity) | 62s  | 5.9 | 1.59 | 32/32/15/23 |
  | tomllib trace (Search,local)| 26s  | 4.4 | 1.69 | 28/32/14/29 |
  | match report (Creative)     | 107s | 6.8 | 2.02 | 32/32/14/24 |
- Uniform **FE+Backend bound, 0% AVX (genuine)** — node.js/V8 + curl/python, no numpy/vectorization.
- DROPPED jigsaw (Code) + social: 32B did minimal work (plan-only / no output) = capability limit, not measurement.
- Short runs are STRUCTURAL: OpenClaw ends the loop on a no-tool-call turn; the 32B concludes early (vs SWE-agent's
  tool_choice:required → 89 steps). Tried to force tool_choice via a request-injecting proxy on the GPU box — the
  OpenClaw gateway did not integrate cleanly through it, so these are the model's natural (short) trajectories.
- web_search broken (dummy Brave key → 422); Search represented by LOCAL tomllib task (worked). temp 0.6 used.

## FINAL cross-workload CPU map (FP-correct, agents verified)
| workload | agent works? | core-s | TMA bound | AVX | what the CPU is |
|---|---|---|---|---|---|
| BCB GT-exec (148 canonical) | n/a (no agent) | 290 | Backend | 25% AVX-512 (FLOP) | heavy numpy/BLAS float64 |
| BCB agentic | YES (4/6 solved) | ~230 | Backend | (numpy) | sustained numpy tests |
| SWE agent | YES (89 steps, submitted) | 37 | Retiring+FE | 0% genuine | scikit-learn COMPILE + python |
| OpenClaw (3 tasks) | YES (output produced) | 4.4-6.8 | FE+Backend | 0% genuine | node.js/V8 + curl/python |
| claw-eval general/multi_turn | YES (temp 0.6) | ~0 | - | - | file-I/O / consultation |
=> Agents all GPU-dominated (85-90% inference). CPU character = the TOOL PAYLOAD: numpy float64 is the only
   AVX/backend-heavy one; agent runtimes (python/node.js) + compiles are retiring/FE-bound, 0% AVX.
