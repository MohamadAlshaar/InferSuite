# OpenClaw (WildClawBench) — CPU characterization

Model: **Qwen2.5-32B-Instruct-AWQ** served by vLLM on an L40S (awq_marlin, FLASH_ATTN,
`--enable-auto-tool-choice --tool-call-parser hermes`, `--max-model-len 32768`).
Agent: OpenClaw (node.js/V8) running in a container on a c7i.metal box; perf attached to
the agent+tools cgroup. FP-correct event set (PG_FP, includes packed-double).

## What's shown — 3 substantive task categories

| task | category | wall | tool-exec | core-s | IPC | TMA (Ret/FE/BS/BE) |
|---|---|---|---|---|---|---|
| arxiv digest    | Productivity     | 62s  | 9s   | 5.9 | 1.59 | 32/32/15/23 |
| tomllib trace   | Search (local)   | 26s  | 6s   | 4.4 | 1.69 | 28/32/14/29 |
| match report    | Creative         | 107s | 12s  | 6.8 | 2.02 | 32/32/14/24 |

- **`openclaw_time_donuts.png`** — wall-clock split: ~85–90% GPU (32B generating) /
  ~10–15% CPU tool-exec, per category.
- **`openclaw_tma.png`** — TMA top-down: uniformly **Frontend + Backend bound**
  (node.js/V8 interpreter + curl/python tool processes), IPC 1.6–2.0, **0% AVX**
  (no FP/SIMD — orchestration + I/O, not numeric kernels).

## Honest caveats (read before citing)

1. **Only 3 of 5 attempted categories are substantive.** `jigsaw` (Code) and a Social
   task were dropped: the 32B did minimal work (plan-only / no output produced) — a
   **model limitation**, not a measurement error.
2. **OpenClaw is a short-run agent with the 32B.** Its loop ends the moment the model
   returns a turn with no tool call (a "final answer"), and the 32B concludes after a
   few steps. Contrast SWE-agent (89 steps), which uses `tool_choice:required` to force
   a tool call every turn. We attempted to force `tool_choice:required` on OpenClaw via
   a request-injecting proxy; OpenClaw's gateway did not integrate cleanly through it, so
   the runs here are the model's natural (short) trajectories.
3. **temp = 0.6** (deviation from a temp-0 default). At temp 0 the 32B emits truncated
   narration + EOS *before* the tool call and quits after 1–2 turns — a documented
   degenerate mode for this model in strict-tool-call harnesses.
4. **`web_search` is non-functional** (dummy Brave API key → 422). Web-search tasks
   (e.g. the original `fuzzy_repo` Search task) fail at the tool. Search is therefore
   represented by the **local** `tomllib_trace` task (no web dependency), which worked.

## Takeaway

The CPU character is consistent across all categories regardless of run length:
agentic orchestration on OpenClaw is **GPU-inference-dominated** (~85–90% wall-clock),
with the CPU portion being **frontend+backend-bound interpreter/I-O work and 0% AVX** —
the magnitude (core-seconds) scales with task length, the *character* does not.
