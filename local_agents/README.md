# Local tool-execution software views (campaign A completion)

Per-workload SOFTWARE VIEW (perf record task-clock) of the agent tool-execution phase on the local
workstation, scope-matched to the CANONICAL counter/TMA measurements:

- **Code agents = deterministic replays of the frontier-API-driven runs** (no model in the loop):
  BCB replays the recorded programs (the campaign's `executed.jsonl` log, local data) natively in the bcb venv;
  SWE replays each recorded trajectory (`sweagent run-replay`) with perf scoped to the sandbox
  container cgroup (DSO view needs no symfs; flat view via a twin container of the same image).
- **OpenClaw = live** (browser agents cannot be replayed), driven by the frontier API model through a
  local litellm proxy (`run_oc_chain.sh` manages the proxy; key read from ~/.anthropic_key, never
  logged), perf record on the task container (agent + all tools), 220 s cap past the warmup fence.

Results (DSO self%): BCB 68% BLAS · astropy 52% kernel + 36% Python (command churn, trajectory
scope) · scikit-learn 88% OpenBLAS (counters said 89% AVX-512 on the identical replay) · sympy 73%
CPython · OpenClaw 70-78% Node/V8 (image-crop: +35% Python doing the image work).

`scripts/` — capture + plot; `data/` (gitignored) — perf.data + flat/dso reports per workload;
`plots/tool_attribution.png` — the 8-donut figure (thesis palette).
