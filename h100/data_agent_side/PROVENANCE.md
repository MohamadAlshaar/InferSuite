# H100 agent-side campaign — provenance

Captured 2026-07-03 on a single Lambda H100 (KVM guest, 26 vCPU, Ubuntu 22.04), fresh instance
192.222.55.68. vLLM 0.24.0 bare-metal serve (no k8s), engine in a systemd user scope.
Models: Qwen/Qwen2.5-Coder-32B-Instruct (swe, bcb) and Qwen/Qwen2.5-32B-Instruct (oc-*),
served at 32K ctx with --enable-auto-tool-choice --tool-call-parser hermes (July serve config
modified: --disable-log-stats removed so /metrics gauges exist; 16K -> 32K ctx or OpenClaw
aborts with context overflow).

Per workload (agent_side_chain.sh): three cgroups in the SAME windows — engine scope, agent
harness scope, tool container — portable counter groups stats-first (core/cache/mlp/fp1/fp2 x
10 s; NO TMA: topdown events do not exist on the KVM guest), three parallel task-clock records,
nvidia-smi 2 Hz GPU timeline from work-guard (vllm:num_requests_running >= 1) to agent exit.

Agent-work evidence (all VALIDATE-OK, 5/5 groups in-window):
- swe: fc sweagent + guided tool_choice on astropy-14096, 24 steps, autosubmitted patch
- bcb: 9/12 solved, 20 tool-exec runs (vs 0/12 at 7B locally)
- oc-calendar/web/pdf/crop: real episodes, 100-452 GPU samples; harness outputs (chat.jsonl,
  gateway.log, scores) under oc_harness_output/
Known blemish: oc-crop group_fp2 engine rows <not counted> (engine idle after episode end);
agent-side rows counted.

Rerun 2026-07-03 (audit): SWE x3 recaptured with max_input 28000 / max_output 4096 (was 14000/2048,
a leftover of the 16K serve config; episodes doubled-to-8x in length: astropy 188, scikit 702, sympy
332 gpu-samples) and 8 s stat groups. BCB uses HEAVY_LIBS=1 numeric tasks. Finding upheld by the
longer episodes: live Coder-32B performs ZERO test executions on SWE (scikit: 54 steps, 0 python/
pytest invocations) — tool-side packed FP is genuinely zero; the BLAS-heavy verification phase exists
only in the canonical (Claude-driven) trajectories. GPU sampler cadence ~1.8 Hz (shares unbiased).
