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
- bcb: HEAVY_LIBS=1 numeric tasks (driver = agentic/bigcodebench/agentic_bcb.py), final data 8/12 solved, 24 tool-exec runs, 60 s records
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

Audit addenda (two independent auditors, 2026-07-03): all figure numbers reproduce from raw
counters; scope attribution collision-free; engine invariance holds (IPC 2.98-3.18, L1>98%).
Caveats: (a) oc-web and oc-pdf episodes made ZERO tool calls (single-turn monologues) - their
container CPU is the OpenClaw runtime, labeled agent+tools/(no tool calls), excluded from the
delegated-work donut; all four OC scores are 0.00; (b) OC "harness" scope = near-idle run_batch
orchestrator (1M loads/window) - excluded from the signature heatmap as statistically thin;
(c) GPU timelines are clipped of the post-episode teardown tail (the sampler follows the driver
pid; oc-crop uncorrected read 40/60, in-episode truth ~94/6); (d) usage.json token counters are
all zero via the vLLM proxy (only elapsed_time valid); (e) engine busy-wait self-share 67-69%
(84-87% incl. vdso clock polling) vs July's 72.5%/96% - different serve config (hermes parser,
log-stats on); (f) BCB packed-FP is <3% in-window, quantitatively consistent with a standalone
re-execution of the same 12 tests (0.82% packed) - "uses numpy/pandas" does not imply packed FP
at these test sizes (calibrated: the same method sees 6.8e11 packed ops/8s from one numpy matmul).

Second calibration (on-box, 2026-07-03): numpy matmul inside a docker scope ON THE H100 GUEST,
same testbed conda python, same method: 39.6e9 128b + 37.7e9 256b + 431e9 512b packed double ops
in 8 s (vec ~99.9%) vs 0 packed during live episodes on the same cgroup type. Packed-FP counting
is proven end-to-end on this exact box/cgroup path; the episode zeros are workload behavior.
