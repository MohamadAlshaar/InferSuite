# SWE-agent case study — LLM↔code-exec loop microarch

**Goal (not a stack benchmark):** characterize the *agentic loop* — alternating
**inference phase (vLLM)** ↔ **test/build phase (pytest in a Docker sandbox)** —
phase-segmented. SWE-agent does NOT touch the RAG storage pods; only vLLM is used.

## Wiring (SWE-agent → local vLLM)
SWE-agent uses LiteLLM → point it at an OpenAI-compatible endpoint = our vLLM.

1. Port-forward the local vLLM (minikube) to the host:
   ```
   kubectl --context minikube port-forward -n llm-d-local svc/ms-local-decode-direct 8000:8000
   ```
2. Configure SWE-agent model:
   - model: `openai/qwen2.5-0.5b`   (0.5B = plumbing test only; real runs = 14B on H100)
   - `OPENAI_API_BASE=http://localhost:8000/v1`
   - `OPENAI_API_KEY=dummy`

CONFIRMED 2026-06-17: local vLLM serves at localhost:8000 (`/v1/models` → qwen2.5-0.5b,
max_model_len 2048; test chat completion OK).

## Dataset
SWE-bench **Lite** from HuggingFace (`princeton-nlp/SWE-bench_Lite`); 1–2 instances for
the wiring test. SWE-agent spins a Docker container per task to apply the patch + run tests.

## Measurement (added after wiring works)
- perf binary on this host: `/usr/lib/linux-tools-6.8.0-124/perf` (the `/usr/bin/perf`
  wrapper is broken for kernel 6.17 — installed build is 6.8.12, works cross-kernel).
  `perf_event_paranoid=-1`, no sudo needed. TMA topdown + PEBS + FP all verified working.
- toplev `-l2` needs pmu-tools (not yet installed — `git clone andikleen/pmu-tools`).
- **Phase-segment**: timestamp each LLM call vs each pytest run; slice perf/toplev into
  inference-phase vs build-phase windows. Co-location knob: agent sandbox same node as
  vLLM (contention) vs separate.

## Caveat
0.5B will likely fail the actual SWE-bench tasks (too weak for valid patches / tool calls);
that's fine — wiring/plumbing validation only. Real solve + final microarch = strong model
on H100.

## Repo hygiene
External SWE-agent clone + venv + datasets + run outputs live under `external/`, `.venv/`,
`data/`, `runs/` — all gitignored (see ../.gitignore). Only this README + our config/run
scripts are tracked.
