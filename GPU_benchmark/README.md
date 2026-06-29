# GPU_benchmark — prefill & decode sweeps

A small, self-contained suite that characterises the **GPU** side of the stack
by isolating the two inference regimes. It is independent of the main CPU
benchmark (`scripts/run_benchmark.sh`) — different goal, different tooling.

## What it measures

| Sweep | Knob | Pinned | Isolates | Key signals |
|-------|------|--------|----------|-------------|
| **Prefill** | input ∈ {128…8192} | output = 1 | compute-bound regime | TTFT, DCGM `PIPE_TENSOR_ACTIVE` |
| **Decode** | output ∈ {64…1024} | input = 1 | memory-bound regime | TPOT, DCGM `DRAM_ACTIVE`, KV-cache % |

Each sweep varies one axis and pins the other at its minimum (see the 2-D
picture below), so prefill and decode never contaminate each other.

```
output
  ↑  ● ● ● ● ●        decode sweep (input=1, output varies)
  │  ●
  │  ●─●─●─●─●─●─●    prefill sweep (output=1, input varies)
  └──────────────→ input
   1            8192
```

## Design choices (why it's clean)

- **Talks directly to vLLM** (`/v1/chat/completions`), not the FastAPI
  orchestrator. vLLM natively honours `ignore_eos` + `min_tokens`, so output
  length is **forced exact** — no schema-bypass in `src/`, no RAG, no cache.
- **Single-stream** (concurrency = 1). Batch/concurrency is deliberately out of
  scope for now (it's the cheapest sweep to add later).
- **Passive metrics only** — no CPU `perf` matrix. The CPU is idle here and is
  already characterised by the main benchmark. We scrape:
  - **DCGM-exporter** (`PROF_*` profiling fields) → real GPU hardware behaviour,
  - **vLLM `/metrics`** → KV-cache %, batch size, throughput.
- **Output length is verified**, not assumed: every request records the actual
  `completion_tokens`; a mismatch with the forced N is flagged.

## Files

| File | Role |
|------|------|
| `config.json` | endpoints, model, sweep points, n (env vars override) |
| `run_sweeps.py` | orchestrator: warmup → prefill → decode, manages scrapers |
| `llm_client.py` | streaming vLLM client → TTFT / TPOT / token counts |
| `prompts.py` | builds input prompts of a target token length |
| `prom_scraper.py` | generic Prometheus poller (used for both vLLM and DCGM) |
| `analyze.py` | results → prefill/decode summary tables |
| `deploy/gpu-benchmark-pod.yaml` | EKS pod spec |

## Run on EKS

```bash
# 0. Make sure the GPU stack is up (vLLM serving) and the image is rebuilt
#    so it contains GPU_benchmark/ (Dockerfile.service COPYs it).

# 1. Find the DCGM-exporter endpoint and set it in the pod spec (DCGM_METRICS_URL).
kubectl get svc -A | grep -i dcgm        # e.g. dcgm-exporter.gpu-operator:9400
#    If there is none, the GPU Operator's dcgm-exporter isn't deployed — either
#    install it, or leave DCGM_METRICS_URL empty (latency + vLLM metrics only).

# 2. Deploy the pod (fills in your ECR account/region):
source deploy/config.env
sed -e "s/YOUR_ACCOUNT_ID/$AWS_ACCOUNT_ID/" -e "s/YOUR_REGION/$AWS_REGION/" \
    GPU_benchmark/deploy/gpu-benchmark-pod.yaml | kubectl apply -f -

# 3. Run the sweeps (~20–25 min GPU-active):
kubectl exec -it -n llm-service gpu-benchmark -- \
    python3 /app/GPU_benchmark/run_sweeps.py

# 4. Copy results out and summarise:
kubectl cp -n llm-service gpu-benchmark:/app/GPU_benchmark/results ./gpu_results
python3 GPU_benchmark/analyze.py gpu_results/run_*
```

`--sweep prefill|decode|all` runs a subset.

## Dry-run locally (free, before any GPU spend)

The whole pipeline is testable on minikube against the local 0.5B model — only
the absolute numbers differ, the plumbing is identical:

```bash
kubectl port-forward -n llm-d-local svc/<vllm-svc> 8000:80   # or your local gateway
VLLM_BASE_URL=http://localhost:8000 \
VLLM_METRICS_URL=http://localhost:8000/metrics \
MODEL=qwen2.5-0.5b \
python3 GPU_benchmark/run_sweeps.py --sweep decode
```

## Output layout

```
results/run_YYYYMMDD_HHMMSS/
├── run_info.json          # config + endpoints used
├── all_requests.csv       # every request, all sweeps (discarded flag included)
├── prefill/
│   └── in4096/
│       ├── requests.csv        # per-request TTFT/TPOT/tokens for this point
│       ├── dcgm.csv            # GPU metric time series (if DCGM enabled)
│       ├── dcgm_summary.json   # min/median/max/last per GPU series
│       ├── vllm.csv            # vLLM engine metric time series
│       └── vllm_summary.json
└── decode/
    └── out512/ ...
```

## Notes

- Forcing exact output uses `ignore_eos=true` + `min_tokens=max_tokens=N`,
  honoured because we hit vLLM directly. `analyze.py` reports the *measured*
  token counts so you can confirm the pinning held.
- DCGM `PROF_*` are fractions (0–1). Short prefill points (128–512 tok) are sub-
  second, so their GPU windows are coarse — trust their **TTFT**, not their
  active%. The GPU story is clearest at the long prefill and the decode points.
- The first measured request at each point is **discarded** (transient guard);
  warmup runs once per sweep, before the scrapers start.
