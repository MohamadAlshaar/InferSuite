# Where Does the Compute Go?
### A CPU-Centric Characterization of an End-to-End GenAI System

This repository is the artifact for a master's thesis that asks one question of a real,
production-style GenAI system: **where does the compute actually go — and what is the CPU doing?**

It contains three things, and they form one story:

1. **The Service** — a deployable RAG + semantic-cache + vLLM chatbot on Kubernetes (managed cloud cluster or local minikube).
2. **The Benchmark Suite** — a load generator + microarchitectural profiler that measures that service.
3. **The Agentic Workloads** — three agents (SWE-agent, BigCodeBench, OpenClaw) profiled the same way.

**The central goal** is to characterize what the CPU does **DURING inference** (the vLLM serving engine)
versus **OUTSIDE inference** (RAG retrieval, semantic-cache lookup, and agentic tool execution), across
every workload, on two axes simultaneously: **wall-clock time** and **CPU core-seconds**.
The **GPU is measured in lockstep** — a full GPU top-down via **Nsight Compute** (a Speed-of-Light roofline plus
*two* warp-scheduler TMAs: the native warp-state breakdown and an Intel-style Retiring/Frontend/Backend mapping)
— so every workload is profiled on **both engines**, and the CPU's role is always read against what the GPU is doing.

Hardware: a local box — **Intel Xeon w5-3425** (Sapphire Rapids, AMX/AVX-512) + **NVIDIA RTX A2000**
(Ampere) — with cloud GPU runs (H100 / L40S) for the parts the local GPU can't measure.

---

## Repository map

```
src/service/            FastAPI orchestrator: semantic cache, RAG, embeddings, observability
deploy/                 Kubernetes manifests, Helm charts, Kustomize overlays (cloud + minikube)
scripts/                Deploy, RAG ingest, benchmark, and report-generation scripts
fastapi_runtime_assets/ Embedding model + tokenizer + seed RAG data (bundled into the image; gitignored)

benchmark_results/run_20260609_140052/   CANONICAL service benchmark (tok64+tok192+tok320)
thesis_plots/                              service-level figures (latency, throughput, cache-hit)
inf_thesis_plots/  (+ gpu/)                serving-side CPU TMA / roofline + the two GPU TMAs

agentic/                The three agentic workloads + measurement harness
  CANONICAL/              single source of truth for the 3 agent benchmarks (microarch.py, *.json)
  common/                 shared perf harness (perf_events.sh, lib_perf.sh, parse_perf.py, microarch.py)
  swe_agent/              SWE-bench code-repair agent (file-I/O tool-exec)
  bigcodebench/           code-gen + execute loop (numeric / FP tool-exec)
  openclaw/               browser / general web agent
  inference/              phantom-CPU experiment (cudasync/), entropy-UQ, GPU-TMA build
  aws_agents/, cloud/     cloud-run results (GPU ncu data) + launch scripts
```

> **Note on size:** large re-creatable artifacts (Python venvs, cloned upstream agent repos, model
> weights, scratch run outputs) are **gitignored** — only ~120 MB of code + canonical results is tracked.
> Recreate them with the per-workload `setup`/`pip install` steps.

---

# Part I — The Service (a "normal chatbot")

A production-style GenAI inference stack deployable on **any Kubernetes cluster** — a managed cloud cluster or a local **minikube** GPU machine.

```
Client
  │
  ▼
FastAPI Orchestrator  (src/service/orchestrator/chat.py)
  ├─► Semantic Cache check   (BGE embed → Milvus → MongoDB)        ~50–100 ms on hit
  ├─► RAG Retrieval          (BGE embed → Milvus → SeaweedFS)
  └─► LLM Inference          (llm-d Istio gateway → vLLM decode pod)
```

The embedding/cache model is **bge-base-en-v1.5**; the generation model is configurable
(default Qwen2.5-14B, also run at 32B). All inter-service traffic is in-cluster (`*.svc.cluster.local`).

## Deploying it — two scripts, one config

Everything is controlled by **one file**, `deploy/config.env` (`DEPLOY_ENV=eks | minikube`).

### Cloud Kubernetes
```bash
# one-time prep: a Kubernetes cluster + a container registry + a kubectl context
cp deploy/config.env.example deploy/config.env     # fill in cluster, registry, region, model
# the two scripts
./setup.sh      # kubeconfig, StorageClass, node pools, build + push the FastAPI image to your registry
./deploy.sh     # storage → FastAPI → vLLM (model download ~10 min) → RAG ingest → health
# chat
python3 scripts/chat_cli.py --base-url http://localhost:18081 --model qwen2.5-14b --tenant tenantA
```
Nodes: one **CPU node** (storage + FastAPI) + one **GPU node** (vLLM) + a small system node. GPU nodes are
auto-provisioned; pause/resume by scaling node pools to 0 (`scripts/pause_*.sh` / `resume_*.sh`) — persistent
volumes are retained.

### Minikube (local GPU)
```bash
./setup.sh      # installs deps + NVIDIA toolkit, downloads models, starts minikube
./deploy.sh     # full stack + RAG ingestion
python3 scripts/chat_cli.py --show-debug
```

### Configuration — one file: `deploy/config.env`
Copy `deploy/config.env.example` → `deploy/config.env` and edit. The knobs you'll touch most:

**Model / serving**
| variable | default | controls |
|---|---|---|
| `MODEL_NAME` | `qwen2.5-14b` | service-facing model id (change *together* with the repo below) |
| `MODEL_HF_REPO` | `Qwen/Qwen2.5-14B-Instruct` | HuggingFace repo downloaded to the model volume |
| `MODEL_MAX_LEN` | `32768` | max context length (14B/32B = 32768, 0.5B = 2048) |
| `MODEL_GPU_MEM_UTIL` | `0.90` | vLLM GPU-memory fraction |
| `MODEL_STORAGE_GI` | `40` | model-weights volume size, Gi (14B ≈ 30, 32B ≈ 70) |
| `MODEL_REPLICAS` | `1` | number of vLLM decode workers (one GPU node each) |
| `GPU_COUNT` *(tensor-parallel)* | `1` | GPUs per replica → wired to vLLM `--tensor-parallel-size` in `deploy/llmd-eks/modelservice-values.tpl.yaml` |

**Cluster**
| variable | default | controls |
|---|---|---|
| `DEPLOY_ENV` | `eks` | deploy target (cloud cluster or `minikube`) |
| `NAMESPACE_SERVICE` | `llm-service` | FastAPI + storage namespace |
| `NAMESPACE_LLMD` | `llm-d` | vLLM + gateway namespace |

Cloud-target details (registry, region, node machine types) live alongside these in `config.env.example`.

**To switch model or tensor-parallelism:** edit the model rows (and `GPU_COUNT` for TP), then re-run `./deploy.sh`
— it re-downloads weights, re-renders the vLLM Helm values, and restarts vLLM. For RAG docs: add PDFs and run
`FORCE_REINGEST=1 ./deploy.sh`.

### What actually runs in Kubernetes

**Cloud — one cluster, namespaces split across nodes:**
```
Cluster
├─ CPU node    [llm-service + istio-system]
│    FastAPI orchestrator · Milvus (+ etcd + minio) · MongoDB · SeaweedFS · Istio istiod
├─ GPU node    [llm-d]
│    vLLM (the generation model) · llm-d routing sidecar + Istio gateway
└─ small node  [kube-system]   metrics-server, coredns
```
**Minikube** runs all of the above on a single node across namespaces `istio-system`, `llm-d-local`, `llm-service`.

All component images are standard upstream (Milvus, MongoDB `mongo:8`, SeaweedFS, llm-d/vLLM
`ghcr.io/llm-d/llm-d-cuda`); only **FastAPI** is built here (`Dockerfile.service`).

### Why the deploy is simple (it's all declarative)
There is **no manual container management** — every service is a Kubernetes manifest, Helm chart, or Kustomize
overlay, and the whole stack comes up from **two scripts driven by one config**:
- **Kustomize base** (`deploy/k8s-storage/base/`, `deploy/k8s-fastapi/base/`) = minikube-ready defaults.
- A **cloud overlay** patches *only* what differs — the block-storage `storageClassName`, the image-registry URL,
  the `nodeSelector`, and region/cache scope. The base files are never edited per-environment.
- Helm charts for llm-d/vLLM are **vendored** in `deploy/llmd-local/` (no network fetch at deploy time).
- Cloud GPU-node quirks (driver/library paths, SELinux read-write model volume, CUDA compatibility) are applied
  automatically by the cloud deploy script.

### What happens on a fresh deploy
`./setup.sh` (run once, idempotent): verify tools → kubeconfig + cluster access → create the block-storage
StorageClass → create/patch node pools (CPU + GPU) → build & push the FastAPI image to your registry.

`./deploy.sh`:
1. Apply the **storage** stack → the CPU node comes up → bootstrap Job creates Milvus collections + the SeaweedFS bucket.
2. Apply the **FastAPI** stack (Deployment, ConfigMap, Service, PVCs).
3. The cloud deploy script: install Gateway API CRDs + Istio (Helm) → create the model PVC and run the
   **model-download Job** (HuggingFace → block volume, ~10 min for 14B) → deploy **vLLM via Helm** → patch the
   FastAPI ConfigMap with the live model name + gateway URL.
4. Run the **RAG ingestion** Jobs (PDF download → BGE embed → Milvus + SeaweedFS).
5. Port-forward FastAPI to `localhost:18081` and print a health summary.

(Minikube follows the same `./setup.sh` → `./deploy.sh` flow on one node; the model is provisioned to the node
filesystem instead of a network volume.) All tunables live in **`deploy/config.env`** (see the table below).

---

# Part II — The Benchmark Suite (measuring the service)

A load generator that drives the service through realistic paths and records per-request metrics **and**
host-level CPU counters, so each path can be attributed to GPU-generation vs. CPU-side work.

**Benchmark paths** (the workload axis):
| path | dataset | isolates |
|---|---|---|
| RAG-standard | open_ragbench | full pipeline (retrieve + generate) |
| RAG pure-fetch | bare questions | retrieval cost alone |
| Semantic-cache | QQP pairs | cache embed + lookup |
| LLM-direct | ShareGPT52K | generation alone |

**Token tiers:** `tok64 / tok192 / tok320` with **exact output-token forcing** (`ignore_eos` + `min_tokens`)
so generation cost is comparable across runs. Flags: `RAG_FORCE_SEAWEED_FETCH=1`, `HISTORY_ENABLED=0`,
`GENERATION_API_MODE=chat`.

**Harness:** `scripts/run_benchmark.sh` (orchestrator) → `query_runner.py` (sends requests, per-request
metrics) → `generate_report.py` (HTML report) / `results_cli.py` (terminal summary). Stability sweep:
`scripts/analyze_stability.py`.

**Canonical results:** `benchmark_results/run_20260609_140052/` (the authoritative merged run) +
`stability_report.html`. Service-level figures in `thesis_plots/`.

---

# Part III — The Agentic Workloads

Three agents, deliberately chosen to span a **tool-execution spectrum** (different *kinds* of CPU work
outside inference). Single source of truth: `agentic/CANONICAL/`; shared 6-pass perf harness: `agentic/common/`.

| workload | what it is | tool-exec CPU character (measured) |
|---|---|---|
| **SWE-agent** (`swe_agent/`) | SWE-bench repo bug-fix, external SWE-agent harness | file nav / edit / search — **FE-bound, IPC ~1.5, ~0 AVX** (I/O + control) |
| **BigCodeBench** (`bigcodebench/`) | own driver `agentic_bcb.py`: generate → run test → fix loop | executing real numeric Python — **heavy (~3 cores), real FP**, FE+BE-bound IPC 1.08 |
| **OpenClaw** (`openclaw/`) | live web/browser tasks (WildClawBench) | browser / playwright — heavy, irregular |

> **Why two harnesses?** SWE-bench is repo-shaped (navigate a codebase); BCB is function-shaped (one
> function + tests). BCB uses a thin instrumented driver so the test runs **every turn** — guaranteeing
> tool-exec CPU fires and is cleanly markered. This is a *characterization of diverse real workloads*, not
> a single-harness controlled ablation; the tool-exec *character* is task-driven, not harness-driven.

---

# Part IV — Measurement Methodology (the CPU/GPU microarchitecture)

### CPU — host-level, whole-pod
`perf` is scoped to the **entire pod cgroup** (`-G kubepods…`), not just the API-server PID. This fixes the
**vLLM CPU-undercount bug**: the API process reads as ~idle (~0.02 cores) because the real work is in the
`VLLM::EngineCore` worker — perf must match it by cgroup. Four counter passes (IPC/uops/branches; cache
hierarchy; stalls/TLB; DRAM + FP) + **TMA via `toplev -l2`** (`td2`, fixed counters), all pods in parallel.
Agentic side adds AMAT/MPKI, arithmetic-intensity + CPU-roofline, AVX%/FMA-aware FLOPs, cgroup-scoped DRAM.

### GPU — Nsight Compute (ncu)
- **Speed-of-Light roofline** (`inf_thesis_plots/gpu/gpu_01`): compute% vs memory% of peak.
- **Two top-downs over warp-scheduler issue slots:**
  - **Native warp-state TMA** (`gpu_02`) — `smsp__average_warps_issue_stalled_*_per_issue_active.ratio`,
    duration-weighted per kernel: Issued / latency-hidden / math-pipe / memory / sync / fetch-branch.
  - **Intel-mapped TMA** (`gpu_05`) — the same data re-binned into Retiring / Frontend / Backend{Core,Memory},
    with Bad-speculation marked N/A (GPUs don't speculate), for direct CPU↔GPU comparison.
- Prefill profiling requires `enable_prefix_caching=False` + a distinct warmup, or "prefill" is a 1-token
  cache hit and inverts the conclusion.

### Phantom-CPU (the signature finding)
During decode the vLLM engine host thread **busy-waits on `cuEventSynchronize` at IPC ~3.4 doing zero work**
(it even reads as "retiring-bound"/healthy in TMA). An `LD_PRELOAD` shim (`agentic/inference/cudasync/`)
forcing `cudaEventBlockingSync` makes it **sleep instead of spin**, recovering **~76% of the engine CPU**
(420.6 → 99.9 core-seconds on a live SWE-agent run) at **no latency or throughput cost**.

### Platform gotchas
Local bare-metal Sapphire Rapids has a full PMU (`sudo perf`, `perf_event_paranoid=-1`); the **cloud GPU
instance exposes no CPU PMU** (so vLLM-during-inference TMA is local-only). Kill stale root `perf -a` orphans before runs.

---

## Key findings

- **The serial alternation.** The agent loop is GPU-busy/CPU-idle (decode) then CPU-busy/GPU-idle (tool) —
  the two engines rarely overlap.
- **The phantom core** — a whole CPU core spins on GPU sync doing nothing; reclaimable for free (above).
- **Where generation compute goes** (code-agent, token-weighted): **35% semantic reasoning / 34% boilerplate /
  28% delegated code / 3% exact-simulated** — exact work is already offloaded to tools.
- **What the idle CPU *cannot* do** (measured negatives that bound the design space): co-compute the matmul
  (1.05×), run a draft model (37 < 45 tok/s), or fill the tool-window with prefill (causality). The CPU
  cannot be a second *compute* engine on commodity hardware.
- **What it *can* do for free** — compute uncertainty (Shannon entropy) in the decode shadow at ~zero
  marginal cost; the signal is informative (beats a random baseline) and can gate adaptive compute.

---

## Reproducing

Service benchmark: deploy (Part I) → `scripts/run_benchmark.sh` → `scripts/generate_report.py`.
Agentic: per-workload `run_*.sh` under `agentic/{swe_agent,bigcodebench,openclaw}/` (recreate venvs/clones
first); shared profiling via `agentic/common/`. Figures regenerate with system `python3` (matplotlib) from
the collected JSON — collection and plotting are separate steps.
