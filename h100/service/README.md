# Service (RAG) CPU characterization — DURING vs OUTSIDE inference (H100 · k3s)

The **full InferSuite service** brought up on a single H100 box and profiled on the **RAG path at tier
tok320**, to measure the one thing the EKS deployment never could: the **CPU *during* inference** (the
EKS GPU node had no PMU). This is the service analogue of the agentic two-view — *during* inference
(the vLLM serving engine) vs *outside* inference (retrieval / embedding / routing / storage) — but now
across the **real multi-pod service graph**.

## Deployment (faithful, not minikube)
Real single-node **k3s** (not minikube's nested-in-docker — a dedicated H100 is not a "weak machine";
k3s gives a real kubelet, host-flat pod cgroups, and clean NVIDIA device-plugin GPU). Stack:

- **llm-d** gateway (Istio/**envoy**) → **vLLM 0.24** serving **Qwen2.5-32B-Instruct (bf16)** on the H100
  (`runtimeClassName: nvidia`, CUDA graphs / no `enforce_eager` = realistic serving, gpu-util 0.92).
- **FastAPI** orchestrator (semantic-cache + RAG), **BGE-base-en-v1.5** embedding on **CPU**.
- **Milvus** (+etcd+minio), **MongoDB**, **SeaweedFS** (master/volume/filer/s3).
- Corpus: `vectara/open_ragbench` — **328 papers → 14,419 chunks** (exactly the papers the benchmark
  queries' qrels reference), ingested into Milvus. RAG verified live (`route=rag_plus_backend`,
  retrieved context injected into the prompt).

## Method (`scripts/service_capture2.sh` — a NEW script; `run_benchmark.sh` untouched)
Learns the pod set from `run_benchmark.sh` but fixes the **vLLM CPU-undercount bug**: instead of
`perf stat -p <pid>` (misses the many-threaded EngineCore), it scopes **`perf … --for-each-cgroup
<pod cgroup>`** (whole container, cgroup-v2 path resolved via `crictl → /proc/<pid>/cgroup`).

**Load** (see `data/PROVENANCE.md`): an **in-cluster loadgen Deployment** drives
`query_runner.py --mode rag --max-tokens 320 --concurrency 6` continuously against FastAPI's ClusterIP.
Verified during the perf window: vLLM **`Running: 6 reqs` sustained, ~2.0 CPUs on the engine cgroup**;
loadgen **493/493 HTTP 200**, median 4 chunks / 282 output tokens per request. (A first capture that
drove load through a host `kubectl port-forward` was **discarded** — the forward died mid-run and vLLM
was measured mostly idle; its files were deleted.)

Per pod, during that steady load:
- **(A) Attribution** — `perf record -e task-clock -g -F199` (software event, no PMU contention), one
  25 s window, all 7 work pods in parallel → `perf report` **self%** by DSO/symbol (`<pod>_flat.txt`,
  `<pod>_dso.txt`).
- **(B) Microarch** — `perf stat` **CANONICAL** groups (`core / fp1 / fp2 / cache / mlp`, ≤6 events →
  no multiplexing) on vLLM(INSIDE) + FastAPI/Milvus/MongoDB(OUTSIDE) → `group_<pod>_<grp>.txt`,
  derived via `agentic/CANONICAL/microarch.py`. **No TMA** — this KVM guest lacks the `slots` PMU
  event (same limitation as the agentic H100 run).

## Findings

### `service_attribution.png` — where each pod's CPU goes (self%)
| pod | class | CPUs util. | dominant CPU (self %) |
|---|---|---|---|
| **vLLM engine** | INSIDE | **1.96** | **73 % `libcuda` + 23 % `[vdso]` ≈ 96 % busy-wait** (`cuEventSynchronize` spin; the vdso clock-poll is driver-spin and/or engine-loop polling — split not resolvable from flat output), 2 % Python |
| **FastAPI + BGE** | OUTSIDE | 0.17 | 60 % `libgomp` (OpenMP thread-pool) + **32 % `libtorch_cpu` `mkl_avx512_sgemm`** (the BGE-embedding GEMM) + 5 % Python |
| **Milvus** | OUTSIDE | 0.04 | 60 % `milvus` self — mostly Go-runtime scheduler/GC; the search itself ≈3 ms/query — + 33 % kernel + 5 % jemalloc |
| **MongoDB** | OUTSIDE | 0.007 | 71 % `mongod`, but ~88 % of samples are its FTDC self-telemetry loop, not RAG-path serving |
| **llm-d gateway (envoy)** | ROUTING | ≈0 | 3 samples total (~15 ms CPU) — effectively idle at this request rate; split not statistically meaningful |
| **SeaweedFS** | OUTSIDE | ≈0 | chunk text is served from the inline Milvus `text` field (`milvus_rag.py` only fetches from S3 when the field is empty or `RAG_FORCE_SEAWEED_FETCH=1`), so the object store is not touched per query |

**The two-view holds for the service:** *during* inference the vLLM host does **no real work** — ~2 full
cores spin at IPC 3.2 inside `libcuda`/vdso waiting on the GPU (the "phantom CPU", identical in character
to the agentic study, now on the self-hosted service engine **under verified full load**, at this
operating point: 6-way concurrency, ~282-token generations).

**Idle-baseline control** (`data/idle_control/`, same cgroup + windows with the loadgen scaled to 0):
the engine drops to **0.020 CPUs utilized** (IPC 0.63, parked in kernel/epoll + asyncio — no spin). So
the ~2-core busy-wait is **~100× load-induced, a per-inference cost, not an always-on engine tax** —
the strongest form of the claim: serving *creates* two cores of pure spin.

Honest magnitudes for the OUTSIDE pods (CPUs utilized over their stat windows): at this operating point
retrieval is *cheap* — FastAPI 0.17 CPUs (the only substantial payload work: the BGE AVX-512 GEMM),
Milvus 0.04 CPUs (its 60 % self is mostly Go-runtime scheduler/GC churn; the search itself is ~3 ms per
query), MongoDB 0.007 CPUs (its profile is ~88 % FTDC self-telemetry, not RAG-path serving), envoy ≈0
(3 samples — effectively idle at this request rate). That *is* the two-view: the engine burns ~2 cores
doing nothing while the entire retrieval path needs a fraction of one core of real work.

### `service_microarch.png` — microarch heatmap
| | IPC | L1 % | L2 % | L3 % | LLC-MPKI | AMAT | MLP | ILP | vec %FP | GFLOP/s |
|---|---|---|---|---|---|---|---|---|---|---|
| **vLLM (INSIDE)** | **3.20** | **99.95** | 0.04 | 0.01 | **0.00** | 5.0 | 1.54 | **3.12** | 0 | 0.08 |
| **FastAPI (OUTSIDE)** | **0.44** | 92 | 7.5 | 0.3 | 1.04 | 6.8 | **3.41** | 0.57 | **99.98** | **5.38** |
| **Milvus (OUTSIDE)** | 0.81 | 97 | 2.5 | 0.5 | 0.66 | 6.1 | 1.68 | 0.94 | 99 (tiny abs) | 0.00 |
| **MongoDB (OUTSIDE)** | 1.08 | 98 | 1.3 | 0.2 | 0.57 | 5.8 | 1.79 | 1.21 | 0 | 0.00 |

DURING inference the engine is a **pure register/L1 spin**: IPC 3.20, L1 99.95 %, **negligible LLC misses
(MPKI 0.0009), zero packed FP**, ILP 3.1 — ~2 cores of completely wasted, "healthy-looking" CPU. OUTSIDE,
the CPU does real work at *low* IPC: FastAPI runs the BGE embedding (**~100 % packed AVX-512,
5.4 GFLOP/s**, MLP 3.4 — streaming GEMM) between Python orchestration; Milvus's search is L1-resident
SIMD distance compute; MongoDB is integer/pointer-chasing.

### Notes
- Milvus `vec≈99 %` is a *ratio* on a tiny absolute FP volume (graph search does few distance FLOPs
  here) — hence GFLOP/s ≈ 0. Its low reported clock (1.8 GHz) is P-state down-clocking on a 4 % duty
  cycle, not a broken counter.
- vLLM's 0.08 GFLOP/s is scalar-single from the sampler — negligible; the model math is all on the GPU.
- The 5 stat groups per pod run **sequentially** (20 s each). vLLM's five windows agree within ~1 %
  (IPC 3.20–3.21), so the INSIDE numbers are steady-state. FastAPI's FP windows are burst-sampled —
  its fp2 window ran ~40 % hotter than its siblings — so **5.38 GFLOP/s is a single-window estimate
  (≈±40 %)**; the vec % ratio is robust.
- Coverage boundary: the donuts/heatmap are **per-pod-cgroup**; etcd, minio, seaweed master/s3, the
  loadgen pod, and host-side CPU (k3s/kubelet, containerd, kernel softirq networking) are not profiled.
  OUTSIDE totals are therefore a per-pod lower bound (all of these are near-idle at this rate).

## Files
- `scripts/service_capture2.sh` — the capture (perf-only; load from the in-cluster loadgen Deployment).
  `scripts/service_capture.sh` — v1 (kept for the record; its port-forward load method is what failed).
- `scripts/plot_service_attribution.py`, `scripts/plot_service_microarch.py` — figure generators (system python3).
- `data/` — `<pod>_flat.txt`, `<pod>_dso.txt` (attribution), `group_<pod>_<grp>.txt` (microarch),
  `PROVENANCE.md` (load verification for the window).
- `data/idle_control/` — the idle baseline (loadgen=0): `group_vllm_idle_core.txt` (0.020 CPUs),
  `vllm_idle_flat.txt`/`_dso.txt` (epoll-parked, no spin), plus the deployed vLLM args
  (`vllm_args.txt`: 32B served name, max-model-len 8192, gpu-memory-utilization 0.92).
- `plots/` — `service_attribution.png`, `service_microarch.png`.
