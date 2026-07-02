# Capture provenance (2026-07-02, v2 — the valid run)

- **Load**: in-cluster `loadgen` Deployment (same FastAPI image), `query_runner.py --mode rag
  --size-bucket short --max-tokens 320 --concurrency 6`, continuous loop against
  `http://llm-service-kernel.llm-service.svc.cluster.local:8080` (ClusterIP — no host port-forward).
- **Verified during the perf window** (12:17–12:26 UTC): vLLM `Running: 6 reqs, Waiting: 0` sustained;
  loadgen aggregate **493/493 requests HTTP 200**, `route=rag_plus_backend`, median
  `rag_num_chunks=4`, median `n_output_tokens=282`, median `rag_embed_ms=16.0`,
  `rag_milvus_ms=2.9`, `e2e_ms=7819`. vLLM cgroup task-clock during the 20 s core window:
  **1.957 CPUs utilized** (the engine host threads fully active).
- **Superseded v1** (deleted): the first capture's host `kubectl port-forward` died at 11:56:27
  (`lost connection to pod`), so its load loop got connection-refused for nearly the whole window and
  vLLM was measured mostly idle. v1 attribution was nonetheless indistinguishable from v2
  (~96 % busy-wait) — the spin looks the same loaded or idle — but only v2 is defensible as
  "DURING inference".
- Capture script: `../scripts/service_capture2.sh` run as transient systemd unit `svccap2`
  (perf record task-clock 25 s all 7 pods in parallel; perf stat 5 groups × 20 s on
  vllm/fastapi/milvus/mongodb; `--for-each-cgroup` scoping, PMC-safe ≤6 events/group; no TMA —
  KVM guest lacks the `slots` event).
- **Idle-baseline control** (`idle_control/`, captured same day, loadgen scaled to 0, same cgroup +
  windows): vLLM engine at **0.020 CPUs utilized** (397.68 ms task-clock / 20 s), IPC 0.63,
  attribution = kernel 55 % / libc 18 % / libpython 18 % / asyncio loop 9 % — parked in epoll, **no
  spin**. Loaded-vs-idle = 1.957 vs 0.020 CPUs → the busy-wait is ~100× load-induced (a per-inference
  cost, not an always-on engine tax).
- **Window caveats**: the 5 stat groups per pod are sequential 20 s windows (12:18:50–12:25:47 UTC,
  all inside the verified load window). Per-window cycle spread: vLLM 142.4–143.9 e9 (~1 %, steady);
  FastAPI 9.6–15.2 e9 (fp2 ran ~40 % hotter → its 5.38 GFLOP/s is a single-window estimate, ≈±40 %).
  Milvus's reported 1.8 GHz is P-state down-clocking at a 4 % duty cycle (773 ms task-clock / 20 s).
- **Corpus/deploy evidence**: ingest job log ended `[ok] Ingested 328 papers → 14419 chunks into
  'rag_chunks_seaweed_v2'` (manifest: tenantA, source=vectara/open_ragbench, kb_version=ragbench-v1);
  deployed vLLM args captured in `idle_control/vllm_args.txt` (`--served-model-name
  qwen2.5-32b-instruct --max-model-len 8192 --gpu-memory-utilization 0.92`).
