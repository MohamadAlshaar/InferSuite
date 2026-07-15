# Isolated service campaign — figure manifest

Data: `local_service/data_iso/` (36/36 cells, validator 366 OK / 0 FAIL — `local_service/scripts/iso/full_validation.txt`).
Regenerate: `python3 local_service/scripts/iso/plot_service_iso.py` (system python3).
Cell labels: 'Nout' = forced output tokens per request; bracket 'N tokens in' = query length measured
with the model's own tokenizer (vLLM /tokenize; short 9-26, others ~150/~435/~720). Vocabulary identical to the agent set: CPU usage (cores) = core-seconds per second; amounts in
core-seconds; shares in % of CPU time; core = logical CPU; measured partition = 20 (10 x SMT-2).

## Figures (reading order)

- **svc_tier_donuts.png** — MAIN RESULT: per output tier, where a request's wall time goes —
  GPU decoding (host CPU busy-waits) 95/98/99% for tok64/192/320 vs the CPU-side RAG stage
  (all buckets + repeats aggregated, 100 ms classification).
- **svc_system_map.png** — what each pod does: the request's 8-step journey witheach pod's role
  and measured identity (read this first for context).

- **svc_cpu_work.png** — steady-state CPU usage per cell, stacked by pod (4 buckets x 3 tiers,
  mean of 3 repeats). The composition result: vLLM host flat ~1.9 cores (busy-wait), fastapi
  climbs with input bucket and falls with output tier, storage pods are slivers.
- **svc_time_split.png** — GPU work vs CPU-side work share of wall time per cell (100 ms
  classification: engine busy-wait >1 core = GPU decoding; fastapi >0.2 = CPU-side RAG stage).
  Decode owns 93-99% of wall at concurrency 1.
- **svc_signature.png** — per-pod microarch signature by input bucket, absolute scales; median run
  per pod/bucket (no pooling). Tier-invariance verified: within a bucket, IPC varies <=0.03 and
  uop-cache <=6 pp across tiers.
  Three CPU species: vLLM spin loop (IPC 3.59, DSB 99%, else ~0, invariant), fastapi compute
  ladder (IPC 0.59->1.21, packed FP 100%), db pods lean on the OS (milvus/mongo 24-28% OS share).
  Seaweed pods dropped from result figures (usage ~0.002 cores); still counted in stacked totals.
- **svc_tma_uop.png** — TMA L1 + frontend uop delivery; NO pooling — one MEDIAN run per row,
  spread stated in the row label (vLLM ±0.8 pp over 36 runs; fastapi per bucket, its profile
  genuinely varies: be-bound 73->69, uop-cache 73->94 short->very_long).
- **svc_timeline.png** — request rhythm at concurrency 1 (long/tok64, 90 s): fastapi embed
  bursts (~10 cores) exactly in vLLM's dips; storage blips at request boundaries.
- **svc_cost_per_token.png** — core-seconds per 1k engine tokens (prompt+generated) by pod and
  cell; rises with output tier (decode tokens cost busy-wait time, prompt tokens are cheap).
- **svc_dispersion.png** — repetition proof: (max-min)/mean across the 3 identical repeats for
  vLLM/fastapi IPC + usage, all 12 cells.

## Known gaps (supplementary captures possible; k3s still up, no API cost)
- Per-request latency anatomy (embed/milvus/seaweed/decode ms) — loadgen CSVs were deleted with
  the per-cell pods; recoverable with a short rerun + kubectl cp.
- No perf records in service cells — no leaf-frame anatomy (fastapi = the service's "harness")
  and no hw-thread lanes; one record-enabled cell per bucket would add both.
