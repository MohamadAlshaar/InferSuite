# LOCAL service run — RAG path, 3 token tiers, full PMU (campaign A extension)

The full InferSuite service on the local ThinkStation P7 (Xeon w5-3425, RTX A2000 12GB) under
single-node k3s, serving **Qwen2.5-7B-Instruct-AWQ** on vLLM 0.24 (llm-d image) — the one machine
where the serving engine and a full Intel PMU coexist. Measures ALL pods, during + outside
inference, per output tier (tok64/192/320) under continuous verified RAG load + one idle control:

- **(A) attribution** — `perf record -e task-clock` per pod cgroup (software view)
- **(B) microarch** — 5 portable CANONICAL groups **+ 2 TMA groups** (topdown L1 + td2), the
  measurement neither cloud campaign could take (EKS GPU node: no PMU; H100 KVM guest: no slots)

Config choices: loadgen concurrency 1 (sequential requests, matching the benchmark protocol), `VLLM_FORCE_EXACT_TOKENS=1`
(controlled decode length per tier), `RAG_FORCE_SEAWEED_FETCH=1` (object store does real per-query
work, like the EKS campaign pod set).

## Run order
1. `scripts/setup_local_k3s.sh`   — k3s + nvidia + model staging + FastAPI image (once)
2. `scripts/deploy_stack.sh`      — storage, istio/llm-d (7B values), FastAPI, corpus ingest (once)
3. `scripts/capture_tiers.sh`     — the measurement (tok64/192/320 + idle) -> `data/`

Uses a dedicated kubeconfig (`~/.kube/k3s-local.yaml`) — does not touch EKS config.
Values/manifests in `llmd-local/` (7B) + `k3s_deploy/loadgen-tier.yaml`; charts symlinked from
`deploy/llmd-local/`. Teardown: `/usr/local/bin/k3s-uninstall.sh`.
