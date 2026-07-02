# k3s deploy artifacts (reproducibility)

The exact box-side files that turned the minikube-oriented repo deploy into the k3s/H100/32B deploy
measured in `../data/` (pulled from the box before termination):

- `sc-standard.yaml` — `standard` StorageClass aliased to k3s `local-path` (repo PVCs expect `standard`).
- `nvidia-device-plugin.yaml` — NVIDIA device plugin DaemonSet (`runtimeClassName: nvidia`, k3s
  auto-detects the runtime from the host toolkit).
- `modelservice-values.yaml` — llm-d values edited for **Qwen2.5-32B-Instruct bf16** (served name,
  `--max-model-len 8192`, `--gpu-memory-utilization 0.92`, no `--enforce-eager`, 16 CPU / 64 Gi limits).
- `model-pv.yaml` / `model-pvc.yaml` — hostPath PV bumped to 130 Gi (`/data/qwen-model`, 62 G weights).
- `fastapi-configmap.fullstack.yaml` — `GENERATION_MODEL_NAME: qwen2.5-32b-instruct`.
- `deploy_llmd_local.sh` / `deploy_fullstack_single_node.sh` — de-minikube'd (no docker-env/minikube
  reqs; decode deployment patched with `runtimeClassName: nvidia`); FastAPI image loaded via
  `docker save | k3s ctr images import -`.
- `ragbench-ingest-k3s.yaml` — ingest Job (local image, `--n-filler 0` → exactly the 328 qrel-required
  papers), `loadgen.yaml` — the in-cluster load Deployment used for the measured window.
