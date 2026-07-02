#!/usr/bin/env bash
# deploy_stack.sh — phase 2 of the LOCAL service run: full stack on local k3s via the k3s-flavored
# fullstack driver (storage, istio/gateway, llm-d + vLLM 7B-AWQ, FastAPI), then patch the FastAPI
# config for the local model + exact-token forcing + forced SeaweedFS fetch, and run the corpus
# ingest (328 open_ragbench papers).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
H100_DEPLOY="$REPO/h100/service/k3s_deploy"
export KUBECONFIG="$HOME/.kube/k3s-local.yaml"
log(){ printf '[deploy] %s\n' "$*"; }

# --- full stack (image already built+imported in phase 1; no minikube anywhere) ---
PREPARE_ASSETS=0 BUILD_FASTAPI_IMAGE=0 PROVISION_MODEL=0 RUN_VALIDATE=0 \
  bash "$REPO/scripts/deploy_fullstack_single_node.k3s.sh"

# --- local config: model name, exact tokens, force object-store fetch on the RAG path ---
log "patching FastAPI configmap for the local run"
kubectl patch configmap llm-service-kernel-config-fullstack -n llm-service --type=merge -p '{
  "data": {
    "GENERATION_MODEL_NAME": "qwen2.5-7b-instruct-awq",
    "VLLM_FORCE_EXACT_TOKENS": "1",
    "RAG_FORCE_SEAWEED_FETCH": "1"
  }
}'
kubectl rollout restart deployment/llm-service-kernel -n llm-service
kubectl rollout status deployment/llm-service-kernel -n llm-service --timeout=300s

# --- corpus ingest (exactly the 328 qrel-referenced papers -> 14,419 chunks) ---
if ! kubectl get job ragbench-ingest -n llm-service >/dev/null 2>&1; then
  log "running open_ragbench ingest job"
  kubectl apply -f "$H100_DEPLOY/ragbench-ingest-k3s.yaml"
  kubectl wait --for=condition=complete job/ragbench-ingest -n llm-service --timeout=7200s
  kubectl logs job/ragbench-ingest -n llm-service --tail=5
else
  log "ingest job already exists; logs:"
  kubectl logs job/ragbench-ingest -n llm-service --tail=3 || true
fi

log "PHASE 2 DONE. Next: local_service/scripts/capture_tiers.sh"
