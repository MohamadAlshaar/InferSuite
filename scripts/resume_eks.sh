#!/usr/bin/env bash
# Resume EKS cluster after pause.
#
# Usage:
#   bash scripts/resume_eks.sh            # resume everything
#   bash scripts/resume_eks.sh --gpu-only # resume GPU only (compute already running)
set -euo pipefail

KERNEL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -a; source "${KERNEL_ROOT}/deploy/config.env"; set +a

CLUSTER="${CLUSTER_NAME:-e2e-cluster-eks}"
REGION="${AWS_REGION:-eu-west-2}"
GPU_ONLY=0
[[ "${1:-}" == "--gpu-only" ]] && GPU_ONLY=1

log()  { printf '\n\033[1;34m[resume]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m  ✓\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m  ⚠\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[resume] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# ── Compute node (skip if gpu-only) ──────────────────────────────────────────
if [ "${GPU_ONLY}" = "1" ]; then
  log "GPU-only resume — skipping compute node (already running)"
else
log "Scaling general-purpose node group to 1 (c7i.metal-24xl)..."
# Use the EKS managed node group — NOT a Karpenter NodePool.
# EKS Auto Mode nodes (Karpenter-provisioned) have an explicit IAM deny on
# ec2:AttachVolume for the EBSCSIDriverRole; managed node group instances do not.
aws eks update-nodegroup-config \
  --cluster-name "${CLUSTER}" \
  --nodegroup-name compute \
  --region "${REGION}" \
  --scaling-config minSize=0,maxSize=2,desiredSize=1 \
  >/dev/null
ok "Compute node group scaling to 1 — instance starting (~3-5 min)"

# ── Wait for managed node group node to be Ready (c7i.metal-24xl in eu-west-2b) ─
log "Waiting for managed node group node to be Ready..."
local_deadline=$((SECONDS + 600))
while [ "${SECONDS}" -lt "${local_deadline}" ]; do
  # Wait for a non-Karpenter, non-system node (instance-type c7i.metal-24xl)
  ready="$(kubectl get nodes -l node.kubernetes.io/instance-type=c7i.metal-24xl \
    --no-headers 2>/dev/null | grep 'Ready' | grep -v 'SchedulingDisabled' | wc -l || echo 0)"
  if [ "${ready}" -ge 1 ]; then
    ok "Compute node Ready (c7i.metal-24xl)"
    break
  fi
  printf '  Waiting for c7i.metal-24xl node...\r'
  sleep 15
done

# ── Scale up + restart service pods (they were evicted when node went away) ───
log "Scaling up storage + FastAPI pods..."
# Scale to 1 first — deployments may have been scaled to 0 during pause
kubectl scale deployment milvus milvus-etcd milvus-minio mongodb \
  seaweed-master seaweed-volume seaweed-filer seaweed-s3 llm-service-kernel \
  -n "${NAMESPACE_SERVICE}" --replicas=1 2>/dev/null || true

log "Restarting storage + FastAPI pods..."
kubectl rollout restart deployment/milvus      -n "${NAMESPACE_SERVICE}" 2>/dev/null || true
kubectl rollout restart deployment/mongodb     -n "${NAMESPACE_SERVICE}" 2>/dev/null || true
kubectl rollout restart deployment/seaweed-s3  -n "${NAMESPACE_SERVICE}" 2>/dev/null || true
kubectl rollout restart deployment/seaweed-master -n "${NAMESPACE_SERVICE}" 2>/dev/null || true
kubectl rollout restart deployment/seaweed-volume -n "${NAMESPACE_SERVICE}" 2>/dev/null || true
kubectl rollout restart deployment/seaweed-filer  -n "${NAMESPACE_SERVICE}" 2>/dev/null || true

log "Waiting for storage stack to be ready..."
kubectl rollout status deployment/milvus   -n "${NAMESPACE_SERVICE}" --timeout=300s
kubectl rollout status deployment/mongodb  -n "${NAMESPACE_SERVICE}" --timeout=120s
ok "Storage ready"

kubectl rollout restart deployment/llm-service-kernel -n "${NAMESPACE_SERVICE}" 2>/dev/null || true
kubectl rollout status  deployment/llm-service-kernel -n "${NAMESPACE_SERVICE}" --timeout=120s
ok "FastAPI ready"
fi  # end GPU_ONLY skip block

# ── Recreate GPU NodePool (deleted during pause) ─────────────────────────────
log "Scaling GPU node group to 1 (${GPU_INSTANCE_TYPE})..."
aws eks update-nodegroup-config \
  --cluster-name "${CLUSTER}" \
  --nodegroup-name gpu \
  --region "${REGION}" \
  --scaling-config minSize=0,maxSize=1,desiredSize=1 \
  >/dev/null
ok "GPU node group scaling to 1 → ${GPU_INSTANCE_TYPE} (~3-5 min)"

# ── Scale vLLM back up ───────────────────────────────────────────────────────
log "Scaling vLLM back up..."
kubectl scale deployment ms-llm-d-modelservice-decode \
  -n "${NAMESPACE_LLMD}" --replicas=1 2>/dev/null || true
kubectl scale deployment infra-inference-gateway-istio \
  -n "${NAMESPACE_LLMD}" --replicas=1 2>/dev/null || true
ok "vLLM scaling up — GPU node provisioning (~3-5 min) + model load (~8-10 min)"


# ── Ensure benchmark-runner pod is up ────────────────────────────────────────
log "Ensuring benchmark-runner pod is running..."
kubectl get pod benchmark-runner -n "${NAMESPACE_SERVICE}" >/dev/null 2>&1 || \
  sed "s|YOUR_ACCOUNT_ID\.dkr\.ecr\.YOUR_REGION|${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}|g" \
    "${KERNEL_ROOT}/deploy/k8s-benchmark/benchmark-runner-pod.yaml" \
    | kubectl apply -f - >/dev/null
ok "benchmark-runner pod ready"

echo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Cluster resuming. Services coming up:"
echo "    Storage + FastAPI : ready now"
echo "    vLLM (GPU node)   : ~10-15 min to be fully ready"
echo
echo "  Check health:"
echo "    kubectl port-forward svc/llm-service-kernel 18081:8080 -n ${NAMESPACE_SERVICE}"
echo "    curl -s http://127.0.0.1:18081/health | python3 -m json.tool"
echo
echo "  Run benchmark (once vLLM is ready):"
echo "    kubectl exec -it benchmark-runner -n ${NAMESPACE_SERVICE} -- bash"
echo "    bash scripts/run_benchmark.sh --tokens 64"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
