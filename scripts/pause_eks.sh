#!/usr/bin/env bash
# Pause EKS cluster — stop compute charges while preserving all data on EBS.
#
# Usage:
#   bash scripts/pause_eks.sh            # pause everything
#   bash scripts/pause_eks.sh --gpu-only # pause GPU only (keeps service running)
#
# Resume: bash scripts/resume_eks.sh [--gpu-only]
set -euo pipefail

KERNEL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -a; source "${KERNEL_ROOT}/deploy/config.env"; set +a

CLUSTER="${CLUSTER_NAME:-e2e-cluster-eks}"
REGION="${AWS_REGION:-eu-west-2}"
GPU_ONLY=0
[[ "${1:-}" == "--gpu-only" ]] && GPU_ONLY=1

log()  { printf '\n\033[1;34m[pause]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m  ✓\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m  ⚠\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[pause] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# ── Safety check (only relevant for full pause) ───────────────────────────────
if [ "${GPU_ONLY}" = "0" ]; then
  log "Checking for running jobs before pausing..."
  running_jobs="$(kubectl get jobs -n "${NAMESPACE_SERVICE}" --no-headers 2>/dev/null \
    | grep -v 'Complete\|Failed' | awk '{print $1}' || true)"
  if [ -n "${running_jobs}" ]; then
    warn "Running jobs detected: ${running_jobs}"
    read -r -p "  Pausing will kill these jobs. Continue? [y/N] " confirm
    [[ "${confirm}" =~ ^[Yy]$ ]] || die "Aborted"
  fi
fi

# ── Always: scale vLLM to 0 + delete GPU NodePool (forces immediate termination) ──
log "Scaling vLLM to 0..."
kubectl scale deployment ms-llm-d-modelservice-decode \
  -n "${NAMESPACE_LLMD}" --replicas=0 2>/dev/null || true
kubectl scale deployment infra-inference-gateway-istio \
  -n "${NAMESPACE_LLMD}" --replicas=0 2>/dev/null || true
ok "vLLM scaled to 0"

log "Scaling GPU node group to 0 (terminates p5.4xlarge)..."
aws eks update-nodegroup-config \
  --cluster-name "${CLUSTER}" \
  --nodegroup-name gpu \
  --region "${REGION}" \
  --scaling-config minSize=0,maxSize=1,desiredSize=0 \
  >/dev/null 2>&1 || true
ok "GPU node group scaling to 0 — instance terminating (~\$16.29/hr saved)"

if [ "${GPU_ONLY}" = "1" ]; then
  echo
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  GPU paused (service still running on compute node)."
  echo "    Saved: ~\$16.29/hr (p5.4xlarge terminated)"
  echo "    Still running: FastAPI, Milvus, MongoDB, SeaweedFS (~\$4.78/hr)"
  echo
  echo "  Resume GPU:  bash scripts/resume_eks.sh --gpu-only"
  echo "  Full pause:  bash scripts/pause_eks.sh"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  exit 0
fi

# ── Full pause: scale compute managed node group to 0 ────────────────────────
log "Scaling compute node group to 0 (terminates c7i.metal-24xl)..."
aws eks update-nodegroup-config \
  --cluster-name "${CLUSTER}" \
  --nodegroup-name compute \
  --region "${REGION}" \
  --scaling-config minSize=0,maxSize=2,desiredSize=0 \
  >/dev/null
ok "Compute node group scaling to 0 — instance will terminate in ~2 min"

log "Waiting for compute node to terminate..."
local_deadline=$((SECONDS + 300))
while [ "${SECONDS}" -lt "${local_deadline}" ]; do
  node_count="$(kubectl get nodes --no-headers 2>/dev/null \
    | grep 'Ready' | grep -v 'SchedulingDisabled' | wc -l || echo 0)"
  if [ "${node_count}" -eq 0 ]; then
    ok "All nodes terminated"
    break
  fi
  printf '  Nodes still up: %s — waiting...\r' "${node_count}"
  sleep 10
done

echo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Cluster fully paused. Ongoing costs:"
echo "    EKS control plane  : \$0.10/hr  (~\$2.40/day)"
echo "    EBS volumes        : ~\$0.43/day (162Gi, model + data preserved)"
echo "    Total paused cost  : ~\$3.00/day"
echo
echo "  Resume with:  bash scripts/resume_eks.sh"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
