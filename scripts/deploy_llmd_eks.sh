#!/usr/bin/env bash
# Deploy llm-d on EKS using config from deploy/config.env.
# To switch models: edit MODEL_NAME + MODEL_HF_REPO + MODEL_MAX_LEN in deploy/config.env, then re-run.
set -euo pipefail

KERNEL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_ENV="${KERNEL_ROOT}/deploy/config.env"
LLMD_LOCAL_DIR="${KERNEL_ROOT}/deploy/llmd-local"
LLMD_EKS_DIR="${KERNEL_ROOT}/deploy/llmd-eks"

# ── Load config ──────────────────────────────────────────────────────────────
[ -f "${CONFIG_ENV}" ] || { echo "ERROR: deploy/config.env not found"; exit 1; }
# shellcheck source=../deploy/config.env
set -a; source "${CONFIG_ENV}"; set +a

INFRA_RELEASE="${INFRA_RELEASE:-infra}"
MODEL_RELEASE="${MODEL_RELEASE:-ms}"
INFRA_DEPLOYMENT="${INFRA_DEPLOYMENT:-infra-inference-gateway-istio}"
MODEL_DEPLOYMENT="${MODEL_DEPLOYMENT:-ms-llm-d-modelservice-decode}"

log() { printf '[deploy_llmd_eks] %s\n' "$*"; }
die() { printf '[deploy_llmd_eks] ERROR: %s\n' "$*" >&2; exit 1; }

command -v kubectl >/dev/null || die "kubectl not found"
command -v helm    >/dev/null || die "helm not found"
command -v envsubst >/dev/null || die "envsubst not found (install gettext)"

[ -f "${LLMD_LOCAL_DIR}/llm-d-infra-v1.3.10.tgz" ]        || die "missing chart: llm-d-infra-v1.3.10.tgz"
[ -f "${LLMD_LOCAL_DIR}/llm-d-modelservice-v0.4.8.tgz" ]   || die "missing chart: llm-d-modelservice-v0.4.8.tgz"
[ -f "${LLMD_EKS_DIR}/modelservice-values.tpl.yaml" ]       || die "missing template: modelservice-values.tpl.yaml"

[ -n "${GPU_COUNT:-}" ] && [[ "${GPU_COUNT}" =~ ^[1-9][0-9]*$ ]] \
    || die "GPU_COUNT not set or invalid in config.env — must be the number of GPUs on ${GPU_INSTANCE_TYPE} (e.g. g6.12xlarge=4)"

log "Deploying llm-d on EKS"
log "  Model:     ${MODEL_NAME}  (${MODEL_HF_REPO})"
log "  Max len:   ${MODEL_MAX_LEN}"
log "  GPU util:  ${MODEL_GPU_MEM_UTIL}"
log "  GPU node:  ${GPU_INSTANCE_TYPE}  (${GPU_COUNT} GPU(s), TP=${GPU_COUNT})"
log "  Namespace: ${NAMESPACE_LLMD}"

# ── Namespace ────────────────────────────────────────────────────────────────
kubectl get namespace "${NAMESPACE_LLMD}" >/dev/null 2>&1 \
  || kubectl create namespace "${NAMESPACE_LLMD}"

# ── StorageClasses (skip if already exist — params are immutable) ─────────────
# ebs-gp3: aws-ebs-csi-driver for managed node group workloads (storage stack)
kubectl get storageclass ebs-gp3 >/dev/null 2>&1 || kubectl apply -f - <<EOF
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: ebs-gp3
provisioner: ebs.csi.aws.com
volumeBindingMode: WaitForFirstConsumer
reclaimPolicy: Delete
parameters:
  type: gp3
  encrypted: "true"
EOF
# ebs-gp3-automode: EKS Auto Mode provisioner for Karpenter/GPU workloads (model PVC)
kubectl get storageclass ebs-gp3-automode >/dev/null 2>&1 || kubectl apply -f - <<EOF
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: ebs-gp3-automode
provisioner: ebs.csi.eks.amazonaws.com
volumeBindingMode: WaitForFirstConsumer
reclaimPolicy: Delete
parameters:
  type: gp3
  encrypted: "true"
EOF

# ── Model PV restore (if an existing EBS volume is tagged for this PVC) ──────
# Avoids a 40-min model re-download when recovering from a cluster teardown.
# The tag "kubernetes.io/created-for/pvc/name=llmd-model-pvc" identifies the volume.
MODEL_RESTORE_VOL=$(aws ec2 describe-volumes \
  --region "${AWS_REGION}" \
  --filters \
    "Name=tag:kubernetes.io/created-for/pvc/name,Values=llmd-model-pvc" \
    "Name=tag:kubernetes.io/cluster/${CLUSTER_NAME:-e2e-cluster-eks},Values=owned" \
    "Name=status,Values=available" \
  --query 'Volumes[0].VolumeId' \
  --output text 2>/dev/null || echo "")
MODEL_RESTORE_AZ=$(aws ec2 describe-volumes \
  --region "${AWS_REGION}" \
  --filters \
    "Name=tag:kubernetes.io/created-for/pvc/name,Values=llmd-model-pvc" \
    "Name=tag:kubernetes.io/cluster/${CLUSTER_NAME:-e2e-cluster-eks},Values=owned" \
    "Name=status,Values=available" \
  --query 'Volumes[0].AvailabilityZone' \
  --output text 2>/dev/null || echo "")

if [ -n "${MODEL_RESTORE_VOL}" ] && [ "${MODEL_RESTORE_VOL}" != "None" ]; then
  log "Found existing model EBS volume ${MODEL_RESTORE_VOL} in ${MODEL_RESTORE_AZ} — restoring"
  # Use ebs.csi.aws.com (addon driver) + ebs-gp3 StorageClass.
  # GPU nodes MUST be managed node groups (not EKS Auto Mode) — the ebs-csi-node
  # daemonset only runs on managed nodes, and the Auto Mode driver cannot attach
  # pre-existing volumes (volumes it did not create with its own tags).
  kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolume
metadata:
  name: restore-llmd-model-pvc
spec:
  capacity:
    storage: ${MODEL_STORAGE_GI}Gi
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  storageClassName: ebs-gp3
  csi:
    driver: ebs.csi.aws.com
    volumeHandle: ${MODEL_RESTORE_VOL}
    fsType: ext4
  nodeAffinity:
    required:
      nodeSelectorTerms:
        - matchExpressions:
            - key: topology.kubernetes.io/zone
              operator: In
              values:
                - ${MODEL_RESTORE_AZ}
  claimRef:
    name: llmd-model-pvc
    namespace: ${NAMESPACE_LLMD}
EOF
  log "Restore PV created for model volume ${MODEL_RESTORE_VOL}"
else
  log "No existing model volume found — will download fresh"
fi

# ── Model PVC (sized from config.env) ────────────────────────────────────────
log "Applying model PVC (${MODEL_STORAGE_GI}Gi)"
sed "s/storage: 40Gi/storage: ${MODEL_STORAGE_GI}Gi/" \
  "${LLMD_EKS_DIR}/model-pvc.yaml" \
  | kubectl apply -n "${NAMESPACE_LLMD}" -f -

# EKS Auto Mode requires pv.kubernetes.io/bind-completed annotation for static PVs
# to consider the PVC as bound during pod scheduling validation.
if [ -n "${MODEL_RESTORE_VOL}" ] && [ "${MODEL_RESTORE_VOL}" != "None" ]; then
  kubectl annotate pvc llmd-model-pvc -n "${NAMESPACE_LLMD}" \
    "pv.kubernetes.io/bind-completed=yes" --overwrite 2>/dev/null || true
fi

# ── Model download (skip if PVC already has data) ────────────────────────────
EXISTING_JOB=$(kubectl get job llmd-model-download -n "${NAMESPACE_LLMD}" \
  --ignore-not-found -o jsonpath='{.status.succeeded}' 2>/dev/null || echo "")
if [ "${EXISTING_JOB}" = "1" ]; then
  log "Model already downloaded (job succeeded) — skipping download"
else
  log "Starting model download job (${MODEL_HF_REPO} → EBS)"
  # Patch the model repo into the job manifest
  sed "s|Qwen/Qwen2.5-32B-Instruct|${MODEL_HF_REPO}|g" \
    "${LLMD_EKS_DIR}/model-download-job.yaml" \
    | kubectl apply -n "${NAMESPACE_LLMD}" -f -

  log "Waiting for model download (14b≈10 min, 32b≈40 min)..."
  kubectl wait job/llmd-model-download -n "${NAMESPACE_LLMD}" \
    --for=condition=complete --timeout=7200s
  log "Download complete"
fi

# ── Prerequisites: Gateway API CRDs + Istio ──────────────────────────────────
log "Installing Gateway API CRDs"
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.2.1/standard-install.yaml

if ! helm list -n istio-system 2>/dev/null | grep -q istiod; then
  log "Installing Istio"
  helm repo add istio https://istio-release.storage.googleapis.com/charts --force-update >/dev/null
  helm upgrade --install istio-base istio/base -n istio-system --create-namespace
  helm upgrade --install istiod istio/istiod -n istio-system --wait --timeout=300s
else
  log "Istio already installed — skipping"
fi

# ── llm-d infra (gateway + istio) ────────────────────────────────────────────
log "Installing llm-d gateway infrastructure"
helm upgrade --install "${INFRA_RELEASE}" "${LLMD_LOCAL_DIR}/llm-d-infra-v1.3.10.tgz" \
  -n "${NAMESPACE_LLMD}" \
  --create-namespace

# ── llm-d model service ───────────────────────────────────────────────────────
log "Generating model values from template"
RENDERED_VALUES="$(mktemp /tmp/modelservice-values-XXXXX.yaml)"
# Limit substitution to our config variables only — avoids clobbering shell vars in the vLLM startup script
envsubst '${MODEL_NAME} ${MODEL_MAX_LEN} ${MODEL_GPU_MEM_UTIL} ${GPU_INSTANCE_TYPE} ${GPU_COUNT} ${NAMESPACE_LLMD} ${MODEL_REPLICAS}' \
  < "${LLMD_EKS_DIR}/modelservice-values.tpl.yaml" > "${RENDERED_VALUES}"

log "Installing llm-d model service (${MODEL_NAME})"
helm upgrade --install "${MODEL_RELEASE}" "${LLMD_LOCAL_DIR}/llm-d-modelservice-v0.4.8.tgz" \
  -n "${NAMESPACE_LLMD}" \
  -f "${RENDERED_VALUES}"
rm -f "${RENDERED_VALUES}"

# Patch decode deployment:
#   1. Pin to GPU node + GPU taint toleration
#   2. Recreate strategy (single GPU, no rolling update)
#   3. Run vllm container as root (needed for SELinux-labeled EBS volume on Bottlerocket)
#   4. Remove readOnly from model volume (Bottlerocket SELinux requires read-write mount for relabeling)
kubectl patch deployment -n "${NAMESPACE_LLMD}" "${MODEL_DEPLOYMENT}" \
  --type='merge' \
  -p "{
    \"spec\":{
      \"strategy\":{\"type\":\"Recreate\",\"rollingUpdate\":null},
      \"progressDeadlineSeconds\":3600,
      \"template\":{
        \"spec\":{
          \"nodeSelector\":{\"node.kubernetes.io/instance-type\":\"${GPU_INSTANCE_TYPE}\"},
          \"tolerations\":[{\"key\":\"nvidia.com/gpu\",\"operator\":\"Exists\",\"effect\":\"NoSchedule\"}]
        }
      }
    }
  }" >/dev/null || true

# Container-level patches (runAsUser + readOnly fix) via JSON patch
kubectl patch deployment -n "${NAMESPACE_LLMD}" "${MODEL_DEPLOYMENT}" \
  --type='json' \
  -p='[
    {"op":"add","path":"/spec/template/spec/containers/0/securityContext","value":{"runAsUser":0,"runAsGroup":0,"privileged":true}},
    {"op":"replace","path":"/spec/template/spec/containers/0/volumeMounts/2/readOnly","value":false}
  ]' >/dev/null || true

kubectl rollout restart deployment/"${MODEL_DEPLOYMENT}" -n "${NAMESPACE_LLMD}"

# ── Direct service + route ────────────────────────────────────────────────────
log "Applying direct decode Service and HTTPRoute"
kubectl apply -n "${NAMESPACE_LLMD}" -f "${LLMD_EKS_DIR}/decode-direct-service.yaml"
kubectl apply -n "${NAMESPACE_LLMD}" -f "${LLMD_EKS_DIR}/route-direct.yaml"

# ── Update FastAPI configmap to point at this model + gateway ────────────────
GATEWAY_URL="http://infra-inference-gateway-istio.${NAMESPACE_LLMD}.svc.cluster.local:80"
log "Patching FastAPI configmap: model=${MODEL_NAME}  gateway=${GATEWAY_URL}"
kubectl patch configmap llm-service-kernel-config-fullstack \
  -n "${NAMESPACE_SERVICE}" \
  --type=merge \
  -p "{\"data\":{
    \"GENERATION_MODEL_NAME\":\"${MODEL_NAME}\",
    \"GENERATION_BASE_URL\":\"${GATEWAY_URL}\"
  }}" || log "Warning: FastAPI configmap not found yet — apply k8s-fastapi first"

# Bounce FastAPI so it picks up the new configmap values
kubectl rollout restart deployment/llm-service-kernel \
  -n "${NAMESPACE_SERVICE}" 2>/dev/null || true

# ── Wait for decode pod ───────────────────────────────────────────────────────
log "Waiting for llm-d deployments (vLLM load can take 3-5 min)..."
kubectl rollout status deployment/"${INFRA_DEPLOYMENT}" \
  -n "${NAMESPACE_LLMD}" --timeout=600s
kubectl rollout status deployment/"${MODEL_DEPLOYMENT}" \
  -n "${NAMESPACE_LLMD}" --timeout=1800s

log ""
log "llm-d EKS deployment complete"
log "Model: ${MODEL_NAME}  |  GPU: ${GPU_INSTANCE_TYPE}  |  Namespace: ${NAMESPACE_LLMD}"
kubectl get pods -n "${NAMESPACE_LLMD}" -o wide
