#!/usr/bin/env bash
set -euo pipefail

KERNEL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLMD_DIR="${KERNEL_ROOT}/deploy/llmd-local"

MINIKUBE_PROFILE="${MINIKUBE_PROFILE:-minikube}"
NAMESPACE="${NAMESPACE:-llm-d-local}"

INFRA_RELEASE="${INFRA_RELEASE:-infra-local}"
MODEL_RELEASE="${MODEL_RELEASE:-ms-local}"
POOL_RELEASE="${POOL_RELEASE:-ms-local-pool}"

LOAD_LOCAL_IMAGES="${LOAD_LOCAL_IMAGES:-0}"
PROVISION_MODEL="${PROVISION_MODEL:-1}"

# New: direct-service path is the default, so pool install is off by default.
DEPLOY_POOL="${DEPLOY_POOL:-0}"

INFRA_DEPLOYMENT="${INFRA_DEPLOYMENT:-infra-local-inference-gateway-istio}"
MODEL_DEPLOYMENT="${MODEL_DEPLOYMENT:-ms-local-llm-d-modelservice-decode}"
POOL_DEPLOYMENT="${POOL_DEPLOYMENT:-ms-local-pool-epp}"

DIRECT_SERVICE_MANIFEST="${LLMD_DIR}/ms-local-decode-direct-service.yaml"
DIRECT_ROUTE_MANIFEST="${LLMD_DIR}/ms-local-route-direct.yaml"

log() {
  printf '[deploy_llmd_local] %s\n' "$*"
}

die() {
  printf '[deploy_llmd_local] ERROR: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

load_tar_if_exists() {
  local tar_path="$1"
  if [ -f "$tar_path" ]; then
    log "Loading image tar: $tar_path"
    docker load -i "$tar_path"
  fi
}

require_cmd minikube
require_cmd docker
require_cmd kubectl
require_cmd helm

[ -d "${LLMD_DIR}" ] || die "llm-d directory not found: ${LLMD_DIR}"
[ -f "${LLMD_DIR}/llm-d-infra-v1.3.10.tgz" ] || die "missing chart: ${LLMD_DIR}/llm-d-infra-v1.3.10.tgz"
[ -f "${LLMD_DIR}/llm-d-modelservice-v0.4.8.tgz" ] || die "missing chart: ${LLMD_DIR}/llm-d-modelservice-v0.4.8.tgz"
[ -f "${LLMD_DIR}/modelservice-values.yaml" ] || die "missing values file: ${LLMD_DIR}/modelservice-values.yaml"
[ -f "${LLMD_DIR}/model-pv.yaml" ] || die "missing manifest: ${LLMD_DIR}/model-pv.yaml"
[ -f "${LLMD_DIR}/model-pvc.yaml" ] || die "missing manifest: ${LLMD_DIR}/model-pvc.yaml"
[ -f "${DIRECT_SERVICE_MANIFEST}" ] || die "missing manifest: ${DIRECT_SERVICE_MANIFEST}"
[ -f "${DIRECT_ROUTE_MANIFEST}" ] || die "missing manifest: ${DIRECT_ROUTE_MANIFEST}"

if [ "${DEPLOY_POOL}" = "1" ]; then
  [ -f "${LLMD_DIR}/inferencepool-v1.3.1.tgz" ] || die "missing chart: ${LLMD_DIR}/inferencepool-v1.3.1.tgz"
  [ -f "${LLMD_DIR}/infpool-values-chip.yaml" ] || die "missing values file: ${LLMD_DIR}/infpool-values-chip.yaml"
fi

log "Pointing Docker to minikube daemon"
eval "$(minikube -p "${MINIKUBE_PROFILE}" docker-env)"

kubectl get namespace "${NAMESPACE}" >/dev/null 2>&1 || kubectl create namespace "${NAMESPACE}"

if [ "${LOAD_LOCAL_IMAGES}" = "1" ]; then
  log "Loading llm-d images into minikube Docker daemon"
  load_tar_if_exists "${LLMD_DIR}/gaie-images.tar"

  if [ -d "${LLMD_DIR}/images" ]; then
    while IFS= read -r -d '' tar_file; do
      load_tar_if_exists "${tar_file}"
    done < <(find "${LLMD_DIR}/images" -maxdepth 1 -type f -name '*.tar' -print0 | sort -z)
  fi
else
  log "Skipping local image loading"
fi

if [ "${PROVISION_MODEL}" = "1" ]; then
  log "Provisioning model artifacts onto minikube node"
  bash "${KERNEL_ROOT}/scripts/provision_model_artifacts.sh"
else
  log "Skipping model provisioning"
fi

log "Applying model PV/PVC"
kubectl apply -f "${LLMD_DIR}/model-pv.yaml"
kubectl apply -f "${LLMD_DIR}/model-pvc.yaml"

log "Installing llm-d gateway infrastructure"
helm upgrade --install "${INFRA_RELEASE}" "${LLMD_DIR}/llm-d-infra-v1.3.10.tgz" \
  -n "${NAMESPACE}" \
  --create-namespace

log "Installing llm-d model service"
helm upgrade --install "${MODEL_RELEASE}" "${LLMD_DIR}/llm-d-modelservice-v0.4.8.tgz" \
  -n "${NAMESPACE}" \
  -f "${LLMD_DIR}/modelservice-values.yaml"

log "Enforcing single-GPU safe rollout strategy on decode deployment"
kubectl patch deployment -n "${NAMESPACE}" "${MODEL_DEPLOYMENT}" \
  --type='merge' \
  -p '{"spec":{"strategy":{"type":"Recreate","rollingUpdate":null},"progressDeadlineSeconds":3600}}' >/dev/null || true

log "Restarting decode deployment so patched strategy takes effect"
kubectl rollout restart deployment/"${MODEL_DEPLOYMENT}" -n "${NAMESPACE}"

if [ "${DEPLOY_POOL}" = "1" ]; then
  log "Installing inference pool (optional legacy path)"
  helm upgrade --install "${POOL_RELEASE}" "${LLMD_DIR}/inferencepool-v1.3.1.tgz" \
    -n "${NAMESPACE}" \
    -f "${LLMD_DIR}/infpool-values-chip.yaml"

  log "Enforcing local image usage on EPP deployment"
  kubectl patch deployment -n "${NAMESPACE}" "${POOL_DEPLOYMENT}" \
    --type='strategic' \
    -p '{"spec":{"template":{"spec":{"containers":[{"name":"epp","imagePullPolicy":"IfNotPresent"}]}}}}' >/dev/null || true

  log "Restarting EPP deployment so patched policy takes effect"
  kubectl rollout restart deployment/"${POOL_DEPLOYMENT}" -n "${NAMESPACE}"
fi

log "Waiting for llm-d deployments"
kubectl rollout status deployment/"${INFRA_DEPLOYMENT}" -n "${NAMESPACE}" --timeout=600s
kubectl rollout status deployment/"${MODEL_DEPLOYMENT}" -n "${NAMESPACE}" --timeout=2000s

if [ "${DEPLOY_POOL}" = "1" ]; then
  kubectl rollout status deployment/"${POOL_DEPLOYMENT}" -n "${NAMESPACE}" --timeout=300s
fi

log "Applying direct decode Service and HTTPRoute"
kubectl apply -f "${DIRECT_SERVICE_MANIFEST}"
kubectl apply -f "${DIRECT_ROUTE_MANIFEST}"

log "Current llm-d resources"
kubectl get pods,svc -n "${NAMESPACE}" -o wide

log "Current Gateway routes"
kubectl get httproute -n "${NAMESPACE}" || true

log "Verifying model files inside serving pod"
kubectl exec -n "${NAMESPACE}" deploy/"${MODEL_DEPLOYMENT}" -- sh -c 'ls -lah /model-cache | head -20'

log "llm-d local deployment complete"
