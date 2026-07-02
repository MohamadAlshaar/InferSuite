#!/usr/bin/env bash
set -euo pipefail

KERNEL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MINIKUBE_PROFILE="${MINIKUBE_PROFILE:-minikube}"

LLMD_NAMESPACE="${LLMD_NAMESPACE:-llm-d-local}"
FASTAPI_NAMESPACE="${FASTAPI_NAMESPACE:-llm-service}"

LLMD_SERVICE="${LLMD_SERVICE:-infra-local-inference-gateway-istio}"
FASTAPI_SERVICE="${FASTAPI_SERVICE:-llm-service-kernel}"
FASTAPI_DEPLOYMENT="${FASTAPI_DEPLOYMENT:-llm-service-kernel}"
BOOTSTRAP_JOB="${BOOTSTRAP_JOB:-llm-service-kernel-bootstrap}"

FASTAPI_IMAGE="${FASTAPI_IMAGE:-llm-service-kernel:fastapi-selfcontained}"
ASSET_MODE="${ASSET_MODE:-auto}"

GATEWAY_API_VERSION="${GATEWAY_API_VERSION:-v1.4.0}"
ISTIO_PROFILE="${ISTIO_PROFILE:-minimal}"
ISTIO_NAMESPACE="${ISTIO_NAMESPACE:-istio-system}"

PREPARE_ASSETS="${PREPARE_ASSETS:-1}"
BUILD_FASTAPI_IMAGE="${BUILD_FASTAPI_IMAGE:-1}"
INSTALL_CLUSTER_PREREQS="${INSTALL_CLUSTER_PREREQS:-1}"
APPLY_STORAGE="${APPLY_STORAGE:-1}"
WAIT_STORAGE="${WAIT_STORAGE:-1}"
DEPLOY_LLMD="${DEPLOY_LLMD:-1}"
WAIT_LLMD="${WAIT_LLMD:-1}"
APPLY_FASTAPI_RESOURCES="${APPLY_FASTAPI_RESOURCES:-1}"
RUN_BOOTSTRAP_JOB="${RUN_BOOTSTRAP_JOB:-1}"
DEPLOY_FASTAPI="${DEPLOY_FASTAPI:-1}"
WAIT_FASTAPI_READY="${WAIT_FASTAPI_READY:-1}"
RUN_VALIDATE="${RUN_VALIDATE:-0}"
DEPLOY_JAEGER="${DEPLOY_JAEGER:-0}"

LLMD_LOCAL_PORT="${LLMD_LOCAL_PORT:-18080}"
FASTAPI_LOCAL_PORT="${FASTAPI_LOCAL_PORT:-18081}"

STORAGE_TIMEOUT_S="${STORAGE_TIMEOUT_S:-1200}"
ISTIO_READY_TIMEOUT_S="${ISTIO_READY_TIMEOUT_S:-1200}"
LLMD_READY_TIMEOUT_S="${LLMD_READY_TIMEOUT_S:-3000}"
BOOTSTRAP_TIMEOUT_S="${BOOTSTRAP_TIMEOUT_S:-1800}"
FASTAPI_READY_TIMEOUT_S="${FASTAPI_READY_TIMEOUT_S:-600}"

_MANAGED_PIDS=()


log() {
  printf '[deploy_fullstack_single_node] %s\n' "$*"
}

die() {
  printf '[deploy_fullstack_single_node] ERROR: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

cleanup() {
  local pid
  for pid in "${_MANAGED_PIDS[@]:-}"; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill "${pid}" >/dev/null 2>&1 || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
}

trap cleanup EXIT INT TERM

use_minikube_docker_env() {
  # shellcheck disable=SC1090
  eval "$(minikube -p "${MINIKUBE_PROFILE}" docker-env)"
}

start_port_forward() {
  local namespace="$1"
  local service="$2"
  local local_port="$3"
  local remote_port="$4"
  local log_file="/tmp/${service}-${local_port}-portforward.log"

  if lsof -iTCP:"${local_port}" -sTCP:LISTEN >/dev/null 2>&1; then
    log "Local port ${local_port} already in use; assuming something is already listening"
    return 0
  fi

  log "Starting port-forward svc/${service} ${local_port}:${remote_port} in namespace ${namespace}"
  kubectl port-forward -n "${namespace}" "svc/${service}" "${local_port}:${remote_port}" >"${log_file}" 2>&1 &
  local pid=$!
  _MANAGED_PIDS+=("${pid}")
  sleep 2

  if ! kill -0 "${pid}" >/dev/null 2>&1; then
    cat "${log_file}" >&2 || true
    die "port-forward failed for svc/${service}"
  fi
}

wait_http_ok() {
  local url="$1"
  local timeout_s="$2"
  local expected_status="${3:-200}"

  python3 - "$url" "$timeout_s" "$expected_status" <<'PY'
import sys
import time
import urllib.error
import urllib.request

url = sys.argv[1]
timeout_s = float(sys.argv[2])
expected_status = int(sys.argv[3])

deadline = time.time() + timeout_s
last = "timeout"

while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            if resp.status == expected_status:
                print(body)
                raise SystemExit(0)
            last = f"status={resp.status} body={body[:400]}"
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        last = f"http_error={exc.code} body={raw[:400]}"
    except Exception as exc:
        last = str(exc)
    time.sleep(2)

print(last, file=sys.stderr)
raise SystemExit(1)
PY
}

prepare_assets() {
  log "Preparing FastAPI runtime assets (ASSET_MODE=${ASSET_MODE})"
  ASSET_MODE="${ASSET_MODE}" bash "${KERNEL_ROOT}/scripts/prepare_fastapi_runtime_assets.sh"
}

build_fastapi_image() {
  log "Building FastAPI image in minikube Docker daemon: ${FASTAPI_IMAGE}"
  use_minikube_docker_env
  docker build --no-cache -t "${FASTAPI_IMAGE}" -f "${KERNEL_ROOT}/Dockerfile.service" "${KERNEL_ROOT}"
}

ensure_gateway_api_crds() {
  if kubectl get crd gateways.gateway.networking.k8s.io >/dev/null 2>&1; then
    log "Gateway API CRDs already installed"
    return 0
  fi

  log "Installing Gateway API CRDs (${GATEWAY_API_VERSION})"
  kubectl kustomize "github.com/kubernetes-sigs/gateway-api/config/crd?ref=${GATEWAY_API_VERSION}" | kubectl apply -f -

  kubectl get crd gateways.gateway.networking.k8s.io >/dev/null 2>&1 \
    || die "Gateway API CRD install did not create gateways.gateway.networking.k8s.io"
  kubectl get crd httproutes.gateway.networking.k8s.io >/dev/null 2>&1 \
    || die "Gateway API CRD install did not create httproutes.gateway.networking.k8s.io"
}

ensure_istio() {
  require_cmd istioctl

  if kubectl get deployment istiod -n "${ISTIO_NAMESPACE}" >/dev/null 2>&1; then
    log "Istio appears to already be installed"
  else
    log "Installing Istio (profile=${ISTIO_PROFILE})"
    istioctl install --set profile="${ISTIO_PROFILE}" -y
  fi

  log "Waiting for Istio CRDs and control plane"

  local deadline=$((SECONDS + ISTIO_READY_TIMEOUT_S))
  local telemetry_ready=0
  local istiod_ready=0

  while [ "${SECONDS}" -lt "${deadline}" ]; do
    if kubectl get crd telemetries.telemetry.istio.io >/dev/null 2>&1; then
      telemetry_ready=1
    fi

    if kubectl get deployment istiod -n "${ISTIO_NAMESPACE}" >/dev/null 2>&1; then
      if kubectl rollout status deployment/istiod -n "${ISTIO_NAMESPACE}" --timeout=20s >/dev/null 2>&1; then
        istiod_ready=1
      fi
    fi

    if [ "${telemetry_ready}" = "1" ] && [ "${istiod_ready}" = "1" ]; then
      log "Istio CRDs and istiod are ready"
      return 0
    fi

    sleep 5
  done

  kubectl get crd | grep istio.io || true
  kubectl get pods -n "${ISTIO_NAMESPACE}" || true

  if [ "${telemetry_ready}" != "1" ]; then
    die "Istio install completed but telemetries.telemetry.istio.io did not become available in time"
  fi

  if [ "${istiod_ready}" != "1" ]; then
    die "Istio install completed but istiod did not become ready in time"
  fi
}
ensure_cluster_prereqs() {
  log "Ensuring cluster prerequisites"
  ensure_gateway_api_crds
  ensure_istio
}

apply_storage_manifests() {
  log "Applying storage manifests"
  kubectl get namespace "${FASTAPI_NAMESPACE}" >/dev/null 2>&1 || kubectl create namespace "${FASTAPI_NAMESPACE}"
  kubectl apply -f "${KERNEL_ROOT}/deploy/k8s-storage/base/mongo.yaml"
  kubectl apply -f "${KERNEL_ROOT}/deploy/k8s-storage/base/milvus.yaml"
  kubectl apply -f "${KERNEL_ROOT}/deploy/k8s-storage/base/seaweedfs.yaml"
}

wait_storage_rollouts() {
  log "Waiting for storage deployments"
  kubectl rollout status deployment/mongodb -n "${FASTAPI_NAMESPACE}" --timeout="${STORAGE_TIMEOUT_S}s"
  kubectl rollout status deployment/milvus-etcd -n "${FASTAPI_NAMESPACE}" --timeout="${STORAGE_TIMEOUT_S}s"
  kubectl rollout status deployment/milvus-minio -n "${FASTAPI_NAMESPACE}" --timeout="${STORAGE_TIMEOUT_S}s"
  kubectl rollout status deployment/milvus -n "${FASTAPI_NAMESPACE}" --timeout="${STORAGE_TIMEOUT_S}s"
  kubectl rollout status deployment/seaweed-master -n "${FASTAPI_NAMESPACE}" --timeout="${STORAGE_TIMEOUT_S}s"
  kubectl rollout status deployment/seaweed-volume -n "${FASTAPI_NAMESPACE}" --timeout="${STORAGE_TIMEOUT_S}s"
  kubectl rollout status deployment/seaweed-filer -n "${FASTAPI_NAMESPACE}" --timeout="${STORAGE_TIMEOUT_S}s"
  kubectl rollout status deployment/seaweed-s3 -n "${FASTAPI_NAMESPACE}" --timeout="${STORAGE_TIMEOUT_S}s"
}

deploy_llmd_stack() {
  log "Deploying llm-d stack"
  MINIKUBE_PROFILE="${MINIKUBE_PROFILE}" bash "${KERNEL_ROOT}/scripts/deploy_llmd_local.sh"
}

wait_llmd_ready() {
  log "Waiting for llm-d /v1/models"
  start_port_forward "${LLMD_NAMESPACE}" "${LLMD_SERVICE}" "${LLMD_LOCAL_PORT}" 80
  wait_http_ok "http://127.0.0.1:${LLMD_LOCAL_PORT}/v1/models" "${LLMD_READY_TIMEOUT_S}" 200 >/dev/null
  log "llm-d is reachable"
}

apply_fastapi_resources() {
  log "Applying FastAPI resources"
  kubectl get namespace "${FASTAPI_NAMESPACE}" >/dev/null 2>&1 || kubectl create namespace "${FASTAPI_NAMESPACE}"

  kubectl apply -f "${KERNEL_ROOT}/deploy/k8s-fastapi/base/rag-store-tenants-pvc.yaml"
  kubectl apply -f "${KERNEL_ROOT}/deploy/k8s-fastapi/base/tenant-ingest-input-pvc.yaml"
  kubectl apply -f "${KERNEL_ROOT}/deploy/k8s-fastapi/base/fastapi-configmap.fullstack.yaml"
  kubectl apply -f "${KERNEL_ROOT}/deploy/k8s-fastapi/base/fastapi-secret.fullstack.yaml"
  kubectl apply -f "${KERNEL_ROOT}/deploy/k8s-fastapi/base/fastapi-service.yaml"

  if [ "${DEPLOY_JAEGER}" = "1" ]; then
    log "Deploying Jaeger (DEPLOY_JAEGER=1)"
    kubectl apply -f "${KERNEL_ROOT}/deploy/k8s-fastapi/base/jaeger.yaml"
    kubectl patch configmap llm-service-kernel-config-fullstack \
      -n "${FASTAPI_NAMESPACE}" \
      --type=merge \
      -p '{"data":{"OTEL_EXPORTER_OTLP_ENDPOINT":"jaeger.llm-service.svc.cluster.local:4317"}}'
    log "Jaeger deployed — UI: kubectl port-forward svc/jaeger 16686:16686 -n ${FASTAPI_NAMESPACE}"
  fi
}

show_bootstrap_diagnostics() {
  echo
  log "Bootstrap diagnostics"
  kubectl get pods -n "${FASTAPI_NAMESPACE}" -l job-name="${BOOTSTRAP_JOB}" -o wide || true
  echo
  kubectl describe job "${BOOTSTRAP_JOB}" -n "${FASTAPI_NAMESPACE}" || true
  echo
  kubectl logs -n "${FASTAPI_NAMESPACE}" "job/${BOOTSTRAP_JOB}" --all-containers=true || true
  echo
}

wait_milvus_grpc_ready() {
  log "Waiting for Milvus gRPC (port 19530) to accept requests"
  local timeout_s=360
  local deadline=$((SECONDS + timeout_s))
  local probe_pod="milvus-grpc-probe"

  kubectl delete pod "${probe_pod}" -n "${FASTAPI_NAMESPACE}" --ignore-not-found=true --wait=true >/dev/null 2>&1

  until kubectl run "${probe_pod}" \
    --image="${FASTAPI_IMAGE}" \
    --image-pull-policy=Never \
    --restart=Never \
    --rm \
    --attach \
    -n "${FASTAPI_NAMESPACE}" \
    --timeout=30s \
    -- python3 -c "
from pymilvus import connections, utility
connections.connect(uri='http://milvus:19530', user='root', password='Milvus', timeout=5)
utility.has_collection('_probe_', timeout=5)
" 2>/dev/null
  do
    kubectl delete pod "${probe_pod}" -n "${FASTAPI_NAMESPACE}" --ignore-not-found=true >/dev/null 2>&1
    if [ "${SECONDS}" -ge "${deadline}" ]; then
      die "Milvus gRPC not ready after ${timeout_s}s"
    fi
    log "Milvus gRPC not ready yet, retrying in 5s..."
    sleep 5
  done
  log "Milvus gRPC is ready"
}

run_bootstrap_job() {
  log "Running bootstrap job"
  kubectl delete job "${BOOTSTRAP_JOB}" -n "${FASTAPI_NAMESPACE}" --ignore-not-found=true --wait=true
  kubectl apply -f "${KERNEL_ROOT}/deploy/k8s-fastapi/base/fastapi-bootstrap-job.yaml"

  if ! kubectl wait --for=condition=complete "job/${BOOTSTRAP_JOB}" -n "${FASTAPI_NAMESPACE}" --timeout="${BOOTSTRAP_TIMEOUT_S}s"; then
    show_bootstrap_diagnostics
    die "bootstrap job failed"
  fi

  kubectl logs -n "${FASTAPI_NAMESPACE}" "job/${BOOTSTRAP_JOB}" --all-containers=true || true
}

deploy_fastapi() {
  log "Deploying FastAPI"
  kubectl apply -f "${KERNEL_ROOT}/deploy/k8s-fastapi/base/fastapi-deployment.fullstack.yaml"
  kubectl rollout restart deployment/"${FASTAPI_DEPLOYMENT}" -n "${FASTAPI_NAMESPACE}"
  kubectl rollout status deployment/"${FASTAPI_DEPLOYMENT}" -n "${FASTAPI_NAMESPACE}" --timeout=600s
}

wait_fastapi_ready() {
  log "Waiting for FastAPI /ready"
  start_port_forward "${FASTAPI_NAMESPACE}" "${FASTAPI_SERVICE}" "${FASTAPI_LOCAL_PORT}" 8080
  wait_http_ok "http://127.0.0.1:${FASTAPI_LOCAL_PORT}/ready" "${FASTAPI_READY_TIMEOUT_S}" 200 >/dev/null
  log "FastAPI /ready is healthy"
}

run_validation() {
  log "Running validation"
  AUTO_PORT_FORWARD=0 \
  LLMD_BASE_URL="http://127.0.0.1:${LLMD_LOCAL_PORT}" \
  FASTAPI_BASE_URL="http://127.0.0.1:${FASTAPI_LOCAL_PORT}" \
  bash "${KERNEL_ROOT}/scripts/validate_fullstack_single_node.sh"
}

print_summary() {
  echo
  log "Deployment summary"
  kubectl get pods,svc -n "${ISTIO_NAMESPACE}" -o wide || true
  echo
  kubectl get pods,svc -n "${LLMD_NAMESPACE}" -o wide || true
  echo
  kubectl get pods,svc -n "${FASTAPI_NAMESPACE}" -o wide || true
  echo

  cat <<EOF
Useful checks:

  kubectl get crd gateways.gateway.networking.k8s.io
  kubectl get crd telemetries.telemetry.istio.io
  kubectl logs -n ${FASTAPI_NAMESPACE} deploy/${FASTAPI_DEPLOYMENT} --tail=200
  kubectl logs -n ${FASTAPI_NAMESPACE} job/${BOOTSTRAP_JOB}
  curl http://127.0.0.1:${FASTAPI_LOCAL_PORT}/health | python3 -m json.tool
  curl -i http://127.0.0.1:${FASTAPI_LOCAL_PORT}/ready

Manual port-forward:
  kubectl port-forward -n ${FASTAPI_NAMESPACE} svc/${FASTAPI_SERVICE} ${FASTAPI_LOCAL_PORT}:8080

CLI:
  python3 scripts/chat_cli.py --show-debug
EOF
}

main() {
  require_cmd bash
  require_cmd kubectl
  require_cmd docker
  require_cmd python3

  if [ "${INSTALL_CLUSTER_PREREQS}" = "1" ]; then
    ensure_cluster_prereqs
  else
    log "Skipping cluster prerequisite install/check"
  fi

  if [ "${PREPARE_ASSETS}" = "1" ]; then
    prepare_assets
  else
    log "Skipping runtime asset preparation"
  fi

  if [ "${BUILD_FASTAPI_IMAGE}" = "1" ]; then
    build_fastapi_image
  else
    log "Skipping FastAPI image build"
  fi

  if [ "${APPLY_STORAGE}" = "1" ]; then
    apply_storage_manifests
  else
    log "Skipping storage manifest apply"
  fi

  if [ "${WAIT_STORAGE}" = "1" ]; then
    wait_storage_rollouts
  else
    log "Skipping storage rollout wait"
  fi

  if [ "${DEPLOY_LLMD}" = "1" ]; then
    deploy_llmd_stack
  else
    log "Skipping llm-d deployment"
  fi

  if [ "${WAIT_LLMD}" = "1" ]; then
    wait_llmd_ready
  else
    log "Skipping llm-d readiness wait"
  fi

  if [ "${APPLY_FASTAPI_RESOURCES}" = "1" ]; then
    apply_fastapi_resources
  else
    log "Skipping FastAPI resource apply"
  fi

  if [ "${RUN_BOOTSTRAP_JOB}" = "1" ]; then
    wait_milvus_grpc_ready
    run_bootstrap_job
  else
    log "Skipping bootstrap job"
  fi

  if [ "${DEPLOY_FASTAPI}" = "1" ]; then
    deploy_fastapi
  else
    log "Skipping FastAPI deployment"
  fi

  if [ "${WAIT_FASTAPI_READY}" = "1" ]; then
    wait_fastapi_ready
  else
    log "Skipping FastAPI readiness wait"
  fi

  if [ "${RUN_VALIDATE}" = "1" ]; then
    run_validation
  else
    log "Skipping validation"
  fi

  print_summary
}

main "$@"
