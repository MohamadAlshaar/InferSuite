#!/usr/bin/env bash
set -euo pipefail

KERNEL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${KERNEL_ROOT}/.." && pwd)"

MINIKUBE_PROFILE="${MINIKUBE_PROFILE:-minikube}"
MODEL_SOURCE_DIR="${MODEL_SOURCE_DIR:-${REPO_ROOT}/Qwen2.5-0.5B-Instruct}"
MODEL_TARGET_DIR="${MODEL_TARGET_DIR:-/data/qwen-model}"
RESET_TARGET="${RESET_TARGET:-0}"

ARCHIVE_BASENAME="qwen-model-copy.tgz"
LOCAL_TMP_ARCHIVE="/tmp/${ARCHIVE_BASENAME}"
NODE_TMP_ARCHIVE="/tmp/${ARCHIVE_BASENAME}"

log() {
  printf '[provision_model_artifacts] %s\n' "$*"
}

die() {
  printf '[provision_model_artifacts] ERROR: %s\n' "$*" >&2
  exit 1
}

cleanup() {
  rm -f "${LOCAL_TMP_ARCHIVE}" || true
  minikube -p "${MINIKUBE_PROFILE}" ssh -- "rm -f '${NODE_TMP_ARCHIVE}'" >/dev/null 2>&1 || true
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

trap cleanup EXIT

require_cmd minikube
require_cmd tar

[ -d "${MODEL_SOURCE_DIR}" ] || die "model source directory not found: ${MODEL_SOURCE_DIR}"
[ -f "${MODEL_SOURCE_DIR}/config.json" ] || die "missing file: ${MODEL_SOURCE_DIR}/config.json"
[ -f "${MODEL_SOURCE_DIR}/model.safetensors" ] || die "missing file: ${MODEL_SOURCE_DIR}/model.safetensors"

log "Using source model directory: ${MODEL_SOURCE_DIR}"
log "Target node directory: ${MODEL_TARGET_DIR}"

if [ "${RESET_TARGET}" = "1" ]; then
  log "Resetting target directory on minikube node"
  minikube -p "${MINIKUBE_PROFILE}" ssh -- "sudo rm -rf '${MODEL_TARGET_DIR}' && sudo mkdir -p '${MODEL_TARGET_DIR}'"
else
  minikube -p "${MINIKUBE_PROFILE}" ssh -- "sudo mkdir -p '${MODEL_TARGET_DIR}'"
fi

log "Creating local archive"
tar -C "${MODEL_SOURCE_DIR}" -czf "${LOCAL_TMP_ARCHIVE}" .

log "Copying archive to minikube node"
minikube -p "${MINIKUBE_PROFILE}" cp "${LOCAL_TMP_ARCHIVE}" "${NODE_TMP_ARCHIVE}"

log "Extracting archive on minikube node"
minikube -p "${MINIKUBE_PROFILE}" ssh -- "sudo tar -xzf '${NODE_TMP_ARCHIVE}' -C '${MODEL_TARGET_DIR}'"

log "Verifying copied files"
minikube -p "${MINIKUBE_PROFILE}" ssh -- "sudo test -f '${MODEL_TARGET_DIR}/config.json' && sudo test -f '${MODEL_TARGET_DIR}/model.safetensors' && sudo du -sh '${MODEL_TARGET_DIR}' && sudo ls -lah '${MODEL_TARGET_DIR}' | head -20"

log "Model provisioning complete"
