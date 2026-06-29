#!/usr/bin/env bash
set -euo pipefail

KERNEL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${MANIFEST:-${KERNEL_ROOT}/scripts/required_images.txt}"
MINIKUBE_PROFILE="${MINIKUBE_PROFILE:-minikube}"

LOAD_TO_MINIKUBE="${LOAD_TO_MINIKUBE:-0}"
PULL_OPTIONAL_VLLM="${PULL_OPTIONAL_VLLM:-0}"

# llm-d image handling
LOAD_LLMD_TARS="${LOAD_LLMD_TARS:-1}"
PREPULL_LLMD_ONLINE="${PREPULL_LLMD_ONLINE:-0}"

LLMD_DIR="${KERNEL_ROOT}/deploy/llmd-local"
LLMD_ROUTING_IMAGE_TARGET="${LLMD_ROUTING_IMAGE_TARGET:-ghcr.io/llm-d/llm-d-routing-sidecar:offline}"
LLMD_CUDA_IMAGE_TARGET="${LLMD_CUDA_IMAGE_TARGET:-ghcr.io/llm-d/llm-d-cuda:offline}"

# Set these only if you want to pull llm-d images from the internet and retag them locally.
LLMD_ROUTING_IMAGE_SOURCE="${LLMD_ROUTING_IMAGE_SOURCE:-}"
LLMD_CUDA_IMAGE_SOURCE="${LLMD_CUDA_IMAGE_SOURCE:-}"

log() {
  printf '[pull_required_images] %s\n' "$*"
}

die() {
  printf '[pull_required_images] ERROR: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

trim() {
  echo "$1" | xargs
}

declare -A WANT_IMAGES=()

add_image() {
  local image="$1"
  [ -n "${image}" ] || return 0
  WANT_IMAGES["${image}"]=1
}

have_image() {
  local image="$1"
  docker image inspect "${image}" >/dev/null 2>&1
}

pull_if_missing() {
  local image="$1"
  if have_image "${image}"; then
    log "Already present: ${image}"
  else
    log "Pulling: ${image}"
    docker pull "${image}"
  fi
  add_image "${image}"
}

load_tar_if_exists() {
  local tar_path="$1"
  if [ -f "${tar_path}" ]; then
    log "Loading tar: ${tar_path}"
    docker load -i "${tar_path}"
  fi
}

pull_manifest_images() {
  [ -f "${MANIFEST}" ] || die "manifest not found: ${MANIFEST}"

  while IFS= read -r raw_line; do
    local line
    line="${raw_line%%#*}"
    line="$(trim "${line}")"
    [ -n "${line}" ] || continue

    if [ "${line}" = "vllm/vllm-openai:latest" ] && [ "${PULL_OPTIONAL_VLLM}" != "1" ]; then
      log "Skipping optional image: ${line}"
      continue
    fi

    pull_if_missing "${line}"
  done < "${MANIFEST}"
}

load_llmd_tars_if_present() {
  [ "${LOAD_LLMD_TARS}" = "1" ] || return 0

  if [ -f "${LLMD_DIR}/gaie-images.tar" ]; then
    load_tar_if_exists "${LLMD_DIR}/gaie-images.tar"
  fi

  if [ -d "${LLMD_DIR}/images" ]; then
    while IFS= read -r -d '' tar_file; do
      load_tar_if_exists "${tar_file}"
    done < <(find "${LLMD_DIR}/images" -maxdepth 1 -type f -name '*.tar' -print0 | sort -z)
  fi

  for image in \
    "${LLMD_ROUTING_IMAGE_TARGET}" \
    "${LLMD_CUDA_IMAGE_TARGET}" \
    "registry.k8s.io/gateway-api-inference-extension/epp:v1.3.1" \
    "docker.io/istio/proxyv2:1.28.1"
  do
    if have_image "${image}"; then
      add_image "${image}"
    fi
  done
}

pull_llmd_online_if_requested() {
  [ "${PREPULL_LLMD_ONLINE}" = "1" ] || return 0

  [ -n "${LLMD_ROUTING_IMAGE_SOURCE}" ] || die "PREPULL_LLMD_ONLINE=1 but LLMD_ROUTING_IMAGE_SOURCE is empty"
  [ -n "${LLMD_CUDA_IMAGE_SOURCE}" ] || die "PREPULL_LLMD_ONLINE=1 but LLMD_CUDA_IMAGE_SOURCE is empty"

  pull_if_missing "${LLMD_ROUTING_IMAGE_SOURCE}"
  if [ "${LLMD_ROUTING_IMAGE_SOURCE}" != "${LLMD_ROUTING_IMAGE_TARGET}" ]; then
    log "Tagging ${LLMD_ROUTING_IMAGE_SOURCE} -> ${LLMD_ROUTING_IMAGE_TARGET}"
    docker tag "${LLMD_ROUTING_IMAGE_SOURCE}" "${LLMD_ROUTING_IMAGE_TARGET}"
  fi
  add_image "${LLMD_ROUTING_IMAGE_TARGET}"

  pull_if_missing "${LLMD_CUDA_IMAGE_SOURCE}"
  if [ "${LLMD_CUDA_IMAGE_SOURCE}" != "${LLMD_CUDA_IMAGE_TARGET}" ]; then
    log "Tagging ${LLMD_CUDA_IMAGE_SOURCE} -> ${LLMD_CUDA_IMAGE_TARGET}"
    docker tag "${LLMD_CUDA_IMAGE_SOURCE}" "${LLMD_CUDA_IMAGE_TARGET}"
  fi
  add_image "${LLMD_CUDA_IMAGE_TARGET}"
}

load_images_to_minikube() {
  [ "${LOAD_TO_MINIKUBE}" = "1" ] || return 0

  require_cmd minikube

  while IFS= read -r image; do
    [ -n "${image}" ] || continue
    if have_image "${image}"; then
      log "Loading into minikube: ${image}"
      minikube -p "${MINIKUBE_PROFILE}" image load "${image}"
    else
      log "Skipping missing local image during minikube load: ${image}"
    fi
  done < <(printf '%s\n' "${!WANT_IMAGES[@]}" | sort -u)
}

print_summary() {
  echo
  log "Images prepared for this repo:"
  printf '%s\n' "${!WANT_IMAGES[@]}" | sort -u
  echo
}

main() {
  require_cmd docker

  pull_manifest_images
  load_llmd_tars_if_present
  pull_llmd_online_if_requested
  load_images_to_minikube
  print_summary
}

main "$@"
