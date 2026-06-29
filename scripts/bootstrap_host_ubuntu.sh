#!/usr/bin/env bash
set -euo pipefail

KERNEL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export DEBIAN_FRONTEND=noninteractive

MINIKUBE_PROFILE="${MINIKUBE_PROFILE:-minikube}"
MINIKUBE_DRIVER="${MINIKUBE_DRIVER:-docker}"
MINIKUBE_CONTAINER_RUNTIME="${MINIKUBE_CONTAINER_RUNTIME:-docker}"
KUBERNETES_VERSION="${KUBERNETES_VERSION:-v1.33.1}"
KUBECTL_VERSION="${KUBECTL_VERSION:-${KUBERNETES_VERSION}}"
MINIKUBE_VERSION="${MINIKUBE_VERSION:-v1.38.1}"
GATEWAY_API_VERSION="${GATEWAY_API_VERSION:-v1.5.1}"
ISTIO_VERSION="${ISTIO_VERSION:-1.29.0}"
ISTIO_PROFILE="${ISTIO_PROFILE:-default}"

MINIKUBE_CPUS="${MINIKUBE_CPUS:-8}"
MINIKUBE_MEMORY="${MINIKUBE_MEMORY:-16384}"
MINIKUBE_DISK_SIZE="${MINIKUBE_DISK_SIZE:-80g}"
MINIKUBE_EXTRA_START_FLAGS="${MINIKUBE_EXTRA_START_FLAGS:-}"

INSTALL_DOCKER="${INSTALL_DOCKER:-1}"
INSTALL_NVIDIA_TOOLKIT="${INSTALL_NVIDIA_TOOLKIT:-1}"
INSTALL_KUBECTL="${INSTALL_KUBECTL:-1}"
INSTALL_HELM="${INSTALL_HELM:-1}"
INSTALL_MINIKUBE="${INSTALL_MINIKUBE:-1}"
INSTALL_ISTIOCTL="${INSTALL_ISTIOCTL:-1}"
INSTALL_PERF="${INSTALL_PERF:-1}"

START_MINIKUBE="${START_MINIKUBE:-1}"
ENABLE_MINIKUBE_STORAGE_ADDONS="${ENABLE_MINIKUBE_STORAGE_ADDONS:-1}"
APPLY_GATEWAY_API="${APPLY_GATEWAY_API:-1}"
INSTALL_ISTIO="${INSTALL_ISTIO:-1}"
CREATE_NAMESPACES="${CREATE_NAMESPACES:-1}"
PRELOAD_IMAGES="${PRELOAD_IMAGES:-0}"

REQUIRE_NVIDIA_SMI="${REQUIRE_NVIDIA_SMI:-1}"
HELM_INSTALLER_URL="${HELM_INSTALLER_URL:-https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3}"
GATEWAY_API_STANDARD_INSTALL_URL="${GATEWAY_API_STANDARD_INSTALL_URL:-https://github.com/kubernetes-sigs/gateway-api/releases/download/${GATEWAY_API_VERSION}/standard-install.yaml}"
PULL_REQUIRED_IMAGES_SCRIPT="${PULL_REQUIRED_IMAGES_SCRIPT:-${KERNEL_ROOT}/scripts/pull_required_images.sh}"

log() {
  printf '[bootstrap_host_ubuntu] %s\n' "$*"
}

die() {
  printf '[bootstrap_host_ubuntu] ERROR: %s\n' "$*" >&2
  exit 1
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

need_cmd() {
  have_cmd "$1" || die "missing required command: $1"
}

sudo_cmd() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  else
    sudo "$@"
  fi
}

retry() {
  local attempts="$1"
  local sleep_s="$2"
  shift 2

  local i
  for ((i=1; i<=attempts; i++)); do
    if "$@"; then
      return 0
    fi
    if [ "$i" -lt "$attempts" ]; then
      sleep "$sleep_s"
    fi
  done
  return 1
}

wait_for_apt_lock() {
  local timeout_s="${1:-300}"
  local start_ts
  start_ts="$(date +%s)"

  while true; do
    if sudo_cmd bash -lc '! fuser /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock /var/lib/apt/lists/lock >/dev/null 2>&1'; then
      return 0
    fi

    if [ $(( "$(date +%s)" - start_ts )) -ge "$timeout_s" ]; then
      die "timed out waiting for apt/dpkg lock"
    fi

    log "waiting for apt/dpkg lock to be released..."
    sleep 5
  done
}

apt_update() {
  wait_for_apt_lock
  sudo_cmd apt-get update -y
}

apt_install() {
  wait_for_apt_lock
  sudo_cmd apt-get install -y --no-install-recommends "$@"
}

require_ubuntu() {
  [ -f /etc/os-release ] || die "/etc/os-release not found"
  # shellcheck disable=SC1091
  . /etc/os-release
  [ "${ID:-}" = "ubuntu" ] || die "this script is intended for Ubuntu; found ID=${ID:-unknown}"
  [ -n "${VERSION_CODENAME:-}" ] || die "missing VERSION_CODENAME in /etc/os-release"
}

get_arch() {
  local arch
  arch="$(uname -m)"
  case "$arch" in
    x86_64|amd64) echo "amd64" ;;
    aarch64|arm64) echo "arm64" ;;
    *) die "unsupported architecture: $arch" ;;
  esac
}

download_file() {
  local url="$1"
  local out="$2"
  retry 3 3 curl -fsSL "$url" -o "$out" || die "failed to download: $url"
}

ensure_base_packages() {
  apt_update
  apt_install ca-certificates curl gnupg lsb-release apt-transport-https jq tar
}

install_docker() {
  if have_cmd docker; then
    log "docker already installed"
  else
    log "installing Docker Engine"
    ensure_base_packages

    sudo_cmd install -m 0755 -d /etc/apt/keyrings
    sudo_cmd curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    sudo_cmd chmod a+r /etc/apt/keyrings/docker.asc

    # shellcheck disable=SC1091
    . /etc/os-release
    local arch
    arch="$(dpkg --print-architecture)"

    sudo_cmd tee /etc/apt/sources.list.d/docker.sources >/dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: ${VERSION_CODENAME}
Components: stable
Architectures: ${arch}
Signed-By: /etc/apt/keyrings/docker.asc
EOF

    apt_update
    apt_install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  fi

  sudo_cmd systemctl enable --now docker

  if [ "$(id -u)" -ne 0 ]; then
    if ! groups "${USER}" | grep -q '\bdocker\b'; then
      log "adding ${USER} to docker group"
      sudo_cmd usermod -aG docker "${USER}" || true
    fi
  fi

  docker version >/dev/null 2>&1 || sudo_cmd docker version >/dev/null 2>&1 || die "docker is installed but not usable"
}

install_nvidia_container_toolkit() {
  if ! have_cmd nvidia-smi; then
    if [ "${REQUIRE_NVIDIA_SMI}" = "1" ]; then
      die "nvidia-smi not found; install the NVIDIA driver first"
    fi
    log "nvidia-smi not found; skipping NVIDIA Container Toolkit"
    return 0
  fi

  if have_cmd nvidia-ctk; then
    log "nvidia-ctk already installed"
  else
    log "installing NVIDIA Container Toolkit"
    ensure_base_packages

    sudo_cmd mkdir -p /usr/share/keyrings
    sudo_cmd bash -lc \
      "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
       gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"

    sudo_cmd bash -lc \
      "curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
       sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
       tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null"

    apt_update
    apt_install nvidia-container-toolkit nvidia-container-toolkit-base libnvidia-container-tools libnvidia-container1
  fi

  log "configuring Docker to use NVIDIA runtime"
  sudo_cmd nvidia-ctk runtime configure --runtime=docker
  sudo_cmd systemctl restart docker
}

install_kubectl() {
  if have_cmd kubectl; then
    log "kubectl already installed"
    return 0
  fi

  log "installing kubectl ${KUBECTL_VERSION}"
  local arch tmpdir
  arch="$(get_arch)"
  tmpdir="$(mktemp -d)"
  trap 'rm -rf "${tmpdir}"' RETURN

  download_file "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/${arch}/kubectl" "${tmpdir}/kubectl"
  download_file "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/${arch}/kubectl.sha256" "${tmpdir}/kubectl.sha256"

  (
    cd "${tmpdir}"
    echo "$(cat kubectl.sha256)  kubectl" | sha256sum --check
  ) || die "kubectl checksum verification failed"

  sudo_cmd install -o root -g root -m 0755 "${tmpdir}/kubectl" /usr/local/bin/kubectl
  trap - RETURN
  rm -rf "${tmpdir}"
}

install_helm() {
  if have_cmd helm; then
    log "helm already installed"
    return 0
  fi

  log "installing Helm"
  local tmpdir
  tmpdir="$(mktemp -d)"
  trap 'rm -rf "${tmpdir}"' RETURN

  download_file "${HELM_INSTALLER_URL}" "${tmpdir}/get_helm.sh"
  chmod 700 "${tmpdir}/get_helm.sh"
  sudo_cmd env USE_SUDO="false" bash "${tmpdir}/get_helm.sh"

  trap - RETURN
  rm -rf "${tmpdir}"
}

install_minikube() {
  if have_cmd minikube; then
    log "minikube already installed"
    return 0
  fi

  log "installing minikube ${MINIKUBE_VERSION}"
  local arch tmpdir
  arch="$(get_arch)"
  tmpdir="$(mktemp -d)"
  trap 'rm -rf "${tmpdir}"' RETURN

  download_file "https://github.com/kubernetes/minikube/releases/download/${MINIKUBE_VERSION}/minikube-linux-${arch}" "${tmpdir}/minikube"
  chmod +x "${tmpdir}/minikube"
  sudo_cmd install -o root -g root -m 0755 "${tmpdir}/minikube" /usr/local/bin/minikube

  trap - RETURN
  rm -rf "${tmpdir}"
}

install_istioctl() {
  if have_cmd istioctl; then
    log "istioctl already installed"
    return 0
  fi

  log "installing istioctl ${ISTIO_VERSION}"
  local arch tmpdir release_dir
  arch="$(get_arch)"
  tmpdir="$(mktemp -d)"
  trap 'rm -rf "${tmpdir}"' RETURN

  (
    cd "${tmpdir}"
    curl -fsSL https://istio.io/downloadIstio | ISTIO_VERSION="${ISTIO_VERSION}" TARGET_ARCH="${arch}" sh -
  ) || die "failed to download Istio"

  release_dir="${tmpdir}/istio-${ISTIO_VERSION}"
  [ -x "${release_dir}/bin/istioctl" ] || die "istioctl binary not found after download"

  sudo_cmd install -o root -g root -m 0755 "${release_dir}/bin/istioctl" /usr/local/bin/istioctl

  trap - RETURN
  rm -rf "${tmpdir}"
}

install_perf() {
  if find /usr/lib/linux-tools* -maxdepth 2 -name perf -type f -executable 2>/dev/null | grep -q .; then
    log "perf already installed"
    return 0
  fi

  log "installing perf (linux-tools)"
  apt_update
  # Install kernel-specific package first (most accurate PMU counters), fall back to generic
  apt_install "linux-tools-$(uname -r)" 2>/dev/null \
    || apt_install linux-tools-generic \
    || { log "WARNING: perf install failed — benchmark CPU counters will not work"; return 0; }
  # Also install generic to ensure the 'perf' symlink exists in PATH
  apt_install linux-tools-generic 2>/dev/null || true
}

start_minikube_cluster() {
  log "starting minikube profile=${MINIKUBE_PROFILE}"
  minikube config set driver "${MINIKUBE_DRIVER}" >/dev/null

  if [ -n "${MINIKUBE_EXTRA_START_FLAGS}" ]; then
    # shellcheck disable=SC2086
    minikube start \
      -p "${MINIKUBE_PROFILE}" \
      --driver="${MINIKUBE_DRIVER}" \
      --container-runtime="${MINIKUBE_CONTAINER_RUNTIME}" \
      --kubernetes-version="${KUBERNETES_VERSION}" \
      --cpus="${MINIKUBE_CPUS}" \
      --memory="${MINIKUBE_MEMORY}" \
      --disk-size="${MINIKUBE_DISK_SIZE}" \
      ${MINIKUBE_EXTRA_START_FLAGS}
  else
    minikube start \
      -p "${MINIKUBE_PROFILE}" \
      --driver="${MINIKUBE_DRIVER}" \
      --container-runtime="${MINIKUBE_CONTAINER_RUNTIME}" \
      --kubernetes-version="${KUBERNETES_VERSION}" \
      --cpus="${MINIKUBE_CPUS}" \
      --memory="${MINIKUBE_MEMORY}" \
      --disk-size="${MINIKUBE_DISK_SIZE}"
  fi

  kubectl config use-context "${MINIKUBE_PROFILE}" >/dev/null 2>&1 || true
}

enable_minikube_storage_addons() {
  log "enabling minikube storage addons"
  minikube addons enable default-storageclass -p "${MINIKUBE_PROFILE}"
  minikube addons enable storage-provisioner -p "${MINIKUBE_PROFILE}"
}

apply_gateway_api_crds() {
  log "applying Gateway API standard CRDs from ${GATEWAY_API_STANDARD_INSTALL_URL}"
  kubectl apply -f "${GATEWAY_API_STANDARD_INSTALL_URL}"
}

install_istio_control_plane() {
  log "installing Istio profile=${ISTIO_PROFILE}"
  istioctl install -y --set profile="${ISTIO_PROFILE}"
  kubectl rollout status deployment/istiod -n istio-system --timeout=300s
}

create_namespaces() {
  for ns in llm-d-local llm-service; do
    kubectl get namespace "${ns}" >/dev/null 2>&1 || kubectl create namespace "${ns}"
  done
}

preload_images_if_requested() {
  if [ "${PRELOAD_IMAGES}" != "1" ]; then
    log "skipping image preload"
    return 0
  fi

  if [ -x "${PULL_REQUIRED_IMAGES_SCRIPT}" ]; then
    log "preloading images using ${PULL_REQUIRED_IMAGES_SCRIPT}"
    LOAD_TO_MINIKUBE=1 MINIKUBE_PROFILE="${MINIKUBE_PROFILE}" bash "${PULL_REQUIRED_IMAGES_SCRIPT}"
    return 0
  fi

  log "PRELOAD_IMAGES=1 but pull_required_images.sh not found or not executable; skipping"
}

print_versions() {
  echo
  log "version summary"

  if have_cmd docker; then
    docker --version || true
    docker compose version || true
  fi

  if have_cmd nvidia-ctk; then
    nvidia-ctk --version || true
  fi

  if have_cmd nvidia-smi; then
    nvidia-smi || true
  fi

  if have_cmd minikube; then
    minikube version || true
  fi

  if have_cmd kubectl; then
    kubectl version --client || true
  fi

  if have_cmd helm; then
    helm version || true
  fi

  if have_cmd istioctl; then
    istioctl version --remote=false || true
  fi
}

run_final_checks() {
  echo
  log "final checks"

  if have_cmd minikube; then
    minikube status -p "${MINIKUBE_PROFILE}" || die "minikube status failed"
  fi

  if have_cmd kubectl; then
    kubectl get nodes || die "kubectl get nodes failed"
    kubectl get crd gateways.gateway.networking.k8s.io >/dev/null 2>&1 || die "Gateway API CRDs not found"
    kubectl get storageclass || die "kubectl get storageclass failed"
  fi
}

main() {
  require_ubuntu
  need_cmd curl
  need_cmd sed
  need_cmd sha256sum

  if [ "${INSTALL_DOCKER}" = "1" ]; then
    install_docker
  else
    log "skipping Docker installation"
  fi

  if [ "${INSTALL_NVIDIA_TOOLKIT}" = "1" ]; then
    install_nvidia_container_toolkit
  else
    log "skipping NVIDIA Container Toolkit installation"
  fi

  if [ "${INSTALL_KUBECTL}" = "1" ]; then
    install_kubectl
  else
    log "skipping kubectl installation"
  fi

  if [ "${INSTALL_HELM}" = "1" ]; then
    install_helm
  else
    log "skipping Helm installation"
  fi

  if [ "${INSTALL_MINIKUBE}" = "1" ]; then
    install_minikube
  else
    log "skipping minikube installation"
  fi

  if [ "${INSTALL_ISTIOCTL}" = "1" ]; then
    install_istioctl
  else
    log "skipping istioctl installation"
  fi

  if [ "${INSTALL_PERF}" = "1" ]; then
    install_perf
  else
    log "skipping perf installation"
  fi

  if [ "${START_MINIKUBE}" = "1" ]; then
    start_minikube_cluster
  else
    log "skipping minikube start"
  fi

  if [ "${ENABLE_MINIKUBE_STORAGE_ADDONS}" = "1" ]; then
    enable_minikube_storage_addons
  else
    log "skipping minikube storage addons"
  fi

  if [ "${APPLY_GATEWAY_API}" = "1" ]; then
    apply_gateway_api_crds
  else
    log "skipping Gateway API CRD installation"
  fi

  if [ "${INSTALL_ISTIO}" = "1" ]; then
    install_istio_control_plane
  else
    log "skipping Istio installation"
  fi

  if [ "${CREATE_NAMESPACES}" = "1" ]; then
    create_namespaces
  else
    log "skipping namespace creation"
  fi

  preload_images_if_requested

  print_versions
  run_final_checks

  echo
  log "bootstrap complete"
  log "note: if Docker was just installed and your user was newly added to the docker group, open a new shell before relying on non-sudo docker commands"
}

main "$@"