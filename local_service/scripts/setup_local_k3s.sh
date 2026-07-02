#!/usr/bin/env bash
# setup_local_k3s.sh — phase 1 of the LOCAL service run (ThinkStation P7, A2000, full PMU):
# install single-node k3s + nvidia runtime + storage class + device plugin, stage the 7B-AWQ
# model onto a hostPath PV, build the FastAPI image with host docker and import it into k3s
# containerd. Idempotent; safe to re-run. Does NOT touch ~/.kube/config (dedicated kubeconfig).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
H100_DEPLOY="$REPO/h100/service/k3s_deploy"
export KUBECONFIG="$HOME/.kube/k3s-local.yaml"
MODEL_DIR=/data/qwen-model
HF_SNAP="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct-AWQ/snapshots"

log(){ printf '[setup] %s\n' "$*"; }

# --- 1. k3s ---
if ! command -v k3s >/dev/null 2>&1; then
  log "installing k3s (single node)"
  curl -sfL https://get.k3s.io | sh -
else
  log "k3s already installed"
fi
sudo systemctl start k3s
mkdir -p "$HOME/.kube"
sudo cp /etc/rancher/k3s/k3s.yaml "$KUBECONFIG.tmp"
sudo chown "$USER" "$KUBECONFIG.tmp" && mv "$KUBECONFIG.tmp" "$KUBECONFIG"
chmod 600 "$KUBECONFIG"
for i in $(seq 1 36); do
  kubectl get nodes 2>/dev/null | grep -q " Ready" && break
  sleep 5
done
kubectl get nodes | grep -q " Ready" || { log "ERROR: node never became Ready"; exit 1; }

# --- 2. nvidia runtime (k3s auto-detects the host toolkit; verify) ---
if ! sudo grep -q nvidia /var/lib/rancher/k3s/agent/etc/containerd/config.toml 2>/dev/null; then
  log "WARNING: nvidia runtime not in k3s containerd config; restarting k3s to re-detect"
  sudo systemctl restart k3s; sleep 10
fi
kubectl get runtimeclass nvidia >/dev/null 2>&1 || kubectl apply -f - <<'EOF'
apiVersion: node.k8s.io/v1
kind: RuntimeClass
metadata: { name: nvidia }
handler: nvidia
EOF

# --- 3. storage class alias + device plugin ---
kubectl apply -f "$H100_DEPLOY/sc-standard.yaml"
kubectl apply -f "$H100_DEPLOY/nvidia-device-plugin.yaml"
log "waiting for device plugin to advertise the GPU"
for i in $(seq 1 30); do
  alloc=$(kubectl get node -o jsonpath='{.items[0].status.allocatable.nvidia\.com/gpu}' 2>/dev/null || true)
  [ "${alloc:-0}" = "1" ] && { log "GPU advertised (nvidia.com/gpu=1)"; break; }
  sleep 5
done
[ "${alloc:-0}" = "1" ] || { log "ERROR: GPU not advertised"; exit 1; }

# --- 4. stage model weights on the hostPath PV ---
if [ ! -f "$MODEL_DIR/config.json" ]; then
  snap=$(ls -d "$HF_SNAP"/*/ | head -1)
  [ -n "$snap" ] || { log "ERROR: no HF snapshot for Qwen2.5-7B-Instruct-AWQ"; exit 1; }
  log "staging model from $snap -> $MODEL_DIR"
  sudo mkdir -p "$MODEL_DIR"
  sudo rsync -aL "$snap"/ "$MODEL_DIR"/
  sudo chmod -R a+rX "$MODEL_DIR"
else
  log "model already staged at $MODEL_DIR"
fi
ls -la "$MODEL_DIR" | head -5

# --- 5. FastAPI image: build on host docker, import into k3s containerd ---
if ! sudo k3s ctr images ls -q | grep -q 'llm-service-kernel:fastapi-selfcontained'; then
  log "preparing runtime assets + building FastAPI image (host docker)"
  ASSET_MODE=auto bash "$REPO/scripts/prepare_fastapi_runtime_assets.sh"
  docker build -t llm-service-kernel:fastapi-selfcontained -f "$REPO/Dockerfile.service" "$REPO"
  log "importing image into k3s containerd"
  docker save llm-service-kernel:fastapi-selfcontained | sudo k3s ctr images import -
else
  log "FastAPI image already imported"
fi

log "PHASE 1 DONE. Next: local_service/scripts/deploy_stack.sh"
