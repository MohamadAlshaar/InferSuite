#!/usr/bin/env bash
set -euo pipefail

KERNEL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${KERNEL_ROOT}/.." && pwd)"
HF_VENV_DIR="${HF_VENV_DIR:-${KERNEL_ROOT}/.venv-download-models}"

log() { printf '\033[1;34m[download_models]\033[0m %s\n' "$*"; }

if [ ! -d "${HF_VENV_DIR}" ]; then
  log "Creating venv at ${HF_VENV_DIR}"
  python3 -m venv "${HF_VENV_DIR}"
fi

# shellcheck disable=SC1090
source "${HF_VENV_DIR}/bin/activate"

log "Installing huggingface_hub..."
python -m pip install --upgrade pip
python -m pip install huggingface_hub

cd "${REPO_ROOT}"

if [ ! -f "bge-base-en-v1.5/config.json" ]; then
  log "Downloading bge-base-en-v1.5..."
  python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="BAAI/bge-base-en-v1.5",
    local_dir="bge-base-en-v1.5",
    local_dir_use_symlinks=False,
)
PY
else
  log "bge-base-en-v1.5 already present, skipping"
fi

if [ ! -f "Qwen2.5-0.5B-Instruct/config.json" ]; then
  log "Downloading Qwen2.5-0.5B-Instruct..."
  python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen2.5-0.5B-Instruct",
    local_dir="Qwen2.5-0.5B-Instruct",
    local_dir_use_symlinks=False,
)
PY
else
  log "Qwen2.5-0.5B-Instruct already present, skipping"
fi
