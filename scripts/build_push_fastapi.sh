#!/usr/bin/env bash
# Build the FastAPI Docker image and push it to ECR.
# Called automatically by setup.sh in EKS mode, or run standalone:
#
#   ./scripts/build_push_fastapi.sh
#
# Reads AWS_ACCOUNT_ID and AWS_REGION from deploy/config.env.
# Skips the push if the image already exists in ECR (use FORCE_BUILD=1 to override).
set -euo pipefail

KERNEL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_ENV="${KERNEL_ROOT}/deploy/config.env"

[ -f "${CONFIG_ENV}" ] || { echo "ERROR: deploy/config.env not found — copy from deploy/config.env.example"; exit 1; }
set -a; source "${CONFIG_ENV}"; set +a

FORCE_BUILD="${FORCE_BUILD:-0}"
ECR_REPO="llm-service/fastapi"
IMAGE_TAG="${IMAGE_TAG:-latest}"
ECR_URL="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}"

log()  { printf '\n\033[1;34m[build_push_fastapi]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m  ✓\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[build_push_fastapi] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

command -v docker >/dev/null 2>&1 || die "docker not found"
command -v aws    >/dev/null 2>&1 || die "aws CLI not found"

# ── Check if already pushed ───────────────────────────────────────────────────
if [ "${FORCE_BUILD}" != "1" ]; then
  if aws ecr describe-images \
      --repository-name "${ECR_REPO}" \
      --region "${AWS_REGION}" \
      --image-ids imageTag="${IMAGE_TAG}" \
      >/dev/null 2>&1; then
    ok "Image already exists in ECR: ${ECR_URL}"
    ok "Skipping build (use FORCE_BUILD=1 to rebuild)"
    exit 0
  fi
fi

# ── Check fastapi_runtime_assets has models ───────────────────────────────────
BGE_DIR="${KERNEL_ROOT}/fastapi_runtime_assets/models/bge-base-en-v1.5"
if [ ! -d "${BGE_DIR}" ]; then
  log "fastapi_runtime_assets/models/bge-base-en-v1.5 not found — running prepare script"
  bash "${KERNEL_ROOT}/scripts/prepare_fastapi_runtime_assets.sh"
fi
ok "fastapi_runtime_assets ready"

# ── Create ECR repo if needed ─────────────────────────────────────────────────
if ! aws ecr describe-repositories \
    --repository-names "${ECR_REPO}" \
    --region "${AWS_REGION}" >/dev/null 2>&1; then
  log "Creating ECR repository: ${ECR_REPO}"
  aws ecr create-repository \
    --repository-name "${ECR_REPO}" \
    --region "${AWS_REGION}" >/dev/null
  ok "ECR repository created"
else
  ok "ECR repository exists"
fi

# ── ECR login ─────────────────────────────────────────────────────────────────
log "Authenticating with ECR"
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin \
    "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
ok "ECR login successful"

# ── Build ─────────────────────────────────────────────────────────────────────
log "Building Docker image (this takes ~3 min on first build)"
docker build \
  --platform linux/amd64 \
  -f "${KERNEL_ROOT}/Dockerfile.service" \
  -t "${ECR_URL}" \
  "${KERNEL_ROOT}"
ok "Build complete"

# ── Push ──────────────────────────────────────────────────────────────────────
log "Pushing to ECR: ${ECR_URL}"
docker push "${ECR_URL}"
ok "Pushed: ${ECR_URL}"

echo
echo "  Image: ${ECR_URL}"
