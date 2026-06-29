#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# create_eks_cluster.sh — create the EKS Auto Mode cluster + ECR setup
#
# Usage:
#   ./scripts/create_eks_cluster.sh
#
# Prerequisites:
#   - aws CLI configured with credentials (aws sts get-caller-identity)
#   - eksctl >= 0.191 (brew install eksctl  /  apt install eksctl)
#   - kubectl
#   - docker
#
# What this does:
#   1. Create EKS Auto Mode cluster in eu-west-2
#   2. Update kubeconfig
#   3. Create ECR repository in eu-west-2
#   4. Build FastAPI Docker image and push to ECR
#   5. Create managed node group for general-purpose compute (bare-metal capable)
#   6. Create ebs-gp3 StorageClass (if not already present)
# ---------------------------------------------------------------------------
set -euo pipefail

KERNEL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Load config.env ───────────────────────────────────────────────────────
CONFIG_ENV="${KERNEL_ROOT}/deploy/config.env"
if [ -f "${CONFIG_ENV}" ]; then
    set -a; source "${CONFIG_ENV}"; set +a
fi

# ── Config ────────────────────────────────────────────────────────────────
AWS_REGION="${AWS_REGION:-eu-west-2}"
CLUSTER_NAME="${CLUSTER_NAME:-e2e-cluster-eks}"
K8S_VERSION="${K8S_VERSION:-1.32}"
INSTANCE_TYPE="${COMPUTE_INSTANCE_TYPE:-${INSTANCE_TYPE:-c7i.metal-24xl}}"

AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-524558748675}"
ECR_REPO="llm-service/fastapi"
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_TAG="latest"
FULL_IMAGE="${ECR_REGISTRY}/${ECR_REPO}:${IMAGE_TAG}"

FASTAPI_IMAGE_LOCAL="llm-service-kernel:fastapi-selfcontained"

# ── Helpers ───────────────────────────────────────────────────────────────
log()  { printf '\n\033[1;34m[create-cluster]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m  ✓\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m  ⚠\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[create-cluster] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1 — install it first"
}

# ── Step 1: Preflight ─────────────────────────────────────────────────────
preflight() {
  log "Step 1/6: Preflight checks"
  require_cmd aws
  require_cmd eksctl
  require_cmd kubectl
  require_cmd docker

  local caller_identity
  caller_identity="$(aws sts get-caller-identity --query 'Account' --output text 2>/dev/null || echo "")"
  if [ -z "${caller_identity}" ]; then
    die "AWS credentials not configured — run: aws configure  OR  set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY"
  fi
  ok "AWS authenticated as account: ${caller_identity}"

  if [ "${caller_identity}" != "${AWS_ACCOUNT_ID}" ]; then
    warn "Active account (${caller_identity}) differs from expected (${AWS_ACCOUNT_ID})"
    warn "Set AWS_ACCOUNT_ID env var if this is intentional"
    AWS_ACCOUNT_ID="${caller_identity}"
    ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
    FULL_IMAGE="${ECR_REGISTRY}/${ECR_REPO}:${IMAGE_TAG}"
  fi

  ok "Region: ${AWS_REGION}"
  ok "Cluster: ${CLUSTER_NAME}"
  ok "Instance: ${INSTANCE_TYPE}"
  ok "Image: ${FULL_IMAGE}"
}

# ── Step 2: Verify cluster exists + ensure OIDC provider ─────────────────
check_cluster() {
  log "Step 2/6: Verifying cluster exists"
  local status
  status=$(aws eks describe-cluster --name "${CLUSTER_NAME}" --region "${AWS_REGION}" \
    --query 'cluster.status' --output text 2>/dev/null || echo "NOT_FOUND")
  if [ "${status}" = "ACTIVE" ]; then
    ok "Cluster '${CLUSTER_NAME}' is ACTIVE"
  else
    die "Cluster '${CLUSTER_NAME}' not found or not ACTIVE (status=${status}) — create it via the AWS console first, then re-run this script"
  fi

  # Ensure the IAM OIDC provider exists for this cluster — required for IRSA.
  # Console-created clusters don't create this automatically.
  local oidc_url oidc_id
  oidc_url=$(aws eks describe-cluster --name "${CLUSTER_NAME}" --region "${AWS_REGION}" \
    --query 'cluster.identity.oidc.issuer' --output text 2>/dev/null || echo "")
  oidc_id="${oidc_url##*/}"

  if [ -z "${oidc_id}" ]; then
    warn "Could not retrieve OIDC issuer — skipping OIDC provider setup"
  elif aws iam list-open-id-connect-providers \
        --query "OIDCProviderList[?ends_with(Arn, '${oidc_id}')]" \
        --output text 2>/dev/null | grep -q "${oidc_id}"; then
    ok "OIDC provider already exists"
  else
    log "Creating IAM OIDC provider for cluster (required for IRSA)"
    local thumbprint
    thumbprint=$(echo | openssl s_client \
      -connect "oidc.eks.${AWS_REGION}.amazonaws.com:443" \
      -servername "oidc.eks.${AWS_REGION}.amazonaws.com" 2>/dev/null \
      | openssl x509 -fingerprint -sha1 -noout 2>/dev/null \
      | sed 's/.*=//;s/://g' | tr '[:upper:]' '[:lower:]')
    aws iam create-open-id-connect-provider \
      --url "${oidc_url}" \
      --thumbprint-list "${thumbprint}" \
      --client-id-list "sts.amazonaws.com" \
      --region "${AWS_REGION}" >/dev/null
    ok "OIDC provider created: ${oidc_url}"
  fi

  # Update IRSA role trust policies to reference this cluster's OIDC provider.
  # Roles created for the old cluster still point at the old OIDC ID.
  local ebs_role="${CLUSTER_NAME}-EBSCSIDriverRole"
  if aws iam get-role --role-name "${ebs_role}" >/dev/null 2>&1; then
    local old_oidc_in_trust
    old_oidc_in_trust=$(aws iam get-role --role-name "${ebs_role}" \
      --query 'Role.AssumeRolePolicyDocument.Statement[0].Principal.Federated' \
      --output text 2>/dev/null | grep -oP '[A-F0-9]{32}' || echo "")
    if [ -n "${old_oidc_in_trust}" ] && [ "${old_oidc_in_trust}" != "${oidc_id}" ]; then
      log "Updating ${ebs_role} trust policy: ${old_oidc_in_trust} → ${oidc_id}"
      local trust_doc
      trust_doc=$(aws iam get-role --role-name "${ebs_role}" \
        --query 'Role.AssumeRolePolicyDocument' --output json | \
        sed "s/${old_oidc_in_trust}/${oidc_id}/g")
      aws iam update-assume-role-policy \
        --role-name "${ebs_role}" \
        --policy-document "${trust_doc}" >/dev/null
      ok "Trust policy updated for ${ebs_role}"
    else
      ok "Trust policy for ${ebs_role} already correct"
    fi
  fi
}

# ── Step 3: Update kubeconfig ─────────────────────────────────────────────
update_kubeconfig() {
  log "Step 3/6: Updating kubeconfig"
  aws eks update-kubeconfig \
    --region "${AWS_REGION}" \
    --name "${CLUSTER_NAME}"
  ok "kubeconfig updated"

  kubectl cluster-info
}

# ── Step 4: ECR repo + image push ─────────────────────────────────────────
setup_ecr_and_push() {
  log "Step 4/6: ECR repository + image"

  # Auth to ECR
  log "Authenticating Docker to ECR"
  aws ecr get-login-password --region "${AWS_REGION}" | \
    docker login --username AWS --password-stdin "${ECR_REGISTRY}"
  ok "ECR auth done"

  # Create repo if it doesn't exist
  if ! aws ecr describe-repositories \
        --repository-names "${ECR_REPO}" \
        --region "${AWS_REGION}" >/dev/null 2>&1; then
    log "Creating ECR repository: ${ECR_REPO}"
    aws ecr create-repository \
      --repository-name "${ECR_REPO}" \
      --region "${AWS_REGION}" \
      --image-scanning-configuration scanOnPush=true \
      --encryption-configuration encryptionType=AES256
    ok "ECR repository created"
  else
    ok "ECR repository already exists"
  fi

  # Try to pull from source ECR (us-east-1) and retag — faster than rebuild
  local src_image="${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}"
  local pulled=0

  # Check if we already have the local image
  if docker image inspect "${FASTAPI_IMAGE_LOCAL}" >/dev/null 2>&1; then
    log "Found local image '${FASTAPI_IMAGE_LOCAL}' — tagging for eu-west-2 ECR"
    docker tag "${FASTAPI_IMAGE_LOCAL}" "${FULL_IMAGE}"
    pulled=1
  else
    # Try pulling from source ECR
    log "Attempting to pull from source ECR (us-east-1)..."
    # Need to auth to us-east-1 too
    aws ecr get-login-password --region us-east-1 | \
      docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com" 2>/dev/null || true

    if docker pull "${src_image}" 2>/dev/null; then
      docker tag "${src_image}" "${FULL_IMAGE}"
      pulled=1
      ok "Pulled and retagged from us-east-1"
    fi
  fi

  if [ "${pulled}" = "0" ]; then
    log "Building FastAPI image locally (no cached image found)"
    docker build -t "${FASTAPI_IMAGE_LOCAL}" -f "${KERNEL_ROOT}/Dockerfile.service" "${KERNEL_ROOT}"
    docker tag "${FASTAPI_IMAGE_LOCAL}" "${FULL_IMAGE}"
    ok "Image built locally"
  fi

  log "Pushing image to ECR eu-west-2..."
  docker push "${FULL_IMAGE}"
  ok "Image pushed: ${FULL_IMAGE}"
}

# ── Step 4b: Core addons (vpc-cni + kube-proxy + aws-ebs-csi-driver) ─────
# EKS Auto Mode manages networking for its own Karpenter nodes but managed
# node groups need vpc-cni and kube-proxy installed as EKS addons, otherwise
# nodes join but never reach Ready (CNI not configured, pods can't schedule).
# aws-ebs-csi-driver is needed for static PV attachment of pre-existing EBS volumes;
# the EKS Auto Mode driver (ebs.csi.eks.amazonaws.com) cannot attach them.
EBS_CSI_ROLE_ARN="${EBS_CSI_ROLE_ARN:-arn:aws:iam::${AWS_ACCOUNT_ID}:role/${CLUSTER_NAME}-EBSCSIDriverRole}"

ensure_core_addons() {
  log "Step 4b: Core addons (vpc-cni, kube-proxy, aws-ebs-csi-driver)"

  for addon in vpc-cni kube-proxy coredns; do
    status=$(aws eks describe-addon \
      --cluster-name "${CLUSTER_NAME}" --addon-name "${addon}" \
      --region "${AWS_REGION}" \
      --query 'addon.status' --output text 2>/dev/null || echo "MISSING")

    if [ "${status}" = "ACTIVE" ]; then
      ok "${addon} already active"
    else
      if [ "${status}" = "MISSING" ]; then
        log "Installing ${addon}..."
        aws eks create-addon \
          --cluster-name "${CLUSTER_NAME}" \
          --addon-name "${addon}" \
          --region "${AWS_REGION}" >/dev/null
      else
        log "${addon} status=${status} — waiting..."
      fi
      log "Waiting for ${addon} to be active..."
      aws eks wait addon-active \
        --cluster-name "${CLUSTER_NAME}" \
        --addon-name "${addon}" \
        --region "${AWS_REGION}"
      ok "${addon} active"
    fi
  done

  # aws-ebs-csi-driver: install with IRSA role if the role exists
  local ebs_status
  ebs_status=$(aws eks describe-addon \
    --cluster-name "${CLUSTER_NAME}" --addon-name aws-ebs-csi-driver \
    --region "${AWS_REGION}" \
    --query 'addon.status' --output text 2>/dev/null || echo "MISSING")

  if [ "${ebs_status}" = "ACTIVE" ]; then
    ok "aws-ebs-csi-driver already active"
  else
    local irsa_flag=""
    if aws iam get-role --role-name "${CLUSTER_NAME}-EBSCSIDriverRole" >/dev/null 2>&1; then
      irsa_flag="--service-account-role-arn ${EBS_CSI_ROLE_ARN}"
      ok "Using IRSA role: ${EBS_CSI_ROLE_ARN}"
    else
      warn "EBSCSIDriverRole not found — installing addon without IRSA (node role must have AmazonEBSCSIDriverPolicy)"
    fi
    if [ "${ebs_status}" = "MISSING" ]; then
      log "Installing aws-ebs-csi-driver..."
      # shellcheck disable=SC2086
      aws eks create-addon \
        --cluster-name "${CLUSTER_NAME}" \
        --addon-name aws-ebs-csi-driver \
        --region "${AWS_REGION}" \
        ${irsa_flag} >/dev/null
    else
      log "aws-ebs-csi-driver status=${ebs_status} — waiting..."
    fi
    log "Waiting for aws-ebs-csi-driver to be active..."
    aws eks wait addon-active \
      --cluster-name "${CLUSTER_NAME}" \
      --addon-name aws-ebs-csi-driver \
      --region "${AWS_REGION}"
    ok "aws-ebs-csi-driver active"
  fi
}

# ── Step 5: Ensure general-purpose managed node group exists ─────────────
ensure_general_purpose_ng() {
  log "Step 5/6: General-purpose node group (${INSTANCE_TYPE})"

  if aws eks describe-nodegroup \
       --cluster-name "${CLUSTER_NAME}" \
       --nodegroup-name general-purpose \
       --region "${AWS_REGION}" >/dev/null 2>&1; then
    ok "Managed node group 'general-purpose' already exists"
    return 0
  fi

  # Leave the EKS Auto Mode 'general-purpose' nodepool in place — it handles
  # GPU instance provisioning (g6.12xlarge) for the llm-d workload.
  # The managed node group below coexists with it: bare-metal pods select
  # node.kubernetes.io/instance-type=c7i.metal-24xl, GPU pods select g6.12xlarge.

  # Resolve node IAM role: look for a role created alongside this cluster,
  # fall back to creating one via eksctl if not found.
  local node_role_arn
  node_role_arn=$(aws iam list-roles \
    --query "Roles[?contains(RoleName, '${CLUSTER_NAME}') && contains(RoleName, 'Node')].Arn" \
    --output text 2>/dev/null | awk '{print $1}')

  # Pin to a single AZ subnet so EBS PVs (created in that AZ) can always attach.
  # Using all subnets causes random AZ placement and EBS PV node affinity failures on resume.
  # Pick the first subnet — all PVs will be created in its AZ automatically (WaitForFirstConsumer).
  local all_subnets subnet_az subnets
  all_subnets=$(aws eks describe-cluster --name "${CLUSTER_NAME}" --region "${AWS_REGION}" \
    --query 'cluster.resourcesVpcConfig.subnetIds[0]' --output text 2>/dev/null)
  subnets="${all_subnets}"

  if [ -z "${node_role_arn}" ] || [ -z "${subnets}" ]; then
    die "Could not resolve node IAM role or subnets — check cluster config"
  fi

  subnet_az=$(aws ec2 describe-subnets --region "${AWS_REGION}" \
    --subnet-ids "${subnets}" --query 'Subnets[0].AvailabilityZone' --output text 2>/dev/null)
  log "  Pinning node group to single AZ: ${subnet_az} (subnet ${subnets})"
  log "  All EBS PVs will be created in ${subnet_az} — no AZ mismatch on resume"

  log "Creating managed node group: ${INSTANCE_TYPE}"
  log "  Role: ${node_role_arn}"
  aws eks create-nodegroup \
    --cluster-name "${CLUSTER_NAME}" \
    --nodegroup-name general-purpose \
    --node-role "${node_role_arn}" \
    --subnets ${subnets} \
    --instance-types "${INSTANCE_TYPE}" \
    --scaling-config minSize=0,maxSize=10,desiredSize=1 \
    --ami-type AL2023_x86_64_STANDARD \
    --disk-size 100 \
    --region "${AWS_REGION}" >/dev/null

  log "Waiting for node group to be Active..."
  aws eks wait nodegroup-active \
    --cluster-name "${CLUSTER_NAME}" \
    --nodegroup-name general-purpose \
    --region "${AWS_REGION}"
  ok "Managed node group ready → ${INSTANCE_TYPE}"
}

# ── Step 5b: GPU managed node group ───────────────────────────────────────
# GPU nodes MUST be managed (not EKS Auto Mode) because:
#   - ebs-csi-node daemonset (ebs.csi.aws.com) only runs on managed nodes
#   - EKS Auto Mode CSI (ebs.csi.eks.amazonaws.com) can only attach volumes it created
#   - We restore model volume from a pre-existing EBS volume via static PV
GPU_INSTANCE_TYPE="${GPU_INSTANCE_TYPE:-g6.12xlarge}"
ensure_gpu_ng() {
  log "Step 5b: GPU managed node group (${GPU_INSTANCE_TYPE})"

  if aws eks describe-nodegroup \
       --cluster-name "${CLUSTER_NAME}" \
       --nodegroup-name gpu \
       --region "${AWS_REGION}" >/dev/null 2>&1; then
    ok "Managed node group 'gpu' already exists"
    # Ensure NVIDIA device plugin is present even on re-runs
    kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.0/deployments/static/nvidia-device-plugin.yml >/dev/null 2>&1 || true
    return 0
  fi

  # Remove any EKS Auto Mode GPU nodepool — it cannot use pre-existing EBS volumes
  if kubectl get nodepool gpu >/dev/null 2>&1; then
    log "Removing EKS Auto Mode GPU nodepool (incompatible with static EBS restore)"
    kubectl delete nodepool gpu
    ok "GPU Auto Mode nodepool removed"
  fi

  local node_role_arn subnets
  node_role_arn=$(aws iam list-roles \
    --query "Roles[?contains(RoleName, '${CLUSTER_NAME}') && contains(RoleName, 'Node')].Arn" \
    --output text 2>/dev/null | awk '{print $1}')
  subnets=$(aws eks describe-cluster --name "${CLUSTER_NAME}" --region "${AWS_REGION}" \
    --query 'cluster.resourcesVpcConfig.subnetIds[0]' --output text 2>/dev/null)

  if [ -z "${node_role_arn}" ] || [ -z "${subnets}" ]; then
    die "Could not resolve node IAM role or subnets for GPU node group"
  fi

  log "Creating GPU managed node group: ${GPU_INSTANCE_TYPE}"
  aws eks create-nodegroup \
    --cluster-name "${CLUSTER_NAME}" \
    --nodegroup-name gpu \
    --node-role "${node_role_arn}" \
    --subnets ${subnets} \
    --instance-types "${GPU_INSTANCE_TYPE}" \
    --scaling-config minSize=0,maxSize=2,desiredSize=1 \
    --ami-type AL2023_x86_64_NVIDIA \
    --disk-size 100 \
    --region "${AWS_REGION}" >/dev/null

  log "Waiting for GPU node group to be Active..."
  aws eks wait nodegroup-active \
    --cluster-name "${CLUSTER_NAME}" \
    --nodegroup-name gpu \
    --region "${AWS_REGION}"
  ok "GPU managed node group ready → ${GPU_INSTANCE_TYPE}"

  # Deploy NVIDIA device plugin so GPUs are advertised to the scheduler
  log "Deploying NVIDIA device plugin"
  kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.0/deployments/static/nvidia-device-plugin.yml >/dev/null
  ok "NVIDIA device plugin deployed"
}

# ── Step 6: StorageClass ──────────────────────────────────────────────────
ensure_storageclass() {
  log "Step 6/6: Checking ebs-gp3 StorageClass"

  # Verify it uses the right provisioner (ebs.csi.aws.com, not ebs.csi.eks.amazonaws.com)
  if kubectl get storageclass ebs-gp3 >/dev/null 2>&1; then
    local provisioner
    provisioner=$(kubectl get storageclass ebs-gp3 \
      -o jsonpath='{.provisioner}' 2>/dev/null || echo "")
    if [ "${provisioner}" = "ebs.csi.aws.com" ]; then
      ok "ebs-gp3 StorageClass already exists with correct provisioner"
      return 0
    else
      warn "ebs-gp3 uses wrong provisioner '${provisioner}' — deleting and recreating"
      kubectl delete storageclass ebs-gp3
    fi
  fi

  log "Creating ebs-gp3 StorageClass (provisioner: ebs.csi.aws.com)"
  kubectl apply -f - <<'EOF'
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: ebs-gp3
  annotations:
    storageclass.kubernetes.io/is-default-class: "true"
provisioner: ebs.csi.aws.com
volumeBindingMode: WaitForFirstConsumer
reclaimPolicy: Delete
parameters:
  type: gp3
  encrypted: "true"
EOF
  ok "ebs-gp3 StorageClass created"
}

# ── Summary ───────────────────────────────────────────────────────────────
print_next_steps() {
  echo
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  Cluster ready! Next steps:"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo
  echo "  1. Deploy the full stack (storage + FastAPI + llm-d):"
  echo "     kubectl create namespace llm-service 2>/dev/null || true"
  echo "     kubectl apply -f deploy/k8s-storage/"
  echo "     kubectl apply -f deploy/k8s-fastapi/"
  echo
  echo "  2. Or run the automated deploy script:"
  echo "     SKIP_INGEST=1 bash scripts/deploy_fullstack_single_node.sh"
  echo
  echo "  3. After services are up — run ingestion:"
  echo "     bash loadgen/benchmark/run_benchmark.sh ingestion"
  echo
  echo "  4. Run benchmarks:"
  echo "     bash loadgen/benchmark/run_benchmark.sh all"
  echo
  echo "  Cluster: ${CLUSTER_NAME} | Region: ${AWS_REGION} | Node: ${INSTANCE_TYPE}"
  echo "  Image:   ${FULL_IMAGE}"
  echo
  echo "  Useful commands:"
  echo "    kubectl get nodes -o wide"
  echo "    kubectl get nodepool general-purpose -o yaml"
  echo "    kubectl get pods -n llm-service"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

main() {
  echo
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  EKS Cluster Setup — ${CLUSTER_NAME} / ${AWS_REGION} / ${INSTANCE_TYPE}"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  preflight
  check_cluster
  update_kubeconfig
  setup_ecr_and_push
  ensure_core_addons
  ensure_general_purpose_ng
  ensure_gpu_ng
  ensure_storageclass
  print_next_steps
}

main "$@"
