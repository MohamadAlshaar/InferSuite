#!/usr/bin/env bash
set -euo pipefail

KERNEL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${KERNEL_ROOT}/.." && pwd)"
OUT_DIR="${KERNEL_ROOT}/fastapi_runtime_assets"

ASSET_MODE="${ASSET_MODE:-auto}"   # auto | local | online
PYTHON_BIN="${PYTHON_BIN:-python3}"

# MiniLM is optional — BGE is used for both RAG and semantic cache by default.
# Set BUNDLE_MINILM=1 and provide MINILM_LOCAL_SRC to include it.
BUNDLE_MINILM="${BUNDLE_MINILM:-0}"
MINILM_LOCAL_SRC="${MINILM_LOCAL_SRC:-${REPO_ROOT}/all-MiniLM-L6-v2}"
BGE_LOCAL_SRC="${BGE_LOCAL_SRC:-${REPO_ROOT}/bge-base-en-v1.5}"
QWEN_LOCAL_SRC="${QWEN_LOCAL_SRC:-${REPO_ROOT}/Qwen2.5-0.5B-Instruct}"
RAG_SEED_LOCAL_SRC="${RAG_SEED_LOCAL_SRC:-${REPO_ROOT}/rag_store_tenants}"

MINILM_HF_REPO="${MINILM_HF_REPO:-sentence-transformers/all-MiniLM-L6-v2}"  # only used when BUNDLE_MINILM=1
BGE_HF_REPO="${BGE_HF_REPO:-BAAI/bge-base-en-v1.5}"
QWEN_HF_REPO="${QWEN_HF_REPO:-Qwen/Qwen2.5-0.5B-Instruct}"

TMP_DIR="${KERNEL_ROOT}/.tmp_runtime_assets"

log() {
  printf '[prepare_fastapi_runtime_assets] %s\n' "$*"
}

die() {
  printf '[prepare_fastapi_runtime_assets] ERROR: %s\n' "$*" >&2
  exit 1
}

have_python() {
  command -v "${PYTHON_BIN}" >/dev/null 2>&1
}

python_ok() {
  "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
print("ok")
PY
}

ensure_python() {
  have_python || die "python not found: ${PYTHON_BIN}"
  python_ok || die "python is not usable: ${PYTHON_BIN}"
}

ensure_hf_hub() {
  ensure_python
  if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import huggingface_hub
print(huggingface_hub.__version__)
PY
  then
    log "huggingface_hub not found, installing into user site-packages..."
    "${PYTHON_BIN}" -m pip install --user huggingface_hub >/dev/null
  fi

  "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import huggingface_hub
print(huggingface_hub.__version__)
PY
}

prepare_dirs() {
  rm -rf "${OUT_DIR}" "${TMP_DIR}"
  mkdir -p \
    "${OUT_DIR}/models/bge-base-en-v1.5" \
    "${OUT_DIR}/models/Qwen2.5-0.5B-Instruct" \
    "${OUT_DIR}/rag_store_tenants" \
    "${TMP_DIR}"
  if [ "${BUNDLE_MINILM}" = "1" ]; then
    mkdir -p "${OUT_DIR}/models/all-MiniLM-L6-v2"
  fi
}

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [ -e "$src" ]; then
    cp -r "$src" "$dst"
  fi
}

copy_optional_file() {
  local src="$1"
  local dst="$2"
  if [ -f "$src" ]; then
    cp "$src" "$dst"
  fi
}

copy_model_minilm_from_dir() {
  local src="$1"
  local dst="${OUT_DIR}/models/all-MiniLM-L6-v2"

  [ -d "${src}" ] || die "MiniLM source dir not found: ${src}"

  copy_if_exists "${src}/1_Pooling" "${dst}/"
  for f in \
    config.json \
    config_sentence_transformers.json \
    data_config.json \
    modules.json \
    model.safetensors \
    sentence_bert_config.json \
    special_tokens_map.json \
    tokenizer_config.json \
    tokenizer.json \
    vocab.txt
  do
    copy_optional_file "${src}/${f}" "${dst}/"
  done
}

copy_model_bge_from_dir() {
  local src="$1"
  local dst="${OUT_DIR}/models/bge-base-en-v1.5"

  [ -d "${src}" ] || die "BGE source dir not found: ${src}"

  copy_if_exists "${src}/1_Pooling" "${dst}/"
  for f in \
    config.json \
    config_sentence_transformers.json \
    modules.json \
    model.safetensors \
    sentence_bert_config.json \
    special_tokens_map.json \
    tokenizer_config.json \
    tokenizer.json \
    vocab.txt
  do
    copy_optional_file "${src}/${f}" "${dst}/"
  done
}

copy_qwen_tokenizer_from_dir() {
  local src="$1"
  local dst="${OUT_DIR}/models/Qwen2.5-0.5B-Instruct"

  [ -d "${src}" ] || die "Qwen tokenizer source dir not found: ${src}"

  for f in \
    tokenizer.json \
    tokenizer_config.json \
    merges.txt \
    vocab.json \
    vocab.txt \
    special_tokens_map.json \
    added_tokens.json
  do
    copy_optional_file "${src}/${f}" "${dst}/"
  done
}

copy_rag_seed_if_present() {
  local src="$1"
  local dst="${OUT_DIR}/rag_store_tenants"

  if [ -d "${src}" ]; then
    cp -r "${src}/." "${dst}/"
    log "Copied local rag_store_tenants seed from ${src}"
  else
    log "No local rag_store_tenants seed found at ${src}; leaving seed dir empty"
  fi
}

download_hf_snapshot() {
  local repo_id="$1"
  local local_dir="$2"
  shift 2
  ensure_hf_hub

  log "Downloading ${repo_id} -> ${local_dir}"
  "${PYTHON_BIN}" - "$repo_id" "$local_dir" "$@" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
local_dir = sys.argv[2]
patterns = sys.argv[3:]

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    allow_patterns=patterns if patterns else None,
)
print(local_dir)
PY
}

use_local_assets() {
  log "Using local asset sources"
  if [ "${BUNDLE_MINILM}" = "1" ]; then
    copy_model_minilm_from_dir "${MINILM_LOCAL_SRC}"
  fi
  copy_model_bge_from_dir "${BGE_LOCAL_SRC}"
  copy_qwen_tokenizer_from_dir "${QWEN_LOCAL_SRC}"
  copy_rag_seed_if_present "${RAG_SEED_LOCAL_SRC}"
}

use_online_assets() {
  log "Using online asset download mode"

  local bge_tmp="${TMP_DIR}/bge-base-en-v1.5"
  local qwen_tmp="${TMP_DIR}/Qwen2.5-0.5B-Instruct"

  if [ "${BUNDLE_MINILM}" = "1" ]; then
    local minilm_tmp="${TMP_DIR}/all-MiniLM-L6-v2"
    download_hf_snapshot "${MINILM_HF_REPO}" "${minilm_tmp}" \
      "1_Pooling/*" \
      "config.json" \
      "config_sentence_transformers.json" \
      "data_config.json" \
      "modules.json" \
      "model.safetensors" \
      "sentence_bert_config.json" \
      "special_tokens_map.json" \
      "tokenizer_config.json" \
      "tokenizer.json" \
      "vocab.txt"
  fi

  download_hf_snapshot "${BGE_HF_REPO}" "${bge_tmp}" \
    "1_Pooling/*" \
    "config.json" \
    "config_sentence_transformers.json" \
    "modules.json" \
    "model.safetensors" \
    "sentence_bert_config.json" \
    "special_tokens_map.json" \
    "tokenizer_config.json" \
    "tokenizer.json" \
    "vocab.txt"

  download_hf_snapshot "${QWEN_HF_REPO}" "${qwen_tmp}" \
    "tokenizer.json" \
    "tokenizer_config.json" \
    "merges.txt" \
    "vocab.json" \
    "vocab.txt" \
    "special_tokens_map.json" \
    "added_tokens.json"

  if [ "${BUNDLE_MINILM}" = "1" ]; then
    copy_model_minilm_from_dir "${minilm_tmp}"
  fi
  copy_model_bge_from_dir "${bge_tmp}"
  copy_qwen_tokenizer_from_dir "${qwen_tmp}"
  copy_rag_seed_if_present "${RAG_SEED_LOCAL_SRC}"
}

validate_outputs() {
  if [ "${BUNDLE_MINILM}" = "1" ]; then
    test -f "${OUT_DIR}/models/all-MiniLM-L6-v2/config.json" \
      || die "Missing MiniLM config.json in output"
  fi
  test -f "${OUT_DIR}/models/bge-base-en-v1.5/config.json" \
    || die "Missing BGE config.json in output"
  test -f "${OUT_DIR}/models/Qwen2.5-0.5B-Instruct/tokenizer.json" \
    || die "Missing Qwen tokenizer.json in output"
}

local_sources_available() {
  [ -d "${BGE_LOCAL_SRC}" ] && [ -d "${QWEN_LOCAL_SRC}" ] && \
    { [ "${BUNDLE_MINILM}" != "1" ] || [ -d "${MINILM_LOCAL_SRC}" ]; }
}

main() {
  prepare_dirs

  case "${ASSET_MODE}" in
    local)
      use_local_assets
      ;;
    online)
      use_online_assets
      ;;
    auto)
      if local_sources_available; then
        use_local_assets
      else
        use_online_assets
      fi
      ;;
    *)
      die "Unsupported ASSET_MODE: ${ASSET_MODE}"
      ;;
  esac

  validate_outputs
  log "Runtime assets ready under ${OUT_DIR}"
  find "${OUT_DIR}" -maxdepth 3 | sort
}

main "$@"
