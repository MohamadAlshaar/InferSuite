from __future__ import annotations

import os
from pathlib import Path


_TRUE_VALUES = {"1", "true", "yes", "on"}
_VALID_MODEL_BACKENDS = {"direct_vllm", "llmd"}
_VALID_LLMD_API_MODES = {"chat", "completions"}
_VALID_RAG_SCORE_MODES = {"distance", "similarity"}


def _load_local_env_files() -> None:
    """
    Best-effort local env loader.

    Priority:
      1. Existing exported environment variables always win.
      2. .env if present
      3. .env.example if present

    This helps local runs outside Kubernetes.
    Inside Kubernetes, env vars from ConfigMap/Secret override these values.
    """
    root = Path(__file__).resolve().parents[2]
    candidates = [root / ".env", root / ".env.example"]

    for path in candidates:
        if not path.exists():
            continue

        try:
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()

                if not line or line.startswith("#"):
                    continue

                if "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                if not key:
                    continue

                if value and (
                    (value.startswith('"') and value.endswith('"'))
                    or (value.startswith("'") and value.endswith("'"))
                ):
                    value = value[1:-1]

                os.environ.setdefault(key, value)
        except Exception:
            pass


_load_local_env_files()


def _env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip()


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in _TRUE_VALUES


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except Exception:
        return default


def _env_first_str(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return default


class Settings:
    """
    Runtime settings for the llm-service-kernel.

    Behavior:
      - local runs can read .env / .env.example automatically
      - exported shell vars override local env files
      - Kubernetes env vars override local env files

    Compatibility note:
      New canonical generation settings are:
        - GENERATION_BACKEND
        - GENERATION_BASE_URL
        - GENERATION_API_MODE
        - GENERATION_MODEL_NAME

      Legacy variables still work as fallbacks:
        - MODEL_BACKEND
        - VLLM_BASE_URL
        - LLMD_BASE_URL
        - LLMD_API_MODE
        - SERVED_MODEL_NAME
    """

    def __init__(self) -> None:
        self.APP_HOST: str = _env_str("APP_HOST", "0.0.0.0")
        self.APP_PORT: int = _env_int("APP_PORT", 8080)

        self.GENERATION_BACKEND: str = _env_first_str(
            "GENERATION_BACKEND",
            "MODEL_BACKEND",
            default="direct_vllm",
        ).lower()

        self.GENERATION_API_MODE: str = _env_first_str(
            "GENERATION_API_MODE",
            "LLMD_API_MODE",
            default="completions",
        ).lower()

        self.GENERATION_MODEL_NAME: str = _env_first_str(
            "GENERATION_MODEL_NAME",
            "SERVED_MODEL_NAME",
            default="qwen2.5-0.5b",
        )

        generation_base_url = _env_str("GENERATION_BASE_URL", "").rstrip("/")
        legacy_vllm_base_url = _env_str("VLLM_BASE_URL", "http://127.0.0.1:8001").rstrip("/")
        legacy_llmd_base_url = _env_str("LLMD_BASE_URL", "").rstrip("/")

        if generation_base_url:
            self.GENERATION_BASE_URL = generation_base_url
        else:
            if self.GENERATION_BACKEND == "llmd":
                self.GENERATION_BASE_URL = legacy_llmd_base_url
            else:
                self.GENERATION_BASE_URL = legacy_vllm_base_url

        self.GENERATION_CHAT_PATH: str = _env_first_str(
            "GENERATION_CHAT_PATH",
            "LLMD_CHAT_PATH",
            default="/v1/chat/completions",
        )
        self.GENERATION_COMPLETIONS_PATH: str = _env_first_str(
            "GENERATION_COMPLETIONS_PATH",
            "LLMD_COMPLETIONS_PATH",
            default="/v1/completions",
        )
        self.MODEL_SERVER_TIMEOUT_S: float = _env_float("MODEL_SERVER_TIMEOUT_S", 300.0)

        self.MODEL_BACKEND: str = self.GENERATION_BACKEND
        self.SERVED_MODEL_NAME: str = self.GENERATION_MODEL_NAME
        self.LLMD_API_MODE: str = self.GENERATION_API_MODE
        self.LLMD_CHAT_PATH: str = self.GENERATION_CHAT_PATH
        self.LLMD_COMPLETIONS_PATH: str = self.GENERATION_COMPLETIONS_PATH

        self.VLLM_BASE_URL: str = (
            self.GENERATION_BASE_URL
            if self.GENERATION_BACKEND == "direct_vllm"
            else legacy_vllm_base_url
        )
        self.LLMD_BASE_URL: str = (
            self.GENERATION_BASE_URL
            if self.GENERATION_BACKEND == "llmd"
            else legacy_llmd_base_url
        )

        self.CACHE_SCOPE: str = _env_str("CACHE_SCOPE", "local-dev")

        self.EXACT_CACHE_ENABLED: bool = _env_bool("EXACT_CACHE_ENABLED", False)
        self.VALKEY_URL: str = _env_str("VALKEY_URL", "redis://127.0.0.1:6379/0")
        self.EXACT_CACHE_TTL_SEC: int = _env_int("EXACT_CACHE_TTL_SEC", 3600)

        self.SEM_CACHE_ENABLED: bool = _env_bool("SEM_CACHE_ENABLED", True)
        self.SEM_CACHE_THRESHOLD: float = _env_float("SEM_CACHE_THRESHOLD", 0.88)
        self.SEM_CACHE_TTL_SEC: int = _env_int("SEM_CACHE_TTL_SEC", 86400)

        self.SEM_CACHE_EMBED_MODEL: str = _env_str(
            "SEM_CACHE_EMBED_MODEL",
            "BAAI/bge-base-en-v1.5",
        )
        self.SEM_CACHE_EMBED_MODEL_PATH: str = _env_str("SEM_CACHE_EMBED_MODEL_PATH", "")
        self.SEM_CACHE_VECTOR_DIM: int = _env_int("SEM_CACHE_VECTOR_DIM", 768)

        self.SEM_CACHE_MONGO_URI: str = _env_str(
            "SEM_CACHE_MONGO_URI",
            "mongodb://127.0.0.1:27017",
        )
        self.SEM_CACHE_MONGO_DB: str = _env_str("SEM_CACHE_MONGO_DB", "semcache")
        self.SEM_CACHE_MONGO_COLLECTION: str = _env_str(
            "SEM_CACHE_MONGO_COLLECTION",
            "semantic_entries",
        )
        self.SEM_CACHE_MONGO_CONNECT_TIMEOUT_MS: int = _env_int(
            "SEM_CACHE_MONGO_CONNECT_TIMEOUT_MS",
            3000,
        )

        self.HISTORY_MONGO_URI: str = _env_str(
            "HISTORY_MONGO_URI",
            "mongodb://127.0.0.1:27017",
        )
        self.HISTORY_MONGO_DB: str = _env_str("HISTORY_MONGO_DB", "llm_service")
        self.HISTORY_MONGO_COLLECTION: str = _env_str("HISTORY_MONGO_COLLECTION", "messages")
        self.HISTORY_ENABLED: bool = _env_bool("HISTORY_ENABLED", True)

        self.SEM_CACHE_TOP_K: int = _env_int("SEM_CACHE_TOP_K", 5)
        self.SEM_CACHE_NORMALIZE: bool = _env_bool("SEM_CACHE_NORMALIZE", True)

        self.SEM_CACHE_MILVUS_URI: str = _env_str(
            "SEM_CACHE_MILVUS_URI",
            "http://127.0.0.1:19530",
        )
        self.SEM_CACHE_MILVUS_USER: str = _env_str("SEM_CACHE_MILVUS_USER", "root")
        self.SEM_CACHE_MILVUS_PASSWORD: str = _env_str("SEM_CACHE_MILVUS_PASSWORD", "Milvus")
        self.SEM_CACHE_MILVUS_SECURE: bool = _env_bool("SEM_CACHE_MILVUS_SECURE", False)
        self.SEM_CACHE_MILVUS_COLLECTION: str = _env_str(
            "SEM_CACHE_MILVUS_COLLECTION",
            "semcache_direct_v2",
        )

        self.SEM_CACHE_ALLOW_WITH_RAG: bool = _env_bool("SEM_CACHE_ALLOW_WITH_RAG", True)

        self.RAG_ENABLED: bool = _env_bool("RAG_ENABLED", True)
        self.RAG_BACKEND: str = _env_str("RAG_BACKEND", "milvus").lower()

        self.RAG_TOP_K: int = _env_int("RAG_TOP_K", 4)
        self.RAG_MAX_CONTEXT_CHARS: int = _env_int("RAG_MAX_CONTEXT_CHARS", 4000)
        self.RAG_SCORE_MODE: str = _env_str("RAG_SCORE_MODE", "similarity").lower()
        self.RAG_SCORE_THRESHOLD: float = _env_float("RAG_SCORE_THRESHOLD", 0.45)

        self.RAG_STORE_ROOT_DIR: str = _env_str("RAG_STORE_ROOT_DIR", "./rag_store_tenants")
        self.RAG_FALLBACK_TENANT: str = _env_str("RAG_FALLBACK_TENANT", "")
        self.RAG_MANIFEST_ROOT_DIR: str = _env_str(
            "RAG_MANIFEST_ROOT_DIR",
            self.RAG_STORE_ROOT_DIR,
        )
        self.RAG_LOCAL_EMBED_MODEL_PATH: str = _env_str(
            "RAG_LOCAL_EMBED_MODEL_PATH",
            self.SEM_CACHE_EMBED_MODEL_PATH or self.SEM_CACHE_EMBED_MODEL,
        )

        self.BGE_MODEL_PATH: str = _env_str("BGE_MODEL_PATH", "")
        self.BGE_DEVICE: str = _env_str("BGE_DEVICE", "cpu")
        self.BGE_DIM: int = _env_int("BGE_DIM", 768)
        self.BGE_NORMALIZE: bool = _env_bool("BGE_NORMALIZE", True)

        self.MILVUS_URI: str = _env_str("MILVUS_URI", "http://127.0.0.1:19530")
        self.MILVUS_TOKEN: str = _env_str("MILVUS_TOKEN", "root:Milvus")
        self.MILVUS_COLLECTION: str = _env_str("MILVUS_COLLECTION", "rag_chunks")
        self.MILVUS_VECTOR_FIELD: str = _env_str("MILVUS_VECTOR_FIELD", "embedding")
        self.MILVUS_TENANT_FIELD: str = _env_str("MILVUS_TENANT_FIELD", "tenant_id")
        self.MILVUS_METRIC_TYPE: str = _env_str("MILVUS_METRIC_TYPE", "COSINE").upper()

        self.RAG_OBJECT_STORE_ENABLED: bool = _env_bool("RAG_OBJECT_STORE_ENABLED", False)
        self.RAG_OBJECT_STORE_ENDPOINT_URL: str = _env_str("RAG_OBJECT_STORE_ENDPOINT_URL", "")
        self.RAG_OBJECT_STORE_ACCESS_KEY_ID: str = _env_str("RAG_OBJECT_STORE_ACCESS_KEY_ID", "")
        self.RAG_OBJECT_STORE_SECRET_ACCESS_KEY: str = _env_str(
            "RAG_OBJECT_STORE_SECRET_ACCESS_KEY",
            "",
        )
        self.RAG_OBJECT_STORE_BUCKET: str = _env_str("RAG_OBJECT_STORE_BUCKET", "")
        self.RAG_OBJECT_STORE_REGION: str = _env_str("RAG_OBJECT_STORE_REGION", "us-east-1")
        self.RAG_OBJECT_STORE_USE_SSL: bool = _env_bool("RAG_OBJECT_STORE_USE_SSL", False)
        self.RAG_OBJECT_STORE_VERIFY: str = _env_str("RAG_OBJECT_STORE_VERIFY", "false")
        self.RAG_OBJECT_STORE_PREFIX: str = _env_str("RAG_OBJECT_STORE_PREFIX", "")
        self.RAG_OBJECT_STORE_LOCAL_CACHE_TTL_SEC: int = _env_int(
            "RAG_OBJECT_STORE_LOCAL_CACHE_TTL_SEC",
            60,
        )

        self.KB_VERSION_FALLBACK: str = _env_str("KB_VERSION", "no-rag")
        self.SYSTEM_PROMPT_VERSION: str = _env_str("SYSTEM_PROMPT_VERSION", "v1")

        self.TOKENIZER_PATH: str = _env_str("TOKENIZER_PATH", "")
        self.TOKENIZER_LOCAL_ONLY: bool = _env_bool("TOKENIZER_LOCAL_ONLY", True)

        self.BENCHMARK_SHADOW_MODE: bool = _env_bool("BENCHMARK_SHADOW_MODE", True)
        self.RAG_RETRIEVE_EVERY_REQUEST: bool = _env_bool("RAG_RETRIEVE_EVERY_REQUEST", True)
        self.RETURN_DEBUG_BLOCKS: bool = _env_bool("RETURN_DEBUG_BLOCKS", True)

        self.HF_HUB_OFFLINE: bool = _env_bool("HF_HUB_OFFLINE", False)
        self.TRANSFORMERS_OFFLINE: bool = _env_bool("TRANSFORMERS_OFFLINE", False)

        self.AUTH_ENABLED: bool = _env_bool("AUTH_ENABLED", False)
        self.KEYCLOAK_ISSUER: str = _env_str("KEYCLOAK_ISSUER", "").rstrip("/")
        self.KEYCLOAK_JWKS_URL: str = _env_str("KEYCLOAK_JWKS_URL", "")
        self.KEYCLOAK_AUDIENCE: str = _env_str("KEYCLOAK_AUDIENCE", "")
        self.TENANT_CLAIM: str = _env_str("TENANT_CLAIM", "tenant_id")
        self.JWKS_CACHE_TTL_SEC: int = _env_int("JWKS_CACHE_TTL_SEC", 300)

        self.DEV_TENANT_ID: str = _env_str("DEV_TENANT_ID", "tenantA")

    @property
    def model_base_url(self) -> str:
        return self.GENERATION_BASE_URL

    def validate(self) -> None:
        if self.GENERATION_BACKEND not in _VALID_MODEL_BACKENDS:
            raise ValueError(
                f"Unsupported GENERATION_BACKEND={self.GENERATION_BACKEND}. "
                f"Expected one of {sorted(_VALID_MODEL_BACKENDS)}"
            )

        if self.GENERATION_API_MODE not in _VALID_LLMD_API_MODES:
            raise ValueError(
                f"Unsupported GENERATION_API_MODE={self.GENERATION_API_MODE}. "
                f"Expected one of {sorted(_VALID_LLMD_API_MODES)}"
            )

        if self.RAG_SCORE_MODE not in _VALID_RAG_SCORE_MODES:
            raise ValueError(
                f"Unsupported RAG_SCORE_MODE={self.RAG_SCORE_MODE}. "
                f"Expected one of {sorted(_VALID_RAG_SCORE_MODES)}"
            )

        if not self.GENERATION_BASE_URL:
            raise ValueError(
                "Missing generation base URL. Set GENERATION_BASE_URL "
                "(or legacy LLMD_BASE_URL / VLLM_BASE_URL)."
            )