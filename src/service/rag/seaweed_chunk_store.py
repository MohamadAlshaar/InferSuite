from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import boto3
from botocore.config import Config


def _env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class SeaweedChunkStoreConfig:
    enabled: bool
    endpoint_url: str
    access_key_id: str
    secret_access_key: str
    bucket: str
    region_name: str
    use_ssl: bool
    verify: bool | str
    prefix: str
    local_cache_ttl_sec: int

    @classmethod
    def from_env(cls, prefix: str = "RAG_OBJECT_STORE_") -> "SeaweedChunkStoreConfig":
        verify_raw = os.getenv(f"{prefix}VERIFY", "false").strip()
        if verify_raw.lower() in {"1", "true", "yes", "on"}:
            verify: bool | str = True
        elif verify_raw.lower() in {"0", "false", "no", "off"}:
            verify = False
        else:
            verify = verify_raw

        return cls(
            enabled=_env_bool(f"{prefix}ENABLED", "0"),
            endpoint_url=os.getenv(f"{prefix}ENDPOINT_URL", "").strip(),
            access_key_id=os.getenv(f"{prefix}ACCESS_KEY_ID", "").strip(),
            secret_access_key=os.getenv(f"{prefix}SECRET_ACCESS_KEY", "").strip(),
            bucket=os.getenv(f"{prefix}BUCKET", "").strip(),
            region_name=os.getenv(f"{prefix}REGION", "us-east-1").strip(),
            use_ssl=_env_bool(f"{prefix}USE_SSL", "0"),
            verify=verify,
            prefix=os.getenv(f"{prefix}PREFIX", "").strip().strip("/"),
            local_cache_ttl_sec=int(os.getenv(f"{prefix}LOCAL_CACHE_TTL_SEC", "60")),
        )


def _join_key(prefix: str, key: str) -> str:
    clean_key = str(key or "").strip().lstrip("/")
    clean_prefix = str(prefix or "").strip().strip("/")
    if not clean_prefix:
        return clean_key
    if not clean_key:
        return clean_prefix
    return f"{clean_prefix}/{clean_key}"


class SeaweedChunkStore:
    """
    Fetches RAG chunk payloads from SeaweedFS via S3 API.

    Expected object payloads:
      1) JSON object:
         {
           "text": "...",
           "source": "...",
           "page": "...",
           "chunk_id": "...",
           "tenant_id": "..."
         }

      2) Plain UTF-8 text blob
    """

    def __init__(self, cfg: SeaweedChunkStoreConfig):
        self.cfg = cfg
        self.enabled = bool(
            cfg.enabled
            and cfg.endpoint_url
            and cfg.access_key_id
            and cfg.secret_access_key
            and cfg.bucket
        )
        self._client = None
        self._cache: Dict[str, tuple[float, Dict[str, Any]]] = {}

        if not self.enabled:
            return

        session = boto3.session.Session()
        self._client = session.client(
            "s3",
            endpoint_url=cfg.endpoint_url,
            aws_access_key_id=cfg.access_key_id,
            aws_secret_access_key=cfg.secret_access_key,
            region_name=cfg.region_name,
            use_ssl=cfg.use_ssl,
            verify=cfg.verify,
            config=Config(
                region_name=cfg.region_name,
                signature_version="s3v4",
                s3={
                    "addressing_style": "path",
                    "payload_signing_enabled": False,
                },
                request_checksum_calculation="when_required",
                response_checksum_validation="when_required",
                retries={"max_attempts": 3, "mode": "standard"},
                tcp_keepalive=True,
                user_agent_extra="llm-service-kernel-rag-seaweed",
            ),
        )

    @classmethod
    def from_env(cls, prefix: str = "RAG_OBJECT_STORE_") -> "SeaweedChunkStore":
        return cls(SeaweedChunkStoreConfig.from_env(prefix=prefix))

    def _cache_get(self, object_key: str) -> Optional[Dict[str, Any]]:
        ttl = max(0, int(self.cfg.local_cache_ttl_sec))
        if ttl <= 0:
            return None

        hit = self._cache.get(object_key)
        if hit is None:
            return None

        ts, payload = hit
        if time.time() - ts > ttl:
            self._cache.pop(object_key, None)
            return None

        return payload

    def _cache_put(self, object_key: str, payload: Dict[str, Any]) -> None:
        ttl = max(0, int(self.cfg.local_cache_ttl_sec))
        if ttl <= 0:
            return
        self._cache[object_key] = (time.time(), payload)

    def get_chunk(self, object_key: str) -> Optional[Dict[str, Any]]:
        if not self.enabled or self._client is None:
            return None

        object_key = str(object_key or "").strip()
        if not object_key:
            return None

        cached = self._cache_get(object_key)
        if cached is not None:
            return cached

        full_key = _join_key(self.cfg.prefix, object_key)

        try:
            response = self._client.get_object(Bucket=self.cfg.bucket, Key=full_key)
            raw = response["Body"].read()
        except Exception:
            return None

        if not raw:
            return None

        try:
            payload = json.loads(raw.decode("utf-8"))
            if isinstance(payload, dict):
                out = {
                    "text": str(payload.get("text") or ""),
                    "source": payload.get("source"),
                    "page": payload.get("page"),
                    "chunk_id": payload.get("chunk_id"),
                    "tenant_id": payload.get("tenant_id"),
                    "object_key": object_key,
                }
                self._cache_put(object_key, out)
                return out
        except Exception:
            pass

        try:
            out = {
                "text": raw.decode("utf-8"),
                "source": None,
                "page": None,
                "chunk_id": None,
                "tenant_id": None,
                "object_key": object_key,
            }
            self._cache_put(object_key, out)
            return out
        except Exception:
            return None
