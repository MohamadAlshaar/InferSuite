from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, ConnectionClosedError
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)
from pymongo import ASCENDING, MongoClient

from src.service.cache.semantic_schema import ensure_semantic_cache_collection


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip()


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except Exception:
        return default


def _parse_verify(raw: str):
    value = (raw or "").strip()
    if value.lower() in {"1", "true", "yes", "on"}:
        return True
    if value.lower() in {"0", "false", "no", "off", ""}:
        return False
    return value


def _parse_milvus_uri(uri: str) -> tuple[str, bool]:
    uri = (uri or "").strip()
    if not uri:
        return "http://127.0.0.1:19530", False

    if "://" not in uri:
        return f"http://{uri}", False

    u = urlparse(uri)
    secure = u.scheme.lower() == "https"
    normalized = f"{u.scheme}://{u.hostname}:{u.port or 19530}"
    return normalized, secure


def _retry(name: str, fn: Callable[[], None], attempts: int = 30, sleep_s: float = 5.0) -> None:
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            print(f"[bootstrap] {name}: start attempt {attempt}/{attempts}", flush=True)
            fn()
            print(f"[bootstrap] {name}: ok", flush=True)
            return
        except Exception as exc:
            last_error = exc
            print(f"[bootstrap] {name}: attempt {attempt}/{attempts} failed: {exc}", flush=True)
            if attempt < attempts:
                time.sleep(sleep_s)

    raise RuntimeError(f"{name} failed after {attempts} attempts: {last_error}") from last_error


def _tenant_manifest_dirs(root: Path) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / "manifest.json").exists():
            out.append(p)
    return out


def _count_tenant_manifest_dirs(root: Path) -> int:
    return len(_tenant_manifest_dirs(root))


def ensure_seaweed_bucket() -> None:
    enabled = _env_bool("RAG_OBJECT_STORE_ENABLED", False)
    if not enabled:
        print("[bootstrap] object store disabled; skipping bucket init", flush=True)
        return

    endpoint_url = _env("RAG_OBJECT_STORE_ENDPOINT_URL")
    access_key_id = _env("RAG_OBJECT_STORE_ACCESS_KEY_ID")
    secret_access_key = _env("RAG_OBJECT_STORE_SECRET_ACCESS_KEY")
    bucket = _env("RAG_OBJECT_STORE_BUCKET")
    region = _env("RAG_OBJECT_STORE_REGION", "us-east-1")
    use_ssl = _env_bool("RAG_OBJECT_STORE_USE_SSL", False)
    verify = _parse_verify(_env("RAG_OBJECT_STORE_VERIFY", "false"))

    if not endpoint_url or not access_key_id or not secret_access_key or not bucket:
        raise RuntimeError("missing object store configuration")

    session = boto3.session.Session()
    client = session.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name=region,
        use_ssl=use_ssl,
        verify=verify,
        config=Config(
            region_name=region,
            signature_version="s3v4",
            s3={"addressing_style": "path", "payload_signing_enabled": False},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
            retries={"max_attempts": 3, "mode": "standard"},
            tcp_keepalive=True,
            user_agent_extra="llm-service-kernel-bootstrap",
        ),
    )

    try:
        resp = client.list_buckets()
        names = {b.get("Name") for b in resp.get("Buckets", []) if b.get("Name")}
        if bucket in names:
            return
    except Exception:
        pass

    try:
        client.create_bucket(Bucket=bucket)
        return
    except ClientError as exc:
        status = int(exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if status in {200, 409} or code in {"BucketAlreadyOwnedByYou", "BucketAlreadyExists"}:
            return
        raise
    except ConnectionClosedError:
        resp = client.list_buckets()
        names = {b.get("Name") for b in resp.get("Buckets", []) if b.get("Name")}
        if bucket in names:
            return
        raise


def ensure_semantic_cache_mongo() -> None:
    mongo_uri = _env("SEM_CACHE_MONGO_URI")
    mongo_db = _env("SEM_CACHE_MONGO_DB", "semcache")
    mongo_collection = _env("SEM_CACHE_MONGO_COLLECTION", "semantic_entries")
    timeout_ms = _env_int("SEM_CACHE_MONGO_CONNECT_TIMEOUT_MS", 3000)

    if not mongo_uri:
        raise RuntimeError("missing SEM_CACHE_MONGO_URI")

    client = MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=timeout_ms,
        tz_aware=True,
        appname="llm-service-kernel-bootstrap",
    )

    try:
        client.admin.command("ping")
        collection = client[mongo_db][mongo_collection]
        collection.create_index([("accept_key", ASCENDING)], name="idx_semantic_accept_key")
        collection.create_index(
            [("expires_at", ASCENDING)],
            name="idx_semantic_expires_at_ttl",
            expireAfterSeconds=0,
        )
    finally:
        client.close()


def ensure_semantic_cache_milvus() -> None:
    collection_name = _env("SEM_CACHE_MILVUS_COLLECTION", "semcache_direct_v2")
    milvus_uri = _env("SEM_CACHE_MILVUS_URI")
    milvus_user = _env("SEM_CACHE_MILVUS_USER", "root")
    milvus_password = _env("SEM_CACHE_MILVUS_PASSWORD", "Milvus")
    milvus_secure = _env_bool("SEM_CACHE_MILVUS_SECURE", False)
    vector_dim = _env_int("SEM_CACHE_VECTOR_DIM", 768)

    if not collection_name:
        raise RuntimeError("missing SEM_CACHE_MILVUS_COLLECTION")

    normalized_uri, secure_from_uri = _parse_milvus_uri(milvus_uri)
    alias = "bootstrap_semcache"

    connections.connect(
        alias=alias,
        uri=normalized_uri,
        user=milvus_user,
        password=milvus_password,
        secure=milvus_secure or secure_from_uri,
        timeout=30,
    )

    try:
        collection = ensure_semantic_cache_collection(
            alias=alias,
            collection_name=collection_name,
            vector_dim=vector_dim,
            create_if_missing=True,
        )
        try:
            collection.load(timeout=10)
        except Exception:
            pass
    finally:
        try:
            connections.disconnect(alias)
        except Exception:
            pass


def seed_rag_manifests() -> None:
    seed_dir = Path(_env("RAG_SEED_MANIFEST_DIR", "/app/fastapi_runtime_assets/rag_store_tenants"))
    runtime_dir = Path(_env("RAG_MANIFEST_ROOT_DIR", "/rag_store_tenants"))

    runtime_dir.mkdir(parents=True, exist_ok=True)

    seed_count = _count_tenant_manifest_dirs(seed_dir)
    runtime_before = _count_tenant_manifest_dirs(runtime_dir)

    print(
        f"[bootstrap] rag manifests: seed_count={seed_count} runtime_before={runtime_before} "
        f"seed_dir={seed_dir} runtime_dir={runtime_dir}",
        flush=True,
    )

    if seed_dir.exists() and seed_dir.is_dir():
        for tenant_dir in _tenant_manifest_dirs(seed_dir):
            dst = runtime_dir / tenant_dir.name
            manifest = dst / "manifest.json"

            if manifest.exists():
                print(
                    f"[bootstrap] manifest already present for tenant {tenant_dir.name}; keeping existing",
                    flush=True,
                )
                continue

            shutil.copytree(tenant_dir, dst, dirs_exist_ok=True)
            print(f"[bootstrap] seeded tenant manifest dir: {dst}", flush=True)

    runtime_after = _count_tenant_manifest_dirs(runtime_dir)
    require_manifests = _env_bool("RAG_REQUIRE_MANIFESTS", True)

    if require_manifests and runtime_after <= 0:
        raise RuntimeError(
            f"no tenant manifest directories available under {runtime_dir}; "
            f"seed_dir={seed_dir} is empty or missing"
        )


def _build_rag_schema(vector_field: str, tenant_field: str, vector_dim: int) -> CollectionSchema:
    fields = [
        FieldSchema(
            name="pk",
            dtype=DataType.VARCHAR,
            is_primary=True,
            auto_id=False,
            max_length=128,
        ),
        FieldSchema(
            name=tenant_field,
            dtype=DataType.VARCHAR,
            max_length=128,
        ),
        FieldSchema(
            name=vector_field,
            dtype=DataType.FLOAT_VECTOR,
            dim=int(vector_dim),
        ),
        FieldSchema(
            name="source",
            dtype=DataType.VARCHAR,
            max_length=1024,
        ),
        FieldSchema(
            name="page",
            dtype=DataType.INT64,
        ),
        FieldSchema(
            name="text",
            dtype=DataType.VARCHAR,
            max_length=65535,
        ),
        FieldSchema(
            name="chunk_id",
            dtype=DataType.VARCHAR,
            max_length=256,
        ),
        FieldSchema(
            name="object_key",
            dtype=DataType.VARCHAR,
            max_length=2048,
        ),
        FieldSchema(
            name="pdf_object_key",
            dtype=DataType.VARCHAR,
            max_length=2048,
        ),
        FieldSchema(
            name="text_sha256",
            dtype=DataType.VARCHAR,
            max_length=64,
        ),
    ]
    try:
        return CollectionSchema(
            fields=fields,
            description="RAG chunks for llm-service-kernel",
            enable_dynamic_field=True,
        )
    except TypeError:
        return CollectionSchema(
            fields=fields,
            description="RAG chunks for llm-service-kernel",
        )


def _validate_rag_collection(
    *,
    collection: Collection,
    vector_field: str,
    tenant_field: str,
    expected_vector_dim: int,
) -> None:
    schema = getattr(collection, "schema", None)
    if schema is None:
        raise RuntimeError("RAG collection has no schema")

    fields = {field.name: field for field in schema.fields}
    for required in ("pk", tenant_field, vector_field):
        if required not in fields:
            raise RuntimeError(f"RAG collection '{collection.name}' missing required field '{required}'")

    if fields["pk"].dtype != DataType.VARCHAR:
        raise RuntimeError(f"RAG collection '{collection.name}' field 'pk' must be VARCHAR")

    if fields[tenant_field].dtype != DataType.VARCHAR:
        raise RuntimeError(
            f"RAG collection '{collection.name}' field '{tenant_field}' must be VARCHAR"
        )

    if fields[vector_field].dtype != DataType.FLOAT_VECTOR:
        raise RuntimeError(
            f"RAG collection '{collection.name}' field '{vector_field}' must be FLOAT_VECTOR"
        )

    params = getattr(fields[vector_field], "params", {}) or {}
    actual_dim = params.get("dim")
    if actual_dim is not None and int(actual_dim) != int(expected_vector_dim):
        raise RuntimeError(
            f"RAG collection '{collection.name}' vector dim={actual_dim}, expected {expected_vector_dim}"
        )


def ensure_rag_milvus_collection() -> None:
    collection_name = _env("MILVUS_COLLECTION", "rag_chunks")
    milvus_uri = _env("MILVUS_URI")
    milvus_token = _env("MILVUS_TOKEN", "root:Milvus")
    metric_type = _env("MILVUS_METRIC_TYPE", "COSINE").upper()
    vector_field = _env("MILVUS_VECTOR_FIELD", "embedding")
    tenant_field = _env("MILVUS_TENANT_FIELD", "tenant_id")
    vector_dim = _env_int("BGE_DIM", 768)
    create_if_missing = _env_bool("RAG_CREATE_COLLECTION_IF_MISSING", True)
    require_non_empty = _env_bool("RAG_REQUIRE_NON_EMPTY_COLLECTION", True)

    if not collection_name:
        raise RuntimeError("missing MILVUS_COLLECTION")

    normalized_uri, secure_from_uri = _parse_milvus_uri(milvus_uri)
    alias = "bootstrap_rag"

    user = "root"
    password = "Milvus"
    if ":" in milvus_token:
        user, password = milvus_token.split(":", 1)

    connections.connect(
        alias=alias,
        uri=normalized_uri,
        user=user.strip(),
        password=password.strip(),
        secure=secure_from_uri,
        timeout=30,
    )

    try:
        _rpc_timeout = 30

        exists = utility.has_collection(collection_name, using=alias, timeout=_rpc_timeout)

        if not exists:
            if not create_if_missing:
                raise RuntimeError(f"RAG Milvus collection '{collection_name}' does not exist")

            print(
                f"[bootstrap] creating RAG Milvus collection '{collection_name}' "
                f"vector_field={vector_field} tenant_field={tenant_field} vector_dim={vector_dim}",
                flush=True,
            )

            collection = Collection(
                name=collection_name,
                schema=_build_rag_schema(
                    vector_field=vector_field,
                    tenant_field=tenant_field,
                    vector_dim=vector_dim,
                ),
                using=alias,
            )
            collection.create_index(
                field_name=vector_field,
                index_params={
                    "index_type": "AUTOINDEX",
                    "metric_type": metric_type,
                    "params": {},
                },
                timeout=_rpc_timeout,
            )
        else:
            collection = Collection(name=collection_name, using=alias)

        _validate_rag_collection(
            collection=collection,
            vector_field=vector_field,
            tenant_field=tenant_field,
            expected_vector_dim=vector_dim,
        )

        try:
            collection.load(timeout=10)
        except Exception:
            pass

        num_entities = int(getattr(collection, "num_entities", 0) or 0)
        print(
            f"[bootstrap] RAG Milvus collection '{collection_name}' num_entities={num_entities}",
            flush=True,
        )

        if require_non_empty and num_entities <= 0:
            raise RuntimeError(
                f"RAG Milvus collection '{collection_name}' exists but is empty; "
                "RAG data has not been provisioned"
            )
    finally:
        try:
            connections.disconnect(alias)
        except Exception:
            pass


def main() -> int:
    if _env_bool("SEM_CACHE_ENABLED", False):
        _retry("mongo semantic-cache indexes", ensure_semantic_cache_mongo)
        _retry("milvus semantic-cache collection", ensure_semantic_cache_milvus)
    else:
        print("[bootstrap] semantic cache disabled; skipping semantic-cache bootstrap", flush=True)

    if _env_bool("RAG_ENABLED", False):
        _retry("seaweed bucket", ensure_seaweed_bucket)
        _retry("seed rag manifests", seed_rag_manifests, attempts=3, sleep_s=1.0)
        _retry("rag milvus collection", ensure_rag_milvus_collection)
    else:
        print("[bootstrap] rag disabled; skipping rag bootstrap", flush=True)

    print("[bootstrap] complete", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[bootstrap] fatal: {exc}", file=sys.stderr, flush=True)
        raise
