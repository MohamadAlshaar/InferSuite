from __future__ import annotations

import os
import uuid
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from pymilvus import Collection, connections, utility
from pymongo import MongoClient

from src.service.auth.keycloak import KeycloakJWTVerifier
from src.service.cache.semantic_gptcache import SemanticCache
from src.service.cache.semantic_schema import validate_semantic_cache_collection
from src.service.clients.vllm_client import VLLMClient
from src.service.config import Settings
from src.service.history.mongo_history import MessageHistoryWriter
from src.service.orchestrator.chat import ChatOrchestrator
from src.service.rag.tenant_router import TenantRAGRouter
from src.service.utils.tokenizer import TokenCounter


@dataclass
class AppRuntime:
    settings: Settings
    vllm: VLLMClient
    exact_cache: Any
    semantic_cache: Optional[SemanticCache]
    rag_router: Optional[TenantRAGRouter]
    auth_verifier: Optional[KeycloakJWTVerifier]
    tokenizer: Any
    orchestrator: ChatOrchestrator
    startup_checks: dict[str, Any]


def _print_startup_banner(settings: Settings) -> None:
    print(f"STARTUP GENERATION_BACKEND={settings.GENERATION_BACKEND}", flush=True)
    print(f"STARTUP GENERATION_BASE_URL={settings.GENERATION_BASE_URL}", flush=True)
    print(f"STARTUP GENERATION_MODEL_NAME={settings.GENERATION_MODEL_NAME}", flush=True)
    print(f"STARTUP RAG_ENABLED={settings.RAG_ENABLED}", flush=True)
    print(f"STARTUP SEM_CACHE_ENABLED={settings.SEM_CACHE_ENABLED}", flush=True)
    print(f"STARTUP SEM_CACHE_VECTOR_DIM={settings.SEM_CACHE_VECTOR_DIM}", flush=True)
    print(f"STARTUP BENCHMARK_SHADOW_MODE={settings.BENCHMARK_SHADOW_MODE}", flush=True)
    if settings.GENERATION_BACKEND == "llmd":
        print(f"STARTUP GENERATION_API_MODE={settings.GENERATION_API_MODE}", flush=True)
        print(f"STARTUP GENERATION_CHAT_PATH={settings.GENERATION_CHAT_PATH}", flush=True)
        print(f"STARTUP GENERATION_COMPLETIONS_PATH={settings.GENERATION_COMPLETIONS_PATH}", flush=True)


def _make_models_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/v1/models"


def _check_generation_backend(settings: Settings) -> dict[str, Any]:
    url = _make_models_url(settings.GENERATION_BASE_URL)
    result = {
        "url": url,
        "reachable": False,
        "status_code": None,
        "reason": None,
    }

    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            result["reachable"] = True
            result["status_code"] = getattr(resp, "status", None)
            return result
    except urllib.error.HTTPError as exc:
        result["status_code"] = exc.code
        result["reason"] = f"http_error: {exc}"
        return result
    except Exception as exc:
        result["reason"] = str(exc)
        return result


def _check_mongo(uri: str, timeout_ms: int) -> dict[str, Any]:
    result = {
        "uri": uri,
        "reachable": False,
        "reason": None,
    }

    if not uri:
        result["reason"] = "missing_uri"
        return result

    client = None
    try:
        client = MongoClient(
            uri,
            serverSelectionTimeoutMS=timeout_ms,
            tz_aware=True,
            appname="llm-service-kernel-startup-check",
        )
        client.admin.command("ping")
        result["reachable"] = True
        return result
    except Exception as exc:
        result["reason"] = str(exc)
        return result
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


def _split_milvus_token(token: str) -> tuple[str, str]:
    token = (token or "").strip()
    if ":" in token:
        user, password = token.split(":", 1)
        return user.strip(), password.strip()
    return "root", "Milvus"


def _check_semantic_cache_milvus(settings: Settings) -> dict[str, Any]:
    result = {
        "uri": settings.SEM_CACHE_MILVUS_URI,
        "collection": settings.SEM_CACHE_MILVUS_COLLECTION,
        "reachable": False,
        "collection_exists": False,
        "schema_ok": False,
        "reason": None,
    }

    alias = f"startup_semcache_{uuid.uuid4().hex[:8]}"
    try:
        connections.connect(
            alias=alias,
            uri=settings.SEM_CACHE_MILVUS_URI,
            user=settings.SEM_CACHE_MILVUS_USER,
            password=settings.SEM_CACHE_MILVUS_PASSWORD,
            secure=settings.SEM_CACHE_MILVUS_SECURE,
        )
        result["reachable"] = True

        exists = utility.has_collection(settings.SEM_CACHE_MILVUS_COLLECTION, using=alias)
        result["collection_exists"] = bool(exists)
        if not exists:
            result["reason"] = "collection_missing"
            return result

        collection = Collection(name=settings.SEM_CACHE_MILVUS_COLLECTION, using=alias)
        validate_semantic_cache_collection(
            collection,
            expected_vector_dim=settings.SEM_CACHE_VECTOR_DIM,
        )
        result["schema_ok"] = True
        return result
    except Exception as exc:
        result["reason"] = str(exc)
        return result
    finally:
        try:
            connections.disconnect(alias)
        except Exception:
            pass


def _check_rag_milvus(settings: Settings) -> dict[str, Any]:
    result = {
        "uri": settings.MILVUS_URI,
        "collection": settings.MILVUS_COLLECTION,
        "reachable": False,
        "collection_exists": False,
        "reason": None,
    }

    user, password = _split_milvus_token(settings.MILVUS_TOKEN)
    alias = f"startup_rag_{uuid.uuid4().hex[:8]}"

    try:
        connections.connect(
            alias=alias,
            uri=settings.MILVUS_URI,
            user=user,
            password=password,
            secure=False,
        )
        result["reachable"] = True

        exists = utility.has_collection(settings.MILVUS_COLLECTION, using=alias)
        result["collection_exists"] = bool(exists)
        if not exists:
            result["reason"] = "collection_missing"
        return result
    except Exception as exc:
        result["reason"] = str(exc)
        return result
    finally:
        try:
            connections.disconnect(alias)
        except Exception:
            pass


def _check_dir(path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    exists = path.exists()
    is_dir = path.is_dir()
    non_empty = False

    if exists and is_dir:
        try:
            non_empty = any(path.iterdir())
        except Exception:
            non_empty = False

    return {
        "path": str(path),
        "exists": exists,
        "is_dir": is_dir,
        "non_empty": non_empty,
    }


def _semantic_cache_status(
    *,
    settings: Settings,
    sem_cache: Optional[SemanticCache],
    checks: dict[str, Any],
) -> dict[str, Any]:
    configured = bool(settings.SEM_CACHE_ENABLED)
    runtime_enabled = bool(sem_cache is not None and getattr(sem_cache, "enabled", False))
    init_error = getattr(sem_cache, "init_error", None) if sem_cache is not None else None
    mongo = checks.get("semantic_cache_mongo", {}) or {}
    milvus = checks.get("semantic_cache_milvus", {}) or {}

    if not configured:
        return {
            "configured": False,
            "runtime_enabled": runtime_enabled,
            "usable": False,
            "reason": "disabled_in_config",
            "init_error": init_error,
        }

    if init_error:
        return {
            "configured": True,
            "runtime_enabled": runtime_enabled,
            "usable": False,
            "reason": f"runtime_init_error:{init_error}",
            "init_error": init_error,
        }

    if not runtime_enabled:
        return {
            "configured": True,
            "runtime_enabled": False,
            "usable": False,
            "reason": "runtime_disabled",
            "init_error": init_error,
        }

    if not mongo.get("reachable"):
        return {
            "configured": True,
            "runtime_enabled": True,
            "usable": False,
            "reason": mongo.get("reason") or "mongo_unreachable",
            "init_error": init_error,
        }

    if not milvus.get("reachable"):
        return {
            "configured": True,
            "runtime_enabled": True,
            "usable": False,
            "reason": milvus.get("reason") or "milvus_unreachable",
            "init_error": init_error,
        }

    if not milvus.get("collection_exists"):
        return {
            "configured": True,
            "runtime_enabled": True,
            "usable": False,
            "reason": "collection_missing",
            "init_error": init_error,
        }

    if not milvus.get("schema_ok"):
        return {
            "configured": True,
            "runtime_enabled": True,
            "usable": False,
            "reason": milvus.get("reason") or "schema_invalid",
            "init_error": init_error,
        }

    return {
        "configured": True,
        "runtime_enabled": True,
        "usable": True,
        "reason": None,
        "init_error": init_error,
    }


def _rag_status(
    *,
    settings: Settings,
    rag_router: Optional[TenantRAGRouter],
    checks: dict[str, Any],
) -> dict[str, Any]:
    configured = bool(settings.RAG_ENABLED)
    runtime_enabled = bool(rag_router is not None and getattr(rag_router, "enabled", False))
    init_error = getattr(rag_router, "init_error", None) if rag_router is not None else None
    milvus = checks.get("rag_milvus", {}) or {}
    manifest_root = checks.get("rag_manifest_root_dir", {}) or {}

    if not configured:
        return {
            "configured": False,
            "runtime_enabled": runtime_enabled,
            "usable": False,
            "reason": "disabled_in_config",
            "init_error": init_error,
        }

    if init_error:
        return {
            "configured": True,
            "runtime_enabled": runtime_enabled,
            "usable": False,
            "reason": f"runtime_init_error:{init_error}",
            "init_error": init_error,
        }

    if not runtime_enabled:
        return {
            "configured": True,
            "runtime_enabled": False,
            "usable": False,
            "reason": "runtime_disabled",
            "init_error": init_error,
        }

    if not milvus.get("reachable"):
        return {
            "configured": True,
            "runtime_enabled": True,
            "usable": False,
            "reason": milvus.get("reason") or "milvus_unreachable",
            "init_error": init_error,
        }

    if not milvus.get("collection_exists"):
        return {
            "configured": True,
            "runtime_enabled": True,
            "usable": False,
            "reason": "collection_missing",
            "init_error": init_error,
        }

    if not manifest_root.get("exists"):
        return {
            "configured": True,
            "runtime_enabled": True,
            "usable": False,
            "reason": "manifest_root_missing",
            "init_error": init_error,
        }

    if not manifest_root.get("non_empty"):
        return {
            "configured": True,
            "runtime_enabled": True,
            "usable": False,
            "reason": "manifest_root_empty",
            "init_error": init_error,
        }

    return {
        "configured": True,
        "runtime_enabled": True,
        "usable": True,
        "reason": None,
        "init_error": init_error,
    }


def _ready_status(
    *,
    settings: Settings,
    checks: dict[str, Any],
) -> dict[str, Any]:
    generation = checks.get("generation_status", {}) or {}
    semantic = checks.get("semantic_cache_status", {}) or {}
    rag = checks.get("rag_status", {}) or {}

    if not generation.get("usable"):
        return {
            "ok": False,
            "reason": generation.get("reason") or "generation_backend_unusable",
        }

    if settings.SEM_CACHE_ENABLED and not semantic.get("usable"):
        return {
            "ok": False,
            "reason": f"semantic_cache_unusable:{semantic.get('reason')}",
        }

    if settings.RAG_ENABLED and not rag.get("usable"):
        return {
            "ok": False,
            "reason": f"rag_unusable:{rag.get('reason')}",
        }

    return {
        "ok": True,
        "reason": None,
    }


def _build_startup_checks(
    settings: Settings,
    sem_cache: Optional[SemanticCache],
    rag_router: Optional[TenantRAGRouter],
) -> dict[str, Any]:
    checks: dict[str, Any] = {}

    checks["generation_backend"] = _check_generation_backend(settings)

    checks["semantic_cache"] = {
        "configured_enabled": settings.SEM_CACHE_ENABLED,
        "runtime_enabled": bool(sem_cache is not None and getattr(sem_cache, "enabled", False)),
        "init_error": getattr(sem_cache, "init_error", None) if sem_cache is not None else None,
    }

    if settings.SEM_CACHE_ENABLED:
        checks["semantic_cache_mongo"] = _check_mongo(
            settings.SEM_CACHE_MONGO_URI,
            settings.SEM_CACHE_MONGO_CONNECT_TIMEOUT_MS,
        )
        checks["semantic_cache_milvus"] = _check_semantic_cache_milvus(settings)
    else:
        checks["semantic_cache_mongo"] = {
            "uri": settings.SEM_CACHE_MONGO_URI,
            "reachable": False,
            "reason": "semantic_cache_disabled",
        }
        checks["semantic_cache_milvus"] = {
            "uri": settings.SEM_CACHE_MILVUS_URI,
            "collection": settings.SEM_CACHE_MILVUS_COLLECTION,
            "reachable": False,
            "collection_exists": False,
            "schema_ok": False,
            "reason": "semantic_cache_disabled",
        }

    rag_seed_dir = os.getenv("RAG_SEED_MANIFEST_DIR", "/app/fastapi_runtime_assets/rag_store_tenants")
    checks["rag"] = {
        "configured_enabled": settings.RAG_ENABLED,
        "runtime_enabled": bool(rag_router is not None and getattr(rag_router, "enabled", False)),
        "init_error": getattr(rag_router, "init_error", None) if rag_router is not None else None,
    }
    checks["rag_manifest_root_dir"] = _check_dir(settings.RAG_MANIFEST_ROOT_DIR)
    checks["rag_seed_manifest_dir"] = _check_dir(rag_seed_dir)

    if settings.RAG_ENABLED:
        checks["rag_milvus"] = _check_rag_milvus(settings)
    else:
        checks["rag_milvus"] = {
            "uri": settings.MILVUS_URI,
            "collection": settings.MILVUS_COLLECTION,
            "reachable": False,
            "collection_exists": False,
            "reason": "rag_disabled",
        }

    checks["tokenizer"] = {
        "path": settings.TOKENIZER_PATH,
        "exists": bool(settings.TOKENIZER_PATH) and Path(settings.TOKENIZER_PATH).exists(),
        "local_only": settings.TOKENIZER_LOCAL_ONLY,
    }

    checks["generation_status"] = {
        "configured": True,
        "usable": bool(checks["generation_backend"].get("reachable")),
        "reason": None
        if checks["generation_backend"].get("reachable")
        else checks["generation_backend"].get("reason") or "generation_backend_unreachable",
    }
    checks["semantic_cache_status"] = _semantic_cache_status(
        settings=settings,
        sem_cache=sem_cache,
        checks=checks,
    )
    checks["rag_status"] = _rag_status(
        settings=settings,
        rag_router=rag_router,
        checks=checks,
    )
    checks["ready_status"] = _ready_status(
        settings=settings,
        checks=checks,
    )

    return checks


def refresh_startup_checks(runtime: AppRuntime) -> dict[str, Any]:
    checks = _build_startup_checks(
        runtime.settings,
        runtime.semantic_cache,
        runtime.rag_router,
    )
    runtime.startup_checks = checks
    return checks


def _print_startup_summary(checks: dict[str, Any]) -> None:
    gb = checks.get("generation_backend", {})
    print(
        f"STARTUP CHECK generation_backend reachable={gb.get('reachable')} "
        f"status={gb.get('status_code')} reason={gb.get('reason')}",
        flush=True,
    )

    sc = checks.get("semantic_cache", {})
    print(
        f"STARTUP CHECK semantic_cache configured={sc.get('configured_enabled')} "
        f"runtime={sc.get('runtime_enabled')} init_error={sc.get('init_error')}",
        flush=True,
    )

    scm = checks.get("semantic_cache_mongo", {})
    print(
        f"STARTUP CHECK semcache_mongo reachable={scm.get('reachable')} "
        f"reason={scm.get('reason')}",
        flush=True,
    )

    scv = checks.get("semantic_cache_milvus", {})
    print(
        f"STARTUP CHECK semcache_milvus reachable={scv.get('reachable')} "
        f"collection_exists={scv.get('collection_exists')} "
        f"schema_ok={scv.get('schema_ok')} reason={scv.get('reason')}",
        flush=True,
    )

    scs = checks.get("semantic_cache_status", {})
    print(
        f"STARTUP CHECK semcache_usable usable={scs.get('usable')} "
        f"reason={scs.get('reason')}",
        flush=True,
    )

    rag = checks.get("rag", {})
    print(
        f"STARTUP CHECK rag configured={rag.get('configured_enabled')} "
        f"runtime={rag.get('runtime_enabled')} init_error={rag.get('init_error')}",
        flush=True,
    )

    ragm = checks.get("rag_milvus", {})
    print(
        f"STARTUP CHECK rag_milvus reachable={ragm.get('reachable')} "
        f"collection_exists={ragm.get('collection_exists')} reason={ragm.get('reason')}",
        flush=True,
    )

    rroot = checks.get("rag_manifest_root_dir", {})
    print(
        f"STARTUP CHECK rag_manifest_root exists={rroot.get('exists')} "
        f"non_empty={rroot.get('non_empty')} path={rroot.get('path')}",
        flush=True,
    )

    rs = checks.get("rag_status", {})
    print(
        f"STARTUP CHECK rag_usable usable={rs.get('usable')} "
        f"reason={rs.get('reason')}",
        flush=True,
    )

    ready = checks.get("ready_status", {})
    print(
        f"STARTUP CHECK ready ok={ready.get('ok')} reason={ready.get('reason')}",
        flush=True,
    )


def build_runtime(settings: Optional[Settings] = None) -> AppRuntime:
    settings = settings or Settings()
    settings.validate()

    _print_startup_banner(settings)

    vllm = VLLMClient(
        backend=settings.GENERATION_BACKEND,
        direct_vllm_base_url=(
            settings.GENERATION_BASE_URL
            if settings.GENERATION_BACKEND == "direct_vllm"
            else settings.VLLM_BASE_URL
        ),
        llmd_base_url=(
            settings.GENERATION_BASE_URL
            if settings.GENERATION_BACKEND == "llmd"
            else settings.LLMD_BASE_URL
        ),
        llmd_api_mode=settings.GENERATION_API_MODE,
        llmd_chat_path=settings.GENERATION_CHAT_PATH,
        llmd_completions_path=settings.GENERATION_COMPLETIONS_PATH,
        served_model_name=settings.GENERATION_MODEL_NAME,
        timeout_s=settings.MODEL_SERVER_TIMEOUT_S,
    )

    exact_cache = None

    sem_cache = SemanticCache(
        enabled=settings.SEM_CACHE_ENABLED,
        similarity_threshold=settings.SEM_CACHE_THRESHOLD,
        ttl_sec=settings.SEM_CACHE_TTL_SEC,
        embed_model=settings.SEM_CACHE_EMBED_MODEL,
        embed_model_path=settings.SEM_CACHE_EMBED_MODEL_PATH,
        mongo_uri=settings.SEM_CACHE_MONGO_URI,
        mongo_db=settings.SEM_CACHE_MONGO_DB,
        mongo_collection=settings.SEM_CACHE_MONGO_COLLECTION,
        mongo_connect_timeout_ms=settings.SEM_CACHE_MONGO_CONNECT_TIMEOUT_MS,
        milvus_uri=settings.SEM_CACHE_MILVUS_URI,
        milvus_user=settings.SEM_CACHE_MILVUS_USER,
        milvus_password=settings.SEM_CACHE_MILVUS_PASSWORD,
        milvus_secure=settings.SEM_CACHE_MILVUS_SECURE,
        milvus_collection=settings.SEM_CACHE_MILVUS_COLLECTION,
        vector_dim=settings.SEM_CACHE_VECTOR_DIM,
        top_k=settings.SEM_CACHE_TOP_K,
        normalize_embeddings=settings.SEM_CACHE_NORMALIZE,
    )

    rag_router = None
    if settings.RAG_ENABLED:
        rag_router = TenantRAGRouter(
            enabled=settings.RAG_ENABLED,
            backend=settings.RAG_BACKEND,
            top_k=settings.RAG_TOP_K,
            kb_version_fallback=settings.KB_VERSION_FALLBACK,
            manifest_root_dir=settings.RAG_MANIFEST_ROOT_DIR,
            store_root_dir=settings.RAG_STORE_ROOT_DIR,
            embed_model_path_local=settings.RAG_LOCAL_EMBED_MODEL_PATH,
            fallback_tenant=settings.RAG_FALLBACK_TENANT or None,
            milvus_uri=settings.MILVUS_URI,
            milvus_token=settings.MILVUS_TOKEN,
            milvus_collection=settings.MILVUS_COLLECTION,
            milvus_vector_field=settings.MILVUS_VECTOR_FIELD,
            milvus_tenant_field=settings.MILVUS_TENANT_FIELD,
            bge_model_path=settings.BGE_MODEL_PATH,
            bge_device=settings.BGE_DEVICE,
            bge_normalize=settings.BGE_NORMALIZE,
        )

    tokenizer = TokenCounter(
        settings.TOKENIZER_PATH,
        local_files_only=settings.TOKENIZER_LOCAL_ONLY,
    )

    auth_verifier = None
    if settings.AUTH_ENABLED:
        auth_verifier = KeycloakJWTVerifier(
            issuer=settings.KEYCLOAK_ISSUER,
            jwks_url=settings.KEYCLOAK_JWKS_URL,
            audience=settings.KEYCLOAK_AUDIENCE or None,
            tenant_claim=settings.TENANT_CLAIM,
            jwks_cache_ttl_sec=settings.JWKS_CACHE_TTL_SEC,
        )

    history_writer: Optional[MessageHistoryWriter] = None
    if settings.HISTORY_ENABLED:
        try:
            history_writer = MessageHistoryWriter(
                mongo_uri=settings.HISTORY_MONGO_URI,
                mongo_db=settings.HISTORY_MONGO_DB,
                mongo_collection=settings.HISTORY_MONGO_COLLECTION,
            )
            print("[history] MongoDB message history writer initialised", flush=True)
        except Exception as e:
            print(f"[history] WARNING: failed to init history writer: {e}", flush=True)

    orchestrator = ChatOrchestrator(
        vllm=vllm,
        exact_cache=exact_cache,
        semantic_cache=sem_cache,
        rag_router=rag_router,
        base_cache_scope=settings.CACHE_SCOPE,
        served_model_name=settings.GENERATION_MODEL_NAME,
        system_prompt_version=settings.SYSTEM_PROMPT_VERSION,
        kb_version_fallback=settings.KB_VERSION_FALLBACK,
        rag_score_threshold=settings.RAG_SCORE_THRESHOLD,
        rag_score_mode=settings.RAG_SCORE_MODE,
        rag_max_context_chars=settings.RAG_MAX_CONTEXT_CHARS,
        tokenizer=tokenizer,
        exact_cache_ttl_sec=settings.EXACT_CACHE_TTL_SEC,
        sem_cache_allow_with_rag=settings.SEM_CACHE_ALLOW_WITH_RAG,
        default_tenant_id=settings.DEV_TENANT_ID,
        benchmark_shadow_mode=settings.BENCHMARK_SHADOW_MODE,
        rag_retrieve_every_request=settings.RAG_RETRIEVE_EVERY_REQUEST,
        return_debug_blocks=settings.RETURN_DEBUG_BLOCKS,
        history_writer=history_writer,
    )

    startup_checks = _build_startup_checks(settings, sem_cache, rag_router)
    _print_startup_summary(startup_checks)

    print(f"STARTUP RAG_SCORE_THRESHOLD={settings.RAG_SCORE_THRESHOLD}", flush=True)
    print(f"STARTUP SEM_CACHE_ALLOW_WITH_RAG={settings.SEM_CACHE_ALLOW_WITH_RAG}", flush=True)

    return AppRuntime(
        settings=settings,
        vllm=vllm,
        exact_cache=exact_cache,
        semantic_cache=sem_cache,
        rag_router=rag_router,
        auth_verifier=auth_verifier,
        tokenizer=tokenizer,
        orchestrator=orchestrator,
        startup_checks=startup_checks,
    )


async def shutdown_runtime(runtime: AppRuntime) -> None:
    if runtime.semantic_cache is not None:
        runtime.semantic_cache.close()

    if runtime.auth_verifier is not None:
        await runtime.auth_verifier.close()

    await runtime.vllm.close()
