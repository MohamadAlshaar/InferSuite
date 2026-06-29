import time
from typing import List

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from src.service.api.schemas import ChatRequest
from src.service.auth.deps import get_auth_context
from src.service.auth.types import AuthContext
from src.service.bootstrap import refresh_startup_checks

router = APIRouter()


def _body_to_dict(body: ChatRequest) -> dict:
    if hasattr(body, "model_dump"):
        return body.model_dump()
    if hasattr(body, "dict"):
        return body.dict()
    return dict(body)


def _current_checks(request: Request) -> dict:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        return getattr(request.app.state, "startup_checks", {}) or {}

    checks = refresh_startup_checks(runtime)
    request.app.state.startup_checks = checks
    return checks


@router.get("/health")
async def health(request: Request):
    s = request.app.state.settings
    sem_cache = getattr(request.app.state, "semantic_cache", None)
    rag_router = getattr(request.app.state, "rag_router", None)
    startup_checks = _current_checks(request)

    semcache_mongo = startup_checks.get("semantic_cache_mongo", {})
    semcache_milvus = startup_checks.get("semantic_cache_milvus", {})
    rag_milvus = startup_checks.get("rag_milvus", {})
    rag_manifest_root = startup_checks.get("rag_manifest_root_dir", {})
    rag_seed_manifest = startup_checks.get("rag_seed_manifest_dir", {})
    generation_backend = startup_checks.get("generation_backend", {})
    generation_status = startup_checks.get("generation_status", {})
    semantic_status = startup_checks.get("semantic_cache_status", {})
    rag_status = startup_checks.get("rag_status", {})
    ready_status = startup_checks.get("ready_status", {})

    return {
        "ok": True,
        "ready": bool(ready_status.get("ok")),
        "ready_reason": ready_status.get("reason"),
        "model_backend": s.MODEL_BACKEND,
        "model_base_url": s.model_base_url,
        "model": s.SERVED_MODEL_NAME,
        "auth_enabled": s.AUTH_ENABLED,
        "tenant_claim": s.TENANT_CLAIM,
        "exact_cache_enabled": s.EXACT_CACHE_ENABLED,
        "semantic_cache_enabled": s.SEM_CACHE_ENABLED,
        "semantic_cache_runtime_enabled": bool(
            sem_cache is not None and getattr(sem_cache, "enabled", False)
        ),
        "semantic_cache_init_error": getattr(sem_cache, "init_error", None)
        if sem_cache is not None
        else None,
        "semantic_cache_usable": bool(semantic_status.get("usable")),
        "semantic_cache_reason": semantic_status.get("reason"),
        "semantic_cache_mongo_reachable": semcache_mongo.get("reachable"),
        "semantic_cache_milvus_reachable": semcache_milvus.get("reachable"),
        "semantic_cache_collection_exists": semcache_milvus.get("collection_exists"),
        "semantic_cache_schema_ok": semcache_milvus.get("schema_ok"),
        "semantic_cache_collection": s.SEM_CACHE_MILVUS_COLLECTION,
        "rag_enabled": s.RAG_ENABLED,
        "rag_runtime_enabled": bool(rag_router is not None and getattr(rag_router, "enabled", False)),
        "rag_init_error": getattr(rag_router, "init_error", None) if rag_router is not None else None,
        "rag_usable": bool(rag_status.get("usable")),
        "rag_reason": rag_status.get("reason"),
        "rag_store_root_dir": s.RAG_STORE_ROOT_DIR,
        "rag_manifest_root_dir": s.RAG_MANIFEST_ROOT_DIR,
        "rag_manifest_root_exists": rag_manifest_root.get("exists"),
        "rag_manifest_root_non_empty": rag_manifest_root.get("non_empty"),
        "rag_seed_manifest_dir": rag_seed_manifest.get("path"),
        "rag_seed_manifest_exists": rag_seed_manifest.get("exists"),
        "rag_seed_manifest_non_empty": rag_seed_manifest.get("non_empty"),
        "rag_milvus_reachable": rag_milvus.get("reachable"),
        "rag_collection_exists": rag_milvus.get("collection_exists"),
        "rag_collection": s.MILVUS_COLLECTION,
        "rag_score_threshold": s.RAG_SCORE_THRESHOLD,
        "benchmark_shadow_mode": s.BENCHMARK_SHADOW_MODE,
        "rag_retrieve_every_request": s.RAG_RETRIEVE_EVERY_REQUEST,
        "cache_scope": s.CACHE_SCOPE,
        "dev_tenant_id": s.DEV_TENANT_ID,
        "generation_backend_reachable": generation_backend.get("reachable"),
        "generation_backend_status_code": generation_backend.get("status_code"),
        "generation_backend_reason": generation_backend.get("reason"),
        "generation_backend_usable": bool(generation_status.get("usable")),
        "generation_backend_usable_reason": generation_status.get("reason"),
        "startup_checks": startup_checks,
    }


@router.get("/ready")
async def ready(request: Request):
    startup_checks = _current_checks(request)
    ready_status = startup_checks.get("ready_status", {}) or {}
    status_code = 200 if bool(ready_status.get("ok")) else 503

    payload = {
        "ok": bool(ready_status.get("ok")),
        "reason": ready_status.get("reason"),
        "generation_backend_usable": bool(
            (startup_checks.get("generation_status", {}) or {}).get("usable")
        ),
        "semantic_cache_usable": bool(
            (startup_checks.get("semantic_cache_status", {}) or {}).get("usable")
        ),
        "rag_usable": bool(
            (startup_checks.get("rag_status", {}) or {}).get("usable")
        ),
    }
    return JSONResponse(status_code=status_code, content=payload)


@router.get("/v1/models")
async def list_models(request: Request):
    s = request.app.state.settings
    return {
        "object": "list",
        "data": [
            {
                "id": s.SERVED_MODEL_NAME,
                "object": "model",
                "owned_by": "local",
            }
        ],
    }


@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    body: ChatRequest,
    auth: AuthContext = Depends(get_auth_context),
):
    orch = request.app.state.orchestrator
    body_dict = _body_to_dict(body)
    if body_dict.get("stream"):
        return StreamingResponse(
            orch.stream_chat_completion(body_dict, auth),
            media_type="text/event-stream",
        )
    return await orch.handle_chat_completion(body_dict, auth)


@router.post("/chat")
async def chat_alias(
    request: Request,
    body: ChatRequest,
    auth: AuthContext = Depends(get_auth_context),
):
    orch = request.app.state.orchestrator
    body_dict = _body_to_dict(body)
    if body_dict.get("stream"):
        return StreamingResponse(
            orch.stream_chat_completion(body_dict, auth),
            media_type="text/event-stream",
        )
    return await orch.handle_chat_completion(body_dict, auth)


# ── Isolated microbenchmark endpoints ────────────────────────────────────────
# These run BGE embedding and HNSW search inside the uvicorn process so that
# perf stat -p <uvicorn_pid> captures the actual work, unlike kubectl exec
# which runs a separate Python process that perf cannot see.

class _IsolatedEmbedRequest(BaseModel):
    texts: List[str]
    tenant_id: str = "tenantA"


class _IsolatedSearchRequest(BaseModel):
    texts: List[str]
    tenant_id: str = "tenantA"
    top_k: int = 4


@router.post("/v1/isolated/embed")
async def isolated_embed(request: Request, body: _IsolatedEmbedRequest):
    """BGE-only microbenchmark: embed each text and return per-text timing.
    Runs inside uvicorn so perf -p captures the AVX-512 FP work correctly."""
    rag_router = getattr(request.app.state, "rag_router", None)
    if rag_router is None or not getattr(rag_router, "_embedder", None):
        return JSONResponse(status_code=503, content={"error": "BGE embedder not available"})

    embedder = rag_router._embedder
    results = []
    for text in body.texts:
        t0 = time.perf_counter()
        embedder.embed_query(text)
        ms = round((time.perf_counter() - t0) * 1000, 2)
        results.append({"text_words": len(text.split()), "embed_ms": ms})

    return {"results": results, "n": len(results)}


@router.post("/v1/isolated/search")
async def isolated_search(request: Request, body: _IsolatedSearchRequest):
    """HNSW-only microbenchmark: embed then search Milvus, return per-search timing.
    Embed time is NOT included in hnsw_ms — only the Milvus search call is timed."""
    rag_router = getattr(request.app.state, "rag_router", None)
    if (rag_router is None
            or not getattr(rag_router, "_embedder", None)
            or not getattr(rag_router, "_milvus", None)):
        return JSONResponse(status_code=503, content={"error": "RAG (Milvus + embedder) not available"})

    embedder   = rag_router._embedder
    milvus     = rag_router._milvus
    collection = rag_router._milvus_collection
    t_field    = rag_router._milvus_tenant_field

    results = []
    for text in body.texts:
        vec = embedder.embed_query(text)          # embed cost not counted in hnsw_ms
        t0  = time.perf_counter()
        hits = milvus.search(
            collection_name=collection,
            data=[vec],
            limit=body.top_k,
            output_fields=["chunk_id"],
            filter=f'{t_field} == "{body.tenant_id}"',
        )
        ms = round((time.perf_counter() - t0) * 1000, 2)
        results.append({
            "text_words": len(text.split()),
            "hnsw_ms":    ms,
            "num_hits":   len(hits[0]) if hits else 0,
        })

    return {"results": results, "n": len(results)}
