from __future__ import annotations

import copy
import hashlib
import inspect
import json
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

from src.service.observability.tracing import get_tracer
from src.service.observability.viz import record_trace


def _now_ts() -> int:
    return int(time.time())


def _new_response_id() -> str:
    return f"chatcmpl-{uuid.uuid4()}"


def _sha256_json(obj: Any) -> str:
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


class ChatOrchestrator:
    def __init__(
        self,
        *,
        vllm: Any,
        exact_cache: Any,  # kept only for constructor compatibility; not used anymore
        semantic_cache: Any,
        rag_router: Any,
        base_cache_scope: str,
        served_model_name: str,
        system_prompt_version: str,
        kb_version_fallback: str,
        rag_score_threshold: float,
        rag_score_mode: str,
        rag_max_context_chars: int,
        tokenizer: Any,
        exact_cache_ttl_sec: int,  # kept only for constructor compatibility; not used anymore
        sem_cache_allow_with_rag: bool,
        default_tenant_id: str = "tenantA",
        benchmark_shadow_mode: bool = True,
        rag_retrieve_every_request: bool = True,
        return_debug_blocks: bool = True,
        history_writer: Any = None,
    ):
        self.vllm = vllm
        self.exact_cache = None
        self.semantic_cache = semantic_cache
        self.rag_router = rag_router
        self.base_cache_scope = base_cache_scope
        self.served_model_name = served_model_name
        self.system_prompt_version = system_prompt_version
        self.kb_version_fallback = kb_version_fallback
        self.rag_score_threshold = float(rag_score_threshold)
        self.rag_score_mode = str(rag_score_mode or "similarity")
        self.rag_max_context_chars = int(rag_max_context_chars)
        self.tokenizer = tokenizer
        self.exact_cache_ttl_sec = 0
        self.sem_cache_allow_with_rag = bool(sem_cache_allow_with_rag)
        self.default_tenant_id = default_tenant_id
        self.benchmark_shadow_mode = bool(benchmark_shadow_mode)
        self.rag_retrieve_every_request = bool(rag_retrieve_every_request)
        self.return_debug_blocks = bool(return_debug_blocks)
        self.history_writer = history_writer

    async def create_chat_completion(
        self, body: Dict[str, Any], auth_ctx: Any = None
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        tracer = get_tracer()
        request_id = str(uuid.uuid4())
        response_id = _new_response_id()

        payload = copy.deepcopy(body or {})
        model = str(payload.get("model") or self.served_model_name)
        payload["model"] = model
        stream = bool(payload.get("stream", False))
        messages = self._coerce_messages(payload.get("messages"))
        bypass_rag = bool(payload.pop("bypass_rag", False))
        bypass_cache = bool(payload.pop("bypass_cache", False))
        session_id = str(payload.pop("session_id", None) or uuid.uuid4())

        tenant_id = self._resolve_tenant_id(auth_ctx)
        cache_scope = f"{self.base_cache_scope}:{tenant_id}"

        backend_target = str(getattr(self.vllm, "backend", "unknown") or "unknown")
        backend_api_mode = (
            str(getattr(self.vllm, "llmd_api_mode", "chat") or "chat")
            if backend_target == "llmd"
            else "chat"
        )

        exact_enabled = False
        semantic_enabled = bool(
            self.semantic_cache is not None
            and getattr(self.semantic_cache, "enabled", False)
            and not bypass_cache
        )
        semantic_init_error = (
            getattr(self.semantic_cache, "init_error", None)
            if self.semantic_cache is not None
            else None
        )
        rag_enabled = self.rag_router is not None and not bypass_rag

        exact_info: Dict[str, Any] = {
            "enabled": False,
            "consulted": False,
            "hit": False,
            "selected": False,
            "reject_reason": "removed",
        }
        semantic_info: Dict[str, Any] = {
            "enabled": semantic_enabled,
            "consulted": False,
            "hit": False,
            "selected": False,
            "reject_reason": None,
            "shadow_consulted": False,
            "shadow_hit": False,
            "shadow_reject_reason": None,
            "init_error": semantic_init_error,
            "mode": "relaxed_accept_key_v1",
        }
        rag_info: Dict[str, Any] = {
            "enabled": rag_enabled,
            "consulted": False,
            "retrieved": False,
            "used": False,
            "selected": False,
            "skip_reason": None,
            "num_chunks": 0,
            "top_score": 0.0,
            "score_mode": self.rag_score_mode,
            "score_threshold": self.rag_score_threshold,
            "max_context_chars": self.rag_max_context_chars,
            "context_fingerprint": hashlib.sha256(b"no_context").hexdigest(),
            "sources": [],
            "kb_version": self.kb_version_fallback,
        }
        route_info: Dict[str, Any] = {
            "route_taken": None,
            "benchmark_shadow_mode": self.benchmark_shadow_mode,
            "rag_retrieve_every_request": self.rag_retrieve_every_request,
            "backend_target": backend_target,
            "backend_api_mode": backend_api_mode,
            "backend_path": None,
        }
        perf: Dict[str, Any] = {
            "request_id": request_id,
            "stream": stream,
            "model": model,
            "tenant_id": tenant_id,
            "cache_lookup_ms": 0.0,
            "cache_meta_validate_ms": 0.0,
            "cache_embed_ms": 0.0,
            "cache_milvus_ms": 0.0,
            "cache_mongo_ms": 0.0,
            "rag_retrieve_ms": 0.0,
            "rag_embed_ms": 0.0,
            "rag_milvus_ms": 0.0,
            "rag_seaweed_ms": 0.0,
            "rag_format_ms": 0.0,
            "model_backend_http_ms": 0.0,
            "model_backend_json_parse_ms": 0.0,
            "model_backend_http_status": None,
            "cache_write_ms": 0.0,
            "shadow_eval_ms": 0.0,
            "exact_cache_lookup_ms": 0.0,
            "semantic_cache_lookup_ms": 0.0,
            "original_prompt_tokens": self._count_message_tokens(messages),
            "augmented_prompt_tokens": self._count_message_tokens(messages),
        }

        rag_items: List[Dict[str, Any]] = []
        rag_context = ""
        kb_version = self.kb_version_fallback

        if rag_enabled and self.rag_retrieve_every_request:
            rag_info["consulted"] = True
            try:
                rag_engine = await self._resolve_rag_engine(tenant_id)
                if rag_engine is None:
                    rag_info["skip_reason"] = "router_unavailable"
                else:
                    kb_version = str(
                        getattr(rag_engine, "kb_version", self.kb_version_fallback)
                        or self.kb_version_fallback
                    )
                    rag_info["kb_version"] = kb_version

                    query_text = self._semantic_text_from_messages(messages)
                    if not query_text.strip():
                        rag_info["skip_reason"] = "empty_query"
                    else:
                        rt0 = time.perf_counter()
                        with tracer.start_as_current_span("rag.retrieve"):
                            retrieve_result = rag_engine.retrieve(query_text)
                            if inspect.isawaitable(retrieve_result):
                                retrieve_result = await retrieve_result
                        rag_items, rag_meta = retrieve_result
                        perf["rag_retrieve_ms"] = (
                            time.perf_counter() - rt0
                        ) * 1000.0
                        perf["rag_embed_ms"] = float((rag_meta or {}).get("embed_ms", 0.0))
                        perf["rag_milvus_ms"] = float((rag_meta or {}).get("milvus_ms", 0.0))
                        perf["rag_seaweed_ms"] = float((rag_meta or {}).get("seaweed_ms", 0.0))

                        rag_info["retrieved"] = bool(rag_items)
                        rag_info["num_chunks"] = int(
                            (rag_meta or {}).get("num_chunks") or len(rag_items)
                        )
                        rag_info["top_score"] = _safe_float(
                            (rag_meta or {}).get("top_score"), 0.0
                        )
                        rag_info["context_fingerprint"] = str(
                            (rag_meta or {}).get("context_fingerprint")
                            or hashlib.sha256(b"no_context").hexdigest()
                        )

                        if rag_items and self._rag_passes_threshold(
                            rag_info["top_score"]
                        ):
                            ft0 = time.perf_counter()
                            formatted = rag_engine.format_context(rag_items)
                            if inspect.isawaitable(formatted):
                                formatted = await formatted
                            rag_context = str(formatted or "")
                            if len(rag_context) > self.rag_max_context_chars:
                                rag_context = rag_context[: self.rag_max_context_chars]
                            perf["rag_format_ms"] = (
                                time.perf_counter() - ft0
                            ) * 1000.0

                            if rag_context.strip():
                                rag_info["used"] = True
                                rag_info["skip_reason"] = None
                                rag_info["sources"] = [
                                    {
                                        "rank": item.get("rank"),
                                        "score": item.get("score"),
                                        "metadata": item.get("metadata", {}),
                                    }
                                    for item in rag_items
                                ]
                            else:
                                rag_info["skip_reason"] = "empty_context"
                        elif not rag_items:
                            rag_info["skip_reason"] = "no_results"
                        else:
                            rag_info["skip_reason"] = "below_threshold"
            except Exception as e:
                rag_info["skip_reason"] = f"retrieve_error:{type(e).__name__}"
        else:
            rag_info["skip_reason"] = "disabled_or_not_consulted"

        augmented_messages = (
            self._augment_messages_with_context(messages, rag_context)
            if rag_info["used"]
            else copy.deepcopy(messages)
        )
        perf["augmented_prompt_tokens"] = self._count_message_tokens(
            augmented_messages
        )

        semantic_accept_key = self._build_semantic_accept_key(
            tenant_id=tenant_id,
            cache_scope=cache_scope,
            model=model,
            request_payload=payload,
            kb_version=kb_version,
            rag_used=bool(rag_info["used"]),
            context_fingerprint=str(rag_info["context_fingerprint"]),
        )

        cache_text = self._semantic_text_from_messages(messages)
        final_response: Optional[Dict[str, Any]] = None

        semantic_allowed_for_this_request = semantic_enabled and (
            self.sem_cache_allow_with_rag or not rag_info["used"]
        )

        if semantic_enabled and not stream:
            semantic_info["consulted"] = True
            if semantic_allowed_for_this_request:
                if cache_text.strip():
                    st0 = time.perf_counter()
                    with tracer.start_as_current_span("cache.lookup"):
                        sem_payload, sem_reason = self.semantic_cache.get(
                            cache_text, semantic_accept_key
                        )
                    perf["semantic_cache_lookup_ms"] = (
                        time.perf_counter() - st0
                    ) * 1000.0
                    _cache_timings = getattr(self.semantic_cache, "_last_get_timings", {})
                    perf["cache_embed_ms"] = float(_cache_timings.get("embed_ms", 0.0))
                    perf["cache_milvus_ms"] = float(_cache_timings.get("milvus_ms", 0.0))
                    perf["cache_mongo_ms"] = float(_cache_timings.get("mongo_ms", 0.0))
                    if isinstance(sem_payload, dict):
                        semantic_info["hit"] = True
                        semantic_info["selected"] = True
                        route_info["route_taken"] = "semantic_cache"
                        final_response = copy.deepcopy(sem_payload)
                    else:
                        semantic_info["reject_reason"] = sem_reason or "not_found"
                else:
                    semantic_info["reject_reason"] = "empty_query"
            else:
                semantic_info["reject_reason"] = "policy_rag_disabled"

        if final_response is None:
            llm_payload = copy.deepcopy(payload)
            llm_payload["messages"] = augmented_messages

            with tracer.start_as_current_span("vllm.generate"):
                llm_response = await self._call_vllm(llm_payload, request_id=request_id)

            if not isinstance(llm_response, dict):
                raise RuntimeError("Model backend response is not a JSON object")

            perf["model_backend_http_ms"] = _safe_float(llm_response.get("_http_ms"), 0.0)
            perf["model_backend_json_parse_ms"] = _safe_float(llm_response.get("_json_ms"), 0.0)
            perf["model_backend_http_status"] = llm_response.get("_http_status")
            route_info["backend_path"] = llm_response.get("_backend_path")

            final_response = self._strip_runtime_fields(llm_response)
            route_info["route_taken"] = (
                "rag_plus_backend" if rag_info["used"] else "plain_backend"
            )
            rag_info["selected"] = bool(rag_info["used"])

            if not stream:
                wt0 = time.perf_counter()
                cache_payload = self._strip_runtime_fields(final_response)
                if semantic_allowed_for_this_request and semantic_enabled and cache_text.strip():
                    with tracer.start_as_current_span("cache.write"):
                        self.semantic_cache.put(
                            cache_text, cache_payload, semantic_accept_key
                        )
                perf["cache_write_ms"] = (time.perf_counter() - wt0) * 1000.0

        exact_info["selected"] = False
        semantic_info["selected"] = route_info["route_taken"] == "semantic_cache"
        rag_info["selected"] = route_info["route_taken"] == "rag_plus_backend"

        perf["cache_lookup_ms"] = perf["semantic_cache_lookup_ms"]
        perf["e2e_serving_ms"] = (time.perf_counter() - t0) * 1000.0
        perf["e2e_ms"] = perf["e2e_serving_ms"]

        response = self._strip_runtime_fields(final_response)
        response["id"] = response_id
        response["created"] = _now_ts()
        response["model"] = response.get("model") or model

        if self.return_debug_blocks:
            response["_route"] = route_info
            response["_cache"] = {
                "enabled": bool(semantic_enabled),
                "hit": bool(semantic_info["hit"]),
                "miss_reason": None if semantic_info["hit"] else semantic_info["reject_reason"],
                "scope": cache_scope,
                "kb_version": kb_version,
                "tenant_id": tenant_id,
                "exact_enabled": False,
                "semantic_enabled": semantic_enabled,
                "exact_hit": False,
                "semantic_hit": semantic_info["hit"],
                "exact": exact_info,
                "semantic": semantic_info,
            }
            response["_rag"] = rag_info
            response["_perf"] = perf

        response["session_id"] = session_id

        if self.history_writer is not None:
            self.history_writer.write_turn_async(
                session_id=session_id,
                tenant_id=tenant_id,
                user_content=self._semantic_text_from_messages(messages),
                assistant_content=self._extract_answer_preview(response),
                path=route_info.get("route_taken"),
                latency_ms=perf.get("e2e_ms", 0.0),
                cache_hit=bool(semantic_info.get("hit")),
            )

        trace = {
            "request_id": request_id,
            "response_id": response_id,
            "tenant_id": tenant_id,
            "route_taken": route_info["route_taken"],
            "cache": response.get("_cache", {}),
            "rag": response.get("_rag", {}),
            "perf": response.get("_perf", {}),
            "prompt_preview": self._semantic_text_from_messages(messages)[:300],
            "answer_preview": self._extract_answer_preview(response),
        }
        record_trace(trace)

        return response

    async def stream_chat_completion(
        self, body: Dict[str, Any], auth_ctx: Any = None
    ) -> AsyncIterator[bytes]:
        """Streaming variant: yields SSE chunks then a final 'llmsk_meta' metadata event.

        Cache lookup and write are skipped — streaming responses cannot be served from
        or written to the semantic cache without buffering the entire generation.
        """
        t0 = time.perf_counter()
        tracer = get_tracer()
        request_id = str(uuid.uuid4())

        payload = copy.deepcopy(body or {})
        model = str(payload.get("model") or self.served_model_name)
        payload["model"] = model
        payload["stream"] = True
        messages = self._coerce_messages(payload.get("messages"))
        bypass_rag = bool(payload.pop("bypass_rag", False))
        payload.pop("bypass_cache", None)
        session_id = str(payload.pop("session_id", None) or uuid.uuid4())

        tenant_id = self._resolve_tenant_id(auth_ctx)
        cache_scope = f"{self.base_cache_scope}:{tenant_id}"

        backend_target = str(getattr(self.vllm, "backend", "unknown") or "unknown")
        backend_api_mode = (
            str(getattr(self.vllm, "llmd_api_mode", "chat") or "chat")
            if backend_target == "llmd"
            else "chat"
        )

        rag_enabled = self.rag_router is not None and not bypass_rag
        rag_info: Dict[str, Any] = {
            "enabled": rag_enabled,
            "consulted": False,
            "retrieved": False,
            "used": False,
            "selected": False,
            "skip_reason": None,
            "num_chunks": 0,
            "top_score": 0.0,
            "score_mode": self.rag_score_mode,
            "score_threshold": self.rag_score_threshold,
            "max_context_chars": self.rag_max_context_chars,
            "context_fingerprint": hashlib.sha256(b"no_context").hexdigest(),
            "sources": [],
            "kb_version": self.kb_version_fallback,
        }
        route_info: Dict[str, Any] = {
            "route_taken": None,
            "benchmark_shadow_mode": self.benchmark_shadow_mode,
            "rag_retrieve_every_request": self.rag_retrieve_every_request,
            "backend_target": backend_target,
            "backend_api_mode": backend_api_mode,
            "backend_path": None,
        }
        perf: Dict[str, Any] = {
            "request_id": request_id,
            "stream": True,
            "model": model,
            "tenant_id": tenant_id,
            "rag_retrieve_ms": 0.0,
            "rag_embed_ms": 0.0,
            "rag_milvus_ms": 0.0,
            "rag_seaweed_ms": 0.0,
            "rag_format_ms": 0.0,
            "model_backend_http_ms": 0.0,
            "cache_embed_ms": 0.0,
            "cache_milvus_ms": 0.0,
            "cache_mongo_ms": 0.0,
            "semantic_cache_lookup_ms": 0.0,
            "cache_write_ms": 0.0,
            "n_output_tokens": 0,
            "original_prompt_tokens": self._count_message_tokens(messages),
            "augmented_prompt_tokens": self._count_message_tokens(messages),
        }

        rag_items: List[Dict[str, Any]] = []
        rag_context = ""
        kb_version = self.kb_version_fallback

        if rag_enabled and self.rag_retrieve_every_request:
            rag_info["consulted"] = True
            try:
                rag_engine = await self._resolve_rag_engine(tenant_id)
                if rag_engine is None:
                    rag_info["skip_reason"] = "router_unavailable"
                else:
                    kb_version = str(
                        getattr(rag_engine, "kb_version", self.kb_version_fallback)
                        or self.kb_version_fallback
                    )
                    rag_info["kb_version"] = kb_version
                    query_text = self._semantic_text_from_messages(messages)
                    if not query_text.strip():
                        rag_info["skip_reason"] = "empty_query"
                    else:
                        rt0 = time.perf_counter()
                        with tracer.start_as_current_span("rag.retrieve"):
                            retrieve_result = rag_engine.retrieve(query_text)
                            if inspect.isawaitable(retrieve_result):
                                retrieve_result = await retrieve_result
                        rag_items, rag_meta = retrieve_result
                        perf["rag_retrieve_ms"] = (time.perf_counter() - rt0) * 1000.0
                        perf["rag_embed_ms"] = float((rag_meta or {}).get("embed_ms", 0.0))
                        perf["rag_milvus_ms"] = float((rag_meta or {}).get("milvus_ms", 0.0))
                        perf["rag_seaweed_ms"] = float((rag_meta or {}).get("seaweed_ms", 0.0))

                        rag_info["retrieved"] = bool(rag_items)
                        rag_info["num_chunks"] = int(
                            (rag_meta or {}).get("num_chunks") or len(rag_items)
                        )
                        rag_info["top_score"] = _safe_float(
                            (rag_meta or {}).get("top_score"), 0.0
                        )
                        rag_info["context_fingerprint"] = str(
                            (rag_meta or {}).get("context_fingerprint")
                            or hashlib.sha256(b"no_context").hexdigest()
                        )

                        if rag_items and self._rag_passes_threshold(rag_info["top_score"]):
                            ft0 = time.perf_counter()
                            formatted = rag_engine.format_context(rag_items)
                            if inspect.isawaitable(formatted):
                                formatted = await formatted
                            rag_context = str(formatted or "")
                            if len(rag_context) > self.rag_max_context_chars:
                                rag_context = rag_context[: self.rag_max_context_chars]
                            perf["rag_format_ms"] = (time.perf_counter() - ft0) * 1000.0

                            if rag_context.strip():
                                rag_info["used"] = True
                                rag_info["skip_reason"] = None
                                rag_info["sources"] = [
                                    {
                                        "rank": item.get("rank"),
                                        "score": item.get("score"),
                                        "metadata": item.get("metadata", {}),
                                    }
                                    for item in rag_items
                                ]
                            else:
                                rag_info["skip_reason"] = "empty_context"
                        elif not rag_items:
                            rag_info["skip_reason"] = "no_results"
                        else:
                            rag_info["skip_reason"] = "below_threshold"
            except Exception as e:
                rag_info["skip_reason"] = f"retrieve_error:{type(e).__name__}"
        else:
            rag_info["skip_reason"] = "disabled_or_not_consulted"

        augmented_messages = (
            self._augment_messages_with_context(messages, rag_context)
            if rag_info["used"]
            else copy.deepcopy(messages)
        )
        perf["augmented_prompt_tokens"] = self._count_message_tokens(augmented_messages)

        llm_payload = copy.deepcopy(payload)
        llm_payload["messages"] = augmented_messages
        # Inject stream_options here — the ChatRequest schema doesn't declare it so
        # Pydantic drops it from the client body before it reaches the orchestrator.
        # We always want usage.completion_tokens in the stream for accurate TPOT.
        llm_payload["stream_options"] = {"include_usage": True}

        route_info["route_taken"] = "rag_plus_backend" if rag_info["used"] else "plain_backend"
        rag_info["selected"] = bool(rag_info["used"])

        # Stream tokens from the model backend
        stream_fn = getattr(self.vllm, "stream_chat_completions", None)
        if not callable(stream_fn):
            err = b'data: {"error": "streaming not supported by backend"}\n\ndata: [DONE]\n\n'
            yield err
            return

        # n_chunks_with_content counts SSE chunks that carry non-empty content.
        # This is NOT the same as the token count — vLLM may batch multiple tokens
        # per chunk.  The authoritative token count comes from usage.completion_tokens
        # which vLLM sends in the final chunk when stream_options.include_usage=True.
        n_chunks_with_content = 0
        n_output_tokens_usage: Optional[int] = None
        t_llm = time.perf_counter()
        try:
            async for raw_bytes in stream_fn(llm_payload, request_id):
                line = raw_bytes.decode("utf-8", errors="replace").rstrip()
                if not line:
                    continue
                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str != "[DONE]":
                        try:
                            chunk_data = json.loads(data_str)
                            choice = (chunk_data.get("choices") or [{}])[0]
                            content = (choice.get("delta") or {}).get("content") or choice.get("text") or ""
                            if content:
                                n_chunks_with_content += 1
                            usage = chunk_data.get("usage") or {}
                            if usage.get("completion_tokens"):
                                n_output_tokens_usage = int(usage["completion_tokens"])
                        except Exception:
                            pass
                # Forward the SSE line with proper double-newline delimiter
                yield line.encode("utf-8") + b"\n\n"
        except Exception as exc:
            perf["error"] = str(exc)[:200]

        perf["model_backend_http_ms"] = (time.perf_counter() - t_llm) * 1000.0
        perf["e2e_serving_ms"] = (time.perf_counter() - t0) * 1000.0
        perf["n_chunks_with_content"] = n_chunks_with_content
        if n_output_tokens_usage is not None:
            perf["n_output_tokens"] = n_output_tokens_usage

        meta_event = {
            "_perf": perf,
            "_cache": {
                "enabled": False,
                "hit": False,
                "miss_reason": "streaming_not_cached",
                "scope": cache_scope,
                "kb_version": kb_version,
                "tenant_id": tenant_id,
                "semantic_hit": False,
            },
            "_rag": rag_info,
            "_route": route_info,
        }
        yield ("event: llmsk_meta\ndata: " + json.dumps(meta_event) + "\n\n").encode()

    async def chat_completions(
        self, body: Dict[str, Any], auth_ctx: Any = None
    ) -> Dict[str, Any]:
        return await self.create_chat_completion(body, auth_ctx)

    async def handle_chat_completion(
        self, body: Dict[str, Any], auth_ctx: Any = None
    ) -> Dict[str, Any]:
        return await self.create_chat_completion(body, auth_ctx)

    async def handle(
        self, body: Dict[str, Any], auth_ctx: Any = None
    ) -> Dict[str, Any]:
        return await self.create_chat_completion(body, auth_ctx)

    def _resolve_tenant_id(self, auth_ctx: Any) -> str:
        if auth_ctx is None:
            return self.default_tenant_id

        if isinstance(auth_ctx, dict):
            for key in ("tenant_id", "tenant", "tenantId"):
                value = auth_ctx.get(key)
                if value:
                    return str(value)

        for key in ("tenant_id", "tenant", "tenantId"):
            value = getattr(auth_ctx, key, None)
            if value:
                return str(value)

        return self.default_tenant_id

    def _coerce_messages(self, messages: Any) -> List[Dict[str, Any]]:
        if not isinstance(messages, list):
            return []
        out: List[Dict[str, Any]] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "user")
            content = msg.get("content")
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(str(part.get("text") or ""))
                content = "\n".join(text_parts)
            out.append({"role": role, "content": str(content or "")})
        return out

    def _semantic_text_from_messages(self, messages: List[Dict[str, Any]]) -> str:
        user_parts = [
            str(m.get("content") or "").strip()
            for m in messages
            if str(m.get("role")) == "user"
        ]
        text = "\n".join([p for p in user_parts if p])
        if text.strip():
            return text
        return "\n".join(
            str(m.get("content") or "").strip()
            for m in messages
            if str(m.get("content") or "").strip()
        )

    def _augment_messages_with_context(
        self, messages: List[Dict[str, Any]], context: str
    ) -> List[Dict[str, Any]]:
        if not context.strip():
            return copy.deepcopy(messages)

        rag_system = {
            "role": "system",
            "content": (
                "Use the following tenant knowledge-base context when answering. "
                "Prefer the context when it is relevant. "
                "If the user asks about the document or knowledge base, ground your answer in the context.\n\n"
                f"<kb_context>\n{context}\n</kb_context>"
            ),
        }
        return [rag_system] + copy.deepcopy(messages)

    def _build_semantic_accept_key(
        self,
        *,
        tenant_id: str,
        cache_scope: str,
        model: str,
        request_payload: Dict[str, Any],
        kb_version: str,
        rag_used: bool,
        context_fingerprint: str,
    ) -> str:
        key_obj = {
            "key_version": "semantic_accept_v1",
            "cache_scope": cache_scope,
            "tenant_id": tenant_id,
            "model": model,
            "system_prompt_version": self.system_prompt_version,
            "kb_version": kb_version,
            "rag": {
                "used": bool(rag_used),
                "context_fingerprint": context_fingerprint if rag_used else "no_context",
            },
            "generation": {
                "max_tokens": request_payload.get("max_tokens"),
                "temperature": request_payload.get("temperature"),
                "top_p": request_payload.get("top_p"),
                "response_format": request_payload.get("response_format"),
                "stop": request_payload.get("stop"),
            },
        }
        return _sha256_json(key_obj)

    def _rag_passes_threshold(self, top_score: float) -> bool:
        if self.rag_score_mode == "distance":
            return top_score <= self.rag_score_threshold
        return top_score >= self.rag_score_threshold

    async def _resolve_rag_engine(self, tenant_id: str) -> Any:
        router = self.rag_router
        if router is None:
            return None

        for name in ("get_for_tenant", "get_engine", "for_tenant", "get", "route"):
            fn = getattr(router, name, None)
            if callable(fn):
                result = fn(tenant_id)
                if inspect.isawaitable(result):
                    result = await result
                if result is not None:
                    return result

        return router

    async def _call_vllm(
        self, llm_payload: Dict[str, Any], request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        client = self.vllm
        if client is None:
            raise RuntimeError("vLLM client is not configured")

        candidate_names = (
            "create_chat_completion",
            "chat_completions",
            "handle_chat_completion",
            "complete_chat",
            "complete",
            "__call__",
        )

        for name in candidate_names:
            fn = getattr(client, name, None)
            if not callable(fn):
                continue

            try:
                sig = inspect.signature(fn)
                kwargs: Dict[str, Any] = {}
                if "request_id" in sig.parameters and request_id is not None:
                    kwargs["request_id"] = request_id
                result = fn(llm_payload, **kwargs)
            except TypeError:
                result = fn(llm_payload)

            if inspect.isawaitable(result):
                result = await result

            if isinstance(result, dict):
                return result

        raise RuntimeError("Unable to coerce model backend response to dict")

    def _strip_runtime_fields(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        out = copy.deepcopy(payload)
        for key in ("_cache", "_rag", "_perf", "_route"):
            out.pop(key, None)
        out.pop("id", None)
        out.pop("created", None)
        return out

    def _extract_answer_preview(self, response: Dict[str, Any]) -> str:
        try:
            return str(response["choices"][0]["message"]["content"])[:300]
        except Exception:
            return ""

    def _count_message_tokens(self, messages: List[Dict[str, Any]]) -> int:
        if self.tokenizer is None:
            return 0

        for name in ("count_messages", "count_messages_tokens", "num_tokens_from_messages"):
            fn = getattr(self.tokenizer, name, None)
            if callable(fn):
                try:
                    return int(fn(messages))
                except Exception:
                    pass

        text = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in messages)
        for name in ("count_text", "count_tokens", "num_tokens"):
            fn = getattr(self.tokenizer, name, None)
            if callable(fn):
                try:
                    return int(fn(text))
                except Exception:
                    pass

        return 0
