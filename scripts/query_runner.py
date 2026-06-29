#!/usr/bin/env python3
"""
Benchmark query runner for GenAI workload characterization.

Modes:
  rag          - RAG path (bypass_cache=True). Full pipeline: BGE→Milvus→SeaweedFS→LLM.
  sc_a         - SC isolated (bypass_rag=True, --sc-scenario a). Measures pure SC overhead.
  sc_b         - SC full pipeline (bypass_rag=False, --sc-scenario b). RAG runs first,
                 then SC checks against the RAG-aware accept_key. Matches production flow.
  llm_direct   - LLM-direct (bypass_rag=True, bypass_cache=True). No retrieval, no cache.
  bge_isolated - BGE-only microbenchmark via POST /v1/isolated/embed inside uvicorn so
                 perf stat -p <uvicorn_pid> captures the actual AVX-512 FP work.
  hnsw_isolated- HNSW-only microbenchmark via POST /v1/isolated/search inside uvicorn;
                 embed cost is excluded server-side, only Milvus search is timed.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

BASE_URL   = os.getenv("BENCHMARK_URL",   "http://localhost:8080")
MODEL_NAME = os.getenv("BENCHMARK_MODEL", "qwen2.5-0.5b")

_session = requests.Session()
TIMEOUT    = int(os.getenv("BENCHMARK_TIMEOUT", "300"))
NAMESPACE  = os.getenv("BENCHMARK_NAMESPACE", "llm-service")

RESULTS_DIR = Path(__file__).parent.parent / "benchmark_results"

CSV_FIELDS = [
    "timestamp", "request_id", "mode", "size_bucket", "concurrency",
    "query_words", "route",
    "cache_hit", "cache_miss_reason",
    "e2e_ms", "frontend_overhead_ms",
    # Streaming-only timing fields (0 for non-streaming and cache-hit rows):
    #   ttft_ms              — t_first_sse_chunk - t_request_start (real TTFT)
    #   generation_ms        — t_final_chunk - t_first_sse_chunk
    #   stream_inter_chunk_ms— generation_ms / (n_chunks_with_content - 1); mean time between
    #                          SSE chunks; NOT the same as per-token time unless chunk==token
    #   tpot_ms              — generation_ms / (n_output_tokens - 1) using usage.completion_tokens;
    #                          true TPOT only when vLLM usage is present; else 0
    #   n_chunks_with_content— number of SSE chunks that carried non-empty content
    "ttft_ms", "generation_ms", "stream_inter_chunk_ms", "tpot_ms", "n_chunks_with_content",
    "rag_embed_ms", "rag_milvus_ms", "rag_seaweed_ms", "rag_retrieve_ms", "rag_format_ms",
    "rag_num_chunks", "rag_top_score",
    "cache_embed_ms", "cache_milvus_ms", "cache_mongo_ms", "semantic_cache_lookup_ms",
    "model_backend_http_ms", "cache_write_ms",
    "original_prompt_tokens", "augmented_prompt_tokens", "n_output_tokens",
    "max_tokens", "bypass_rag", "bypass_cache",
    "http_status", "error",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _f(s) -> float:
    """Safe float: returns 0.0 for empty string, None, or non-numeric values."""
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def _pct(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    return round(s[min(int(len(s) * p / 100), len(s) - 1)], 2)


def _mean(data: List[float]) -> float:
    return round(statistics.mean(data), 2) if data else 0.0


def send_query(
    query: str,
    mode: str,
    size_bucket: str,
    bypass_rag: bool = False,
    bypass_cache: bool = False,
    max_tokens: int = 64,
    concurrency: int = 1,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {f: "" for f in CSV_FIELDS}
    row.update({
        "timestamp": _now_iso(),
        "request_id": str(uuid.uuid4()),
        "mode": mode,
        "size_bucket": size_bucket,
        "concurrency": concurrency,
        "query_words": len(query.split()),
        "route": "",
        "cache_hit": False,
        "cache_miss_reason": "",
        "e2e_ms": 0.0, "frontend_overhead_ms": 0.0,
        # Streaming-only fields: always empty for non-streaming rows (including cache hits).
        # Do not compare 0 vs a real measurement — use "" to mean "not applicable".
        "ttft_ms": "", "generation_ms": "",
        "stream_inter_chunk_ms": "", "tpot_ms": "", "n_chunks_with_content": "",
        "rag_embed_ms": 0.0, "rag_milvus_ms": 0.0, "rag_seaweed_ms": 0.0,
        "rag_retrieve_ms": 0.0, "rag_format_ms": 0.0,
        "rag_num_chunks": 0, "rag_top_score": 0.0,
        "cache_embed_ms": 0.0, "cache_milvus_ms": 0.0, "cache_mongo_ms": 0.0,
        "semantic_cache_lookup_ms": 0.0,
        "model_backend_http_ms": 0.0, "cache_write_ms": 0.0,
        "original_prompt_tokens": 0, "augmented_prompt_tokens": 0,
        "n_output_tokens": 0, "max_tokens": max_tokens,
        "bypass_rag": bypass_rag, "bypass_cache": bypass_cache,
        "http_status": 0, "error": "",
    })

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": query}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    if bypass_rag:
        payload["bypass_rag"] = True
    if bypass_cache:
        payload["bypass_cache"] = True

    t0 = time.perf_counter()
    try:
        resp = _session.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            timeout=TIMEOUT,
        )
        row["e2e_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        row["http_status"] = resp.status_code

        if resp.status_code != 200:
            row["error"] = f"http_{resp.status_code}"
            try:
                err_body = resp.json()
                if isinstance(err_body.get("error"), dict):
                    row["error"] = err_body["error"].get("message", row["error"])[:120]
            except Exception:
                pass
            return row

        body = resp.json()
        if body.get("error"):
            err = body["error"]
            row["error"] = (err.get("message", str(err)) if isinstance(err, dict) else str(err))[:120]
            return row

        perf  = body.get("_perf", {}) or {}
        cache = body.get("_cache", {}) or {}
        rag   = body.get("_rag", {}) or {}
        route = body.get("_route", {}) or {}

        row["route"]               = str(route.get("route_taken") or "")
        row["cache_hit"]           = bool(cache.get("semantic_hit", False))
        row["cache_miss_reason"]   = str(cache.get("miss_reason") or "")
        row["rag_embed_ms"]        = float(perf.get("rag_embed_ms", 0.0))
        row["rag_milvus_ms"]       = float(perf.get("rag_milvus_ms", 0.0))
        row["rag_seaweed_ms"]      = float(perf.get("rag_seaweed_ms", 0.0))
        row["rag_retrieve_ms"]     = float(perf.get("rag_retrieve_ms", 0.0))
        row["rag_format_ms"]       = float(perf.get("rag_format_ms", 0.0))
        row["rag_num_chunks"]      = int(rag.get("num_chunks", 0))
        row["rag_top_score"]       = float(rag.get("top_score", 0.0))
        row["cache_embed_ms"]      = float(perf.get("cache_embed_ms", 0.0))
        row["cache_milvus_ms"]     = float(perf.get("cache_milvus_ms", 0.0))
        row["cache_mongo_ms"]      = float(perf.get("cache_mongo_ms", 0.0))
        row["semantic_cache_lookup_ms"] = float(perf.get("semantic_cache_lookup_ms", 0.0))
        row["model_backend_http_ms"]    = float(perf.get("model_backend_http_ms", 0.0))
        row["cache_write_ms"]      = float(perf.get("cache_write_ms", 0.0))
        row["original_prompt_tokens"]   = int(perf.get("original_prompt_tokens", 0))
        row["augmented_prompt_tokens"]  = int(perf.get("augmented_prompt_tokens", 0))
        row["frontend_overhead_ms"] = round(row["e2e_ms"] - row["model_backend_http_ms"], 2)
        usage = body.get("usage") or {}
        row["n_output_tokens"]     = int(usage.get("completion_tokens") or 0)

    except requests.exceptions.Timeout:
        row["e2e_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        row["error"] = "timeout"
    except Exception as exc:
        row["e2e_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        row["error"] = str(exc)[:200]

    return row


def send_query_streaming(
    query: str,
    mode: str,
    size_bucket: str,
    bypass_rag: bool = False,
    bypass_cache: bool = False,
    max_tokens: int = 64,
    concurrency: int = 1,
) -> Dict[str, Any]:
    """Streaming variant of send_query.

    Uses requests.post(..., stream=True) and parses SSE chunks.

    Timing:
      t_request_start  = time.perf_counter() before POST
      t_first_sse_chunk= time of first SSE chunk with non-empty content
      t_final_chunk    = time of last SSE chunk with non-empty content

      ttft_ms              = t_first_sse_chunk - t_request_start  (REAL TTFT)
      generation_ms        = t_final_chunk    - t_first_sse_chunk

    Throughput (two distinct fields):
      stream_inter_chunk_ms= generation_ms / (n_chunks_with_content - 1)
                             Mean time between SSE chunks.  NOT per-token unless
                             vLLM happens to send exactly one token per chunk.
      tpot_ms              = generation_ms / (n_output_tokens - 1)
                             TRUE per-token time — computed only when vLLM sends
                             usage.completion_tokens in the stream (stream_options
                             include_usage=True).  Left 0 when not available.

    Server-side perf/rag/route fields come from the 'llmsk_meta' SSE event that
    the orchestrator injects after [DONE].  n_output_tokens from llmsk_meta is
    the orchestrator's chunk count (less accurate); prefer usage.completion_tokens.

    Cache-hit rows: streaming bypasses the semantic cache, so cache_hit is always
    False for streaming requests.  ttft_ms / tpot_ms are 0 for non-streaming rows.
    """
    row: Dict[str, Any] = {f: "" for f in CSV_FIELDS}
    row.update({
        "timestamp": _now_iso(),
        "request_id": str(uuid.uuid4()),
        "mode": mode,
        "size_bucket": size_bucket,
        "concurrency": concurrency,
        "query_words": len(query.split()),
        "route": "",
        "cache_hit": False,
        "cache_miss_reason": "",
        "e2e_ms": 0.0, "frontend_overhead_ms": 0.0,
        # Initialized to "" — filled in only when the measurement is made.
        # Left as "" on error so 0 is never confused with a real 0 ms measurement.
        "ttft_ms": "", "generation_ms": "",
        "stream_inter_chunk_ms": "", "tpot_ms": "", "n_chunks_with_content": "",
        "rag_embed_ms": 0.0, "rag_milvus_ms": 0.0, "rag_seaweed_ms": 0.0,
        "rag_retrieve_ms": 0.0, "rag_format_ms": 0.0,
        "rag_num_chunks": 0, "rag_top_score": 0.0,
        "cache_embed_ms": 0.0, "cache_milvus_ms": 0.0, "cache_mongo_ms": 0.0,
        "semantic_cache_lookup_ms": 0.0,
        "model_backend_http_ms": 0.0, "cache_write_ms": 0.0,
        "original_prompt_tokens": 0, "augmented_prompt_tokens": 0,
        "n_output_tokens": 0, "max_tokens": max_tokens,
        "bypass_rag": bypass_rag, "bypass_cache": bypass_cache,
        "http_status": 0, "error": "",
    })

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": query}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if bypass_rag:
        payload["bypass_rag"] = True
    if bypass_cache:
        payload["bypass_cache"] = True

    t_request_start = time.perf_counter()
    t_first_sse_chunk: Optional[float] = None
    t_final_chunk: Optional[float] = None
    n_chunks_with_content = 0
    n_output_tokens_from_usage: Optional[int] = None
    pending_event_type: Optional[str] = None

    try:
        resp = _session.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            timeout=TIMEOUT,
            stream=True,
        )
        row["http_status"] = resp.status_code
        if resp.status_code != 200:
            row["error"] = f"http_{resp.status_code}"
            row["e2e_ms"] = round((time.perf_counter() - t_request_start) * 1000.0, 2)
            return row

        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line:
                pending_event_type = None
                continue

            if raw_line.startswith("event:"):
                pending_event_type = raw_line[6:].strip()
                continue

            if not raw_line.startswith("data:"):
                continue

            data_str = raw_line[5:].strip()

            if data_str == "[DONE]":
                # Don't break — the orchestrator's llmsk_meta event follows [DONE]
                continue

            if pending_event_type == "llmsk_meta":
                try:
                    meta = json.loads(data_str)
                    perf  = meta.get("_perf",  {}) or {}
                    cache = meta.get("_cache", {}) or {}
                    rag   = meta.get("_rag",   {}) or {}
                    route = meta.get("_route", {}) or {}
                    row["route"]               = str(route.get("route_taken") or "")
                    row["cache_hit"]           = bool(cache.get("semantic_hit", False))
                    row["cache_miss_reason"]   = str(cache.get("miss_reason") or "")
                    row["rag_embed_ms"]        = float(perf.get("rag_embed_ms", 0.0))
                    row["rag_milvus_ms"]       = float(perf.get("rag_milvus_ms", 0.0))
                    row["rag_seaweed_ms"]      = float(perf.get("rag_seaweed_ms", 0.0))
                    row["rag_retrieve_ms"]     = float(perf.get("rag_retrieve_ms", 0.0))
                    row["rag_format_ms"]       = float(perf.get("rag_format_ms", 0.0))
                    row["rag_num_chunks"]      = int(rag.get("num_chunks", 0))
                    row["rag_top_score"]       = float(rag.get("top_score", 0.0))
                    row["model_backend_http_ms"] = float(perf.get("model_backend_http_ms", 0.0))
                    row["original_prompt_tokens"]  = int(perf.get("original_prompt_tokens", 0))
                    row["augmented_prompt_tokens"] = int(perf.get("augmented_prompt_tokens", 0))
                    # n_chunks_with_content from server is the orchestrator's chunk count;
                    # prefer n_output_tokens_from_usage (vLLM usage) for tpot_ms denominator.
                except Exception:
                    pass
                pending_event_type = None
                continue

            # Standard token chunk — handle both chat (delta.content) and
            # completions (choices[0].text) formats from vLLM.
            try:
                chunk = json.loads(data_str)
                choice = (chunk.get("choices") or [{}])[0]
                # Chat completions format: delta.content
                # Text completions format: text
                content = (choice.get("delta") or {}).get("content") or choice.get("text") or ""
                if content:
                    now = time.perf_counter()
                    if t_first_sse_chunk is None:
                        t_first_sse_chunk = now
                        row["ttft_ms"] = round((now - t_request_start) * 1000.0, 2)
                    t_final_chunk = now
                    n_chunks_with_content += 1
                # vLLM sends usage.completion_tokens in the last chunk when
                # stream_options.include_usage=True — this is the authoritative token count.
                usage = chunk.get("usage") or {}
                if usage.get("completion_tokens"):
                    n_output_tokens_from_usage = int(usage["completion_tokens"])
            except Exception:
                pass

        row["e2e_ms"] = round((time.perf_counter() - t_request_start) * 1000.0, 2)

        # Only populate streaming metrics when we actually received content chunks.
        # Leave as "" when no content arrived (error, empty response, etc.) so that
        # 0 is never confused with a real measurement.
        if t_first_sse_chunk and t_final_chunk:
            gen_ms = round((t_final_chunk - t_first_sse_chunk) * 1000.0, 2)
            row["generation_ms"] = gen_ms
        else:
            gen_ms = None  # no content received

        if n_chunks_with_content > 0:
            row["n_chunks_with_content"] = n_chunks_with_content

        if gen_ms is not None and gen_ms > 0:
            if n_chunks_with_content > 1:
                row["stream_inter_chunk_ms"] = round(
                    gen_ms / (n_chunks_with_content - 1), 2
                )

        if n_output_tokens_from_usage is not None:
            row["n_output_tokens"] = n_output_tokens_from_usage
            if gen_ms is not None and gen_ms > 0 and n_output_tokens_from_usage > 1:
                row["tpot_ms"] = round(gen_ms / (n_output_tokens_from_usage - 1), 2)

        row["frontend_overhead_ms"] = round(
            row["e2e_ms"] - row["model_backend_http_ms"], 2
        )

    except requests.exceptions.Timeout:
        row["e2e_ms"] = round((time.perf_counter() - t_request_start) * 1000.0, 2)
        row["error"] = "timeout"
    except Exception as exc:
        row["e2e_ms"] = round((time.perf_counter() - t_request_start) * 1000.0, 2)
        row["error"] = str(exc)[:200]

    return row


def run_phase(
    queries: List[str],
    mode: str,
    size_bucket: str,
    writer: Optional[csv.DictWriter],
    bypass_rag: bool = False,
    bypass_cache: bool = False,
    max_tokens: int = 64,
    concurrency: int = 1,
    silent: bool = False,
    streaming: bool = False,
) -> List[Dict[str, Any]]:
    _send = send_query_streaming if streaming else send_query
    results: List[Dict[str, Any]] = []
    batches = [queries[i:i + concurrency] for i in range(0, len(queries), concurrency)]
    done = 0
    for batch in batches:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(_send, q, mode, size_bucket,
                            bypass_rag, bypass_cache, max_tokens, concurrency): q
                for q in batch
            }
            for fut in as_completed(futures):
                row = fut.result()
                if writer and not silent:
                    writer.writerow(row)
                results.append(row)
                done += 1
                if not silent and done % 50 == 0:
                    ok_so_far = [r for r in results if not r["error"]]
                    e2e = [r["e2e_ms"] for r in ok_so_far]
                    print(f"  [{mode}/{size_bucket}] {done}/{len(queries)} "
                          f"p50={_pct(e2e,50):.0f}ms errors={len(results)-len(ok_so_far)}",
                          flush=True)
    return results


def print_summary(results: List[Dict[str, Any]], title: str) -> None:
    ok     = [r for r in results if not r["error"]]
    errors = [r for r in results if r["error"]]
    # Use _f() everywhere so "" (non-streaming fields) safely becomes 0.0
    e2e    = [_f(r["e2e_ms"]) for r in ok]
    fe     = [_f(r["frontend_overhead_ms"]) for r in ok]
    gpu    = [_f(r["model_backend_http_ms"]) for r in ok if _f(r["model_backend_http_ms"]) > 0]
    tpot   = [_f(r["model_backend_http_ms"]) / _f(r["n_output_tokens"])
               for r in ok if _f(r["n_output_tokens"]) > 0 and _f(r["model_backend_http_ms"]) > 0]
    rag_embed  = [_f(r["rag_embed_ms"])   for r in ok if _f(r["rag_embed_ms"]) > 0]
    rag_milvus = [_f(r["rag_milvus_ms"])  for r in ok if _f(r["rag_milvus_ms"]) > 0]
    rag_sw     = [_f(r["rag_seaweed_ms"]) for r in ok if _f(r["rag_seaweed_ms"]) > 0]
    cache_hits = sum(1 for r in ok if r["cache_hit"])
    rag_hits   = sum(1 for r in ok if _f(r["rag_num_chunks"]) > 0)
    routes: Dict[str, int] = {}
    for r in ok:
        routes[r["route"]] = routes.get(r["route"], 0) + 1

    # Streaming fields are "" for non-streaming rows — use truthiness to filter
    ttft     = [_f(r["ttft_ms"])               for r in ok if r.get("ttft_ms")]
    gen_ms   = [_f(r["generation_ms"])         for r in ok if r.get("generation_ms")]
    ichunk   = [_f(r["stream_inter_chunk_ms"]) for r in ok if r.get("stream_inter_chunk_ms")]
    tpot_s   = [_f(r["tpot_ms"])              for r in ok if r.get("tpot_ms")]

    w = 65
    print(f"\n{'─' * w}")
    print(f"  {title}")
    print(f"{'─' * w}")
    small_n = len(ok) < 20
    print(f"  n={len(results)}  ok={len(ok)}  errors={len(errors)}")
    if small_n:
        print(f"  {'metric':<30} {'min':>8} {'median':>8} {'max':>8} {'mean':>8}")
    else:
        print(f"  {'metric':<30} {'p50':>8} {'p95':>8} {'p99':>8} {'mean':>8}")
    print(f"  {'─' * 58}")
    for label, data in [
        ("e2e_ms", e2e), ("frontend_overhead_ms", fe),
        ("ttft_ms", ttft), ("generation_ms", gen_ms),
        ("stream_inter_chunk_ms (not per-tok)", ichunk),
        ("tpot_ms (true, from usage)", tpot_s),
        ("model_backend_ms", gpu), ("backend_ms_per_output_token", tpot),
        ("rag_embed_ms", rag_embed), ("rag_milvus_ms", rag_milvus),
        ("rag_seaweed_ms", rag_sw),
    ]:
        if data:
            if small_n:
                print(f"  {label:<30} {min(data):>8.1f} {_pct(data,50):>8.1f} "
                      f"{max(data):>8.1f} {_mean(data):>8.1f}")
            else:
                print(f"  {label:<30} {_pct(data,50):>8.1f} {_pct(data,95):>8.1f} "
                      f"{_pct(data,99):>8.1f} {_mean(data):>8.1f}")
    print()
    print(f"  routes       : {routes}")
    print(f"  cache hits   : {cache_hits}/{len(ok)} ({100*cache_hits//max(len(ok),1)}%)")
    print(f"  rag matches  : {rag_hits}/{len(ok)} ({100*rag_hits//max(len(ok),1)}%)")
    if errors:
        counts: Dict[str, int] = {}
        for r in errors:
            counts[r["error"][:40]] = counts.get(r["error"][:40], 0) + 1
        print(f"  error types  : {counts}")
    print(f"{'─' * w}\n")


def load_queries(path: str, count: Optional[int] = None) -> List[str]:
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] query file not found: {path}", file=sys.stderr)
        sys.exit(1)
    lines = [l.strip() for l in p.read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    if not lines:
        print(f"[ERROR] query file is empty: {path}", file=sys.stderr)
        sys.exit(1)
    if count:
        while len(lines) < count:
            lines = lines + lines
        lines = lines[:count]
    return lines



def run_bge_isolated(size_bucket: str, texts: List[str], out_path: Path) -> None:
    """BGE-only microbenchmark via POST /v1/isolated/embed.

    Work runs inside the FastAPI uvicorn process so perf stat -p <uvicorn_pid>
    captures the actual AVX-512 FP instructions — unlike kubectl exec which
    spawns a separate Python process invisible to perf.
    """
    print(f"[bge_isolated/{size_bucket}] {len(texts)} queries via /v1/isolated/embed")
    try:
        resp = _session.post(
            f"{BASE_URL}/v1/isolated/embed",
            json={"texts": texts, "tenant_id": "tenantA"},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        rows = resp.json()["results"]
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["text_words", "embed_ms", "mode", "size_bucket"])
            w.writeheader()
            for r in rows:
                r.update({"mode": "bge_isolated", "size_bucket": size_bucket})
                w.writerow(r)
        times = [r["embed_ms"] for r in rows]
        print(f"  min={min(times):.1f}ms  median={_pct(times,50):.1f}ms  max={max(times):.1f}ms  n={len(rows)}")
    except Exception as e:
        print(f"[ERROR] BGE isolated failed: {e}", file=sys.stderr)


def run_hnsw_isolated(size_bucket: str, texts: List[str], out_path: Path) -> None:
    """HNSW-only microbenchmark via POST /v1/isolated/search.

    Work runs inside the FastAPI uvicorn process so perf stat captures it.
    Embed time is excluded from hnsw_ms on the server side.
    """
    print(f"[hnsw_isolated/{size_bucket}] {len(texts)} searches via /v1/isolated/search")
    try:
        resp = _session.post(
            f"{BASE_URL}/v1/isolated/search",
            json={"texts": texts, "tenant_id": "tenantA", "top_k": 4},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        rows = resp.json()["results"]
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["text_words", "hnsw_ms", "num_hits", "mode", "size_bucket"])
            w.writeheader()
            for r in rows:
                r.update({"mode": "hnsw_isolated", "size_bucket": size_bucket})
                w.writerow(r)
        times = [r["hnsw_ms"] for r in rows]
        print(f"  min={min(times):.1f}ms  median={_pct(times,50):.1f}ms  max={max(times):.1f}ms  n={len(rows)}")
    except Exception as e:
        print(f"[ERROR] HNSW isolated failed: {e}", file=sys.stderr)


def run_sc(
    args: argparse.Namespace,
    out_path: Path,
    scenario: str,  # "a" = bypass_rag, "b" = full pipeline
    concurrency: int = 1,
    streaming: bool = False,
) -> None:
    """SC benchmark.

    Scenario A (bypass_rag=True):  measures pure SC overhead without RAG cost.
    Scenario B (bypass_rag=False): production flow — RAG runs first, SC accept_key
                                   includes context_fingerprint.  Warm and measure
                                   queries must retrieve identical chunks for hits.

    Flags:
      --warmup-only  Run warmup only (no measurement). Call this BEFORE perf starts.
      --no-warmup    Skip warmup, run measurement only. Call this AFTER warmup-only run.
    """
    do_rag = (scenario == "b")
    warmup_only = getattr(args, "warmup_only", False)
    no_warmup   = getattr(args, "no_warmup",   False)
    sc_label    = "B(rag)" if do_rag else "A(iso)"

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        # Warmup phase — populate cache so measure queries get hits.
        if not no_warmup and args.warmup_queries:
            warmup_qs = load_queries(args.warmup_queries, args.warmup)
            print(f"  [warmup/{sc_label}] {len(warmup_qs)} queries (populating cache)...")
            warmup_results = run_phase(
                warmup_qs, "cache_warmup", args.size_bucket, writer,
                bypass_rag=(not do_rag), bypass_cache=False,
                max_tokens=args.max_tokens, concurrency=concurrency, silent=True,
            )
            hit_rate = sum(1 for r in warmup_results if r["cache_hit"]) / max(len(warmup_results), 1)
            print(f"  [warmup/{sc_label}] done. Hit rate: {hit_rate*100:.0f}%")
            if hit_rate < 0.50:
                print(f"  [warmup/{sc_label}] low hit rate — second pass")
                run_phase(warmup_qs, "cache_warmup", args.size_bucket, writer,
                          bypass_rag=(not do_rag), bypass_cache=False,
                          max_tokens=args.max_tokens, concurrency=concurrency, silent=True)

        if warmup_only:
            print(f"  [warmup/{sc_label}] warmup-only mode — stopping before measurement.")
            return

        # Measurement phase — must run after warmup.
        # Stream measurement queries so cache MISSES (which actually hit vLLM)
        # capture real TTFT/TPOT. Cache HITS won't have stream timings — that's
        # expected and documented in send_query_streaming.
        queries = load_queries(args.queries, args.count)
        mode_label = "cache_b" if do_rag else "cache_a"
        results = run_phase(
            queries, mode_label, args.size_bucket, writer,
            bypass_rag=(not do_rag), bypass_cache=False,
            max_tokens=args.max_tokens, concurrency=concurrency,
            streaming=streaming,
        )

        ok_r   = [r for r in results if not r["error"]]
        hits   = [r for r in ok_r if r["cache_hit"]]
        misses = [r for r in ok_r if not r["cache_hit"]]
        hr     = len(hits) / max(len(ok_r), 1)
        sc_tag = f"SC-{'B-FullPipeline' if do_rag else 'A-Isolated'} / {args.size_bucket}"
        print_summary(hits,   f"{sc_tag} — HITS  ({len(hits)}/{len(ok_r)}, {hr*100:.0f}%)")
        if misses:
            print_summary(misses, f"{sc_tag} — MISSES ({len(misses)}/{len(ok_r)})")
        if hr < 0.80:
            print(f"[WARN] SC-{scenario.upper()} hit rate {hr*100:.0f}% below 80%", file=sys.stderr)


def main() -> None:
    global BASE_URL
    parser = argparse.ArgumentParser(description="GenAI workload characterization query runner")
    parser.add_argument("--mode", required=True,
                        choices=["rag", "cache", "cache_a", "cache_b",
                                 "llm_direct", "bge_isolated", "hnsw_isolated"])
    parser.add_argument("--queries",        required=True)
    parser.add_argument("--warmup-queries", default=None)
    parser.add_argument("--size-bucket",    default="medium",
                        choices=["short", "medium", "long", "very_long"])
    parser.add_argument("--count",    type=int, default=300)
    parser.add_argument("--warmup",   type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--url",      default=BASE_URL)
    parser.add_argument("--out-dir",  default=None)
    parser.add_argument("--sc-scenario", default="a", choices=["a", "b"],
                        help="SC scenario: a=bypass_rag (isolated), b=full pipeline")
    parser.add_argument("--warmup-only", action="store_true",
                        help="SC: run warmup phase only, skip measurement. Use before perf starts.")
    parser.add_argument("--no-warmup", action="store_true",
                        help="SC: skip warmup, run measurement only. Use after separate warmup.")
    parser.add_argument("--stream", action="store_true",
                        help="Use SSE streaming to measure real TTFT and TPOT.")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of queries fired in parallel via ThreadPoolExecutor. "
                             "Default 1 (strict serial). Applies to warmup AND measurement phases. "
                             "Higher values keep CPU busier during the perf window and shorten wall-clock; "
                             "vLLM continuous batching absorbs them on the GPU side.")
    args = parser.parse_args()
    BASE_URL = args.url

    out_dir = Path(args.out_dir) if args.out_dir else \
              RESULTS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Normalise mode aliases
    mode = args.mode
    if mode == "cache":
        mode = f"cache_{args.sc_scenario}"

    print(f"\n[query_runner] mode={mode}  size={args.size_bucket}  "
          f"count={args.count}  max_tokens={args.max_tokens}  url={BASE_URL}")

    out_path = out_dir / f"{mode}_{args.size_bucket}_{ts}.csv"

    if mode == "bge_isolated":
        run_bge_isolated(args.size_bucket, load_queries(args.queries, args.count), out_path)
        return

    if mode == "hnsw_isolated":
        run_hnsw_isolated(args.size_bucket, load_queries(args.queries, args.count), out_path)
        return

    if mode in ("cache_a", "cache_b"):
        # SC cells force non-streaming regardless of --stream flag.
        # The orchestrator skips the semantic cache lookup entirely when
        # stream=true (chat.py:296 `if semantic_enabled and not stream`),
        # which would make cache_embed_ms/cache_milvus_ms/cache_mongo_ms all 0
        # and force every query through the LLM path — defeating SC characterization.
        # Non-streaming SC still captures e2e_ms + cache sub-timings (the actual
        # hardware characterization targets); we just don't get per-token TTFT/TPOT
        # for SC, which is fine because cache-hit "TTFT" is fully decomposed into
        # the cache_*_ms columns already.
        scenario = mode[-1]  # "a" or "b"
        run_sc(args, out_path, scenario, concurrency=args.concurrency,
               streaming=False)
        print(f"[query_runner] results → {out_path}")
        return

    # RAG / LLM-direct
    bypass_rag   = (mode == "llm_direct")
    bypass_cache = (mode == "llm_direct") or (mode == "rag")
    streaming    = bool(args.stream)

    if streaming:
        print(f"[query_runner] streaming=True — measuring real TTFT/TPOT via SSE")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        queries = load_queries(args.queries, args.count)
        results = run_phase(queries, mode, args.size_bucket, writer,
                            bypass_rag=bypass_rag, bypass_cache=bypass_cache,
                            max_tokens=args.max_tokens, concurrency=args.concurrency,
                            streaming=streaming)
        print_summary(results, f"{mode.upper()} / {args.size_bucket} [tok={args.max_tokens}]")

    print(f"[query_runner] results → {out_path}")


if __name__ == "__main__":
    main()
