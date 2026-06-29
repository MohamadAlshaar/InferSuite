#!/usr/bin/env python3
"""Streaming OpenAI-compatible client for the GPU sweeps.

Talks DIRECTLY to the vLLM server (not the FastAPI orchestrator), so:
  - vLLM natively honours `ignore_eos` + `min_tokens` → exact output length,
  - there is no RAG / cache / embed path to contaminate the measurement,
  - TTFT and TPOT are measured purely client-side from the SSE stream.

One request → one dict of measurements. Sequential use only (concurrency=1).
"""
from __future__ import annotations

import json
import time
from typing import Dict, Optional

import requests


def send(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    *,
    force_exact: bool = True,
    timeout_s: float = 600.0,
) -> Dict[str, object]:
    """Send one streaming chat request and measure prefill/decode timings.

    force_exact=True pins the output to exactly `max_tokens` tokens via
    vLLM's `ignore_eos` + `min_tokens` (so EOS cannot end generation early).

    Returns a row dict. Timing fields are None when no content arrived.
    """
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if force_exact:
        # vLLM extensions to the OpenAI schema — honoured because we hit vLLM directly.
        payload["min_tokens"] = max_tokens
        payload["ignore_eos"] = True

    row: Dict[str, object] = {
        "http_status": 0,
        "error": "",
        "ttft_ms": None,
        "generation_ms": None,
        "tpot_ms": None,
        "e2e_ms": None,
        "n_chunks_with_content": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }

    t_start = time.perf_counter()
    t_first: Optional[float] = None
    t_last: Optional[float] = None
    n_chunks = 0
    completion_tokens = 0
    prompt_tokens = 0

    try:
        resp = requests.post(url, json=payload, timeout=timeout_s, stream=True)
        row["http_status"] = resp.status_code
        if resp.status_code != 200:
            row["error"] = f"http_{resp.status_code}: {resp.text[:200]}"
            row["e2e_ms"] = round((time.perf_counter() - t_start) * 1000.0, 3)
            return row

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                continue
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            choice = (chunk.get("choices") or [{}])[0]
            content = (choice.get("delta") or {}).get("content") or choice.get("text") or ""
            if content:
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                    row["ttft_ms"] = round((now - t_start) * 1000.0, 3)
                t_last = now
                n_chunks += 1
            usage = chunk.get("usage") or {}
            if usage:
                completion_tokens = int(usage.get("completion_tokens") or completion_tokens)
                prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)

    except requests.RequestException as exc:
        row["error"] = f"request_error: {exc}"
        row["e2e_ms"] = round((time.perf_counter() - t_start) * 1000.0, 3)
        return row

    row["e2e_ms"] = round((time.perf_counter() - t_start) * 1000.0, 3)
    row["n_chunks_with_content"] = n_chunks
    row["prompt_tokens"] = prompt_tokens
    row["completion_tokens"] = completion_tokens

    if t_first is not None and t_last is not None:
        gen_ms = round((t_last - t_first) * 1000.0, 3)
        row["generation_ms"] = gen_ms
        # TPOT: time per output token over the decode phase. Needs >=2 tokens.
        denom = (completion_tokens - 1) if completion_tokens > 1 else (n_chunks - 1)
        if gen_ms > 0 and denom and denom > 0:
            row["tpot_ms"] = round(gen_ms / denom, 4)

    return row
