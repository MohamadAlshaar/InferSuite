#!/usr/bin/env python3
"""Build prompts of a target token length for the prefill sweep.

Prefill cost depends on the *number* of input tokens, not their content, so we
build prompts from a repeated natural-language seed and size them to a target
token count. Exactness is not required client-side: every request records the
*actual* `prompt_tokens` reported by vLLM, and analysis bins on that ground
truth — the target is only a label.

If `transformers` + the model tokenizer are available, prompts are truncated to
the exact token count; otherwise a ~chars-per-token heuristic is used. Either
way the measured `prompt_tokens` is authoritative.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

# A neutral English paragraph. Content is irrelevant to prefill cost; we only
# need a realistic stream of tokens to repeat up to the target length.
_SEED = (
    "The system processes incoming requests through a sequence of stages, each "
    "transforming the data and passing it forward. Measurements are recorded at "
    "every boundary so that the cost of each stage can be attributed precisely. "
    "Hardware counters expose how the processor spends its cycles, while engine "
    "metrics describe queue depth and memory occupancy under load. Together they "
    "explain where time goes and which resource limits throughput. "
)

# Rough average for English with this tokenizer family; only used in the fallback.
_CHARS_PER_TOKEN = 4.0


@lru_cache(maxsize=1)
def _tokenizer(model_repo: str):
    """Load the HF tokenizer for exact truncation, or return None on any failure."""
    try:
        from transformers import AutoTokenizer  # type: ignore

        return AutoTokenizer.from_pretrained(model_repo)
    except Exception:
        return None


def build_prompt(target_tokens: int, model_repo: str = "Qwen/Qwen2.5-14B-Instruct") -> str:
    """Return a prompt whose token length is ~target_tokens (exact if tokenizer available)."""
    if target_tokens <= 1:
        # Minimal prompt for the decode sweep (input pinned at its minimum).
        return "Hi"

    # Prefer the pre-generated real ragbench prompt for this length (bundled in
    # the image by prepare_prompts.py). Falls through to tokenizer/heuristic if absent.
    bundled = PROMPT_DIR / f"prefill_{target_tokens}.txt"
    if bundled.exists():
        return bundled.read_text()

    tok = _tokenizer(model_repo)
    if tok is not None:
        # Repeat the seed until we have enough tokens, then truncate exactly.
        text = _SEED
        while len(tok.encode(text)) < target_tokens + 8:
            text += _SEED
        ids = tok.encode(text)[:target_tokens]
        return tok.decode(ids, skip_special_tokens=True)

    # Heuristic fallback: size by characters. Actual token count is measured
    # at request time, so a small miss here is harmless.
    target_chars = int(target_tokens * _CHARS_PER_TOKEN)
    text = ""
    while len(text) < target_chars:
        text += _SEED
    return text[:target_chars]
