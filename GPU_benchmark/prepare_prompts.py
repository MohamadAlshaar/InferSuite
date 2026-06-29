#!/usr/bin/env python3
"""Pre-generate prefill-sweep prompts from real open_ragbench arXiv papers.

This is NOT ingestion. It downloads a couple of papers' raw text from the same
HF corpus the RAG ingest uses (vectara/open_ragbench), concatenates it, and
slices it to each target length. No embedding, no Milvus, no SeaweedFS — seconds,
not hours. Output: GPU_benchmark/prompts/prefill_<N>.txt, bundled into the image.

Token counts are approximate here (sized by characters at ~CHARS_PER_TOKEN); the
sweep records the *actual* prompt_tokens vLLM reports, so analysis bins on ground
truth. If HF/pypdf are unavailable, falls back to a repeated seed paragraph
(prefill cost depends on token count, not content, so the curve is identical).

Usage:  python3 prepare_prompts.py [--tokens 128,256,...] [--out prompts]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

CHARS_PER_TOKEN = 4.2  # English prose, this tokenizer family (approx; measured at runtime)
HF_REPO = "vectara/open_ragbench"

_SEED = (
    "The system processes incoming requests through a sequence of stages, each "
    "transforming the data and passing it forward. Measurements are recorded at "
    "every boundary so that the cost of each stage can be attributed precisely. "
)


def _ragbench_text(min_chars: int) -> str:
    """Download open_ragbench arXiv corpus papers until we have >= min_chars of text.

    The corpus is JSON (pdf/arxiv/corpus/<id>.json) with {abstract, sections:[{text}]}.
    """
    import json

    from huggingface_hub import hf_hub_download, list_repo_files

    corpus = sorted(f for f in list_repo_files(HF_REPO, repo_type="dataset")
                    if f.startswith("pdf/arxiv/corpus/") and f.endswith(".json"))
    text = ""
    for fname in corpus:
        path = hf_hub_download(HF_REPO, fname, repo_type="dataset")
        try:
            doc = json.load(open(path))
            text += (doc.get("abstract", "") or "") + "\n"
            for sec in doc.get("sections", []) or []:
                text += (sec.get("text", "") or "") + "\n"
        except Exception:
            continue
        print(f"  + {fname}  ({len(text):,} chars)")
        if len(text) >= min_chars:
            break
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="128,256,512,1024,2048,4096,8192")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "prompts"))
    args = ap.parse_args()

    targets = sorted(int(t) for t in args.tokens.split(","))
    max_chars = int(max(targets) * CHARS_PER_TOKEN * 1.3)  # headroom for slicing

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    try:
        text = _ragbench_text(max_chars)
        source = "open_ragbench"
        if len(text) < max_chars:
            raise RuntimeError("not enough ragbench text")
    except Exception as exc:
        print(f"  ragbench unavailable ({exc}); using filler (same numbers, prefill is content-independent)")
        text = ""
        while len(text) < max_chars:
            text += _SEED
        source = "filler"

    for n in targets:
        chars = int(n * CHARS_PER_TOKEN)
        (out / f"prefill_{n}.txt").write_text(text[:chars])
    print(f"wrote {len(targets)} prompts to {out} (source: {source})")


if __name__ == "__main__":
    main()
