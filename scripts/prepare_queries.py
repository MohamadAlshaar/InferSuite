#!/usr/bin/env python3
"""
Prepare benchmark query files from standard datasets.

Datasets:
  RAG        — vectara/open_ragbench: real QA pairs from various corpora
  Cache (SC) — QQP (glue/qqp is_duplicate=1): Quora paraphrase pairs
  LLM-direct — RyokoAI/ShareGPT52K: real human conversation turns

Outputs:
  benchmark_queries/rag/short.txt          (10–30 tok, bare questions)
  benchmark_queries/rag/medium.txt         (100–150 tok, question + padding)
  benchmark_queries/rag/long.txt           (350–512 tok)
  benchmark_queries/rag/very_long.txt      (600–1000 tok)
  benchmark_queries/rag_pure_fetch/queries.txt  (5–25 tok, bare questions)
  benchmark_queries/cache/short_warm.txt   QQP question1
  benchmark_queries/cache/short_measure.txt QQP question2 (paired)
  benchmark_queries/cache/medium_warm.txt
  benchmark_queries/cache/medium_measure.txt
  benchmark_queries/llm_direct/short.txt
  benchmark_queries/llm_direct/medium.txt
  benchmark_queries/llm_direct/long.txt
  benchmark_queries/llm_direct/very_long.txt

Usage:
  python3 scripts/prepare_queries.py --generate --count 400
  python3 scripts/prepare_queries.py --ingest   # downloads PDFs + ingests
  python3 scripts/prepare_queries.py --verify   # checks RAG/SC hit rates
  python3 scripts/prepare_queries.py --check    # inspect query files on disk
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Constants ─────────────────────────────────────────────────────────────────

KERNEL_ROOT = Path(__file__).parent.parent
QUERIES_DIR = KERNEL_ROOT / "benchmark_queries"
NAMESPACE   = os.getenv("BENCHMARK_NAMESPACE", "llm-service")
BASE_URL    = os.getenv("BENCHMARK_URL", "http://localhost:8080")

# Token bucket targets matching the benchmark spec
BUCKET_TARGETS = {
    "short":     (10,   30),    # 10–30 tokens
    "medium":    (100, 150),    # 100–150 tokens
    "long":      (350, 512),    # 350–512 tokens
    "very_long": (600, 1000),   # 600–1000 tokens
    "pure_fetch": (5,   25),    # bare questions for pure-fetch path
    "sc_short":  (10,   25),    # SC short bucket
    "sc_medium": (80,  130),    # SC medium bucket
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def _token_count(text: str) -> int:
    return len(re.findall(r'\S+', text))


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    words = text.split()
    return " ".join(words[:max_tokens])


def _pad_to_tokens(base_query: str, context: str, target_min: int, target_max: int) -> str:
    current = _token_count(base_query)
    if current >= target_min:
        return _truncate_to_tokens(base_query, target_max) if current > target_max else base_query
    needed = target_min - current
    ctx_words = context.split()
    pad = " ".join(ctx_words[:needed + 30])
    combined = f"{base_query}\n\nContext: {pad}"
    if _token_count(combined) > target_max:
        combined = _truncate_to_tokens(combined, target_max)
    return combined


def write_query_file(path: Path, queries: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(q.strip().replace("\n", " ") for q in queries if q.strip())
    path.write_text(text + "\n")
    print(f"  wrote {len(queries)} queries → {path.relative_to(KERNEL_ROOT)}")


def log(msg: str) -> None:
    print(f"[prepare_queries] {msg}", flush=True)


# ── open_ragbench (RAG dataset) ────────────────────────────────────────────────

_RAGBENCH_BASE = (
    "https://huggingface.co/datasets/vectara/open_ragbench/resolve/main/pdf/arxiv"
)


def load_open_ragbench(n: int = 5000) -> List[Dict]:
    """Load vectara/open_ragbench via direct JSON download from HuggingFace.

    The dataset stores queries in pdf/arxiv/queries.json and relevance judgments
    (query→doc_id mapping) in pdf/arxiv/qrels.json.  We download both and join them
    so each record also carries the source doc_id — used by ingest to download the
    matching paper.
    """
    log(f"Loading vectara/open_ragbench queries from HuggingFace (up to {n} records)...")
    try:
        with urllib.request.urlopen(_RAGBENCH_BASE + "/queries.json", timeout=30) as r:
            queries_raw = json.loads(r.read())
        log(f"  downloaded {len(queries_raw)} queries")
    except Exception as e:
        log(f"WARNING: Could not download open_ragbench queries.json: {e}")
        log("Falling back to synthetic RAG queries")
        return []

    try:
        with urllib.request.urlopen(_RAGBENCH_BASE + "/qrels.json", timeout=30) as r:
            qrels_raw = json.loads(r.read())
    except Exception as e:
        log(f"WARNING: Could not download open_ragbench qrels.json: {e}")
        qrels_raw = {}

    records = []
    for qid, qdata in queries_raw.items():
        q = str(qdata.get("query", "")).strip()
        if not q or len(q) < 5:
            continue
        qrel = qrels_raw.get(qid, {})
        records.append({
            "question": q,
            "answer": "",
            "dataset": qdata.get("source", ""),
            "doc_id": qrel.get("doc_id", ""),   # arXiv paper ID this question references
        })
        if len(records) >= n:
            break

    log(f"  loaded {len(records)} open_ragbench records")
    return records


def build_rag_queries(records: List[Dict], count: int) -> Dict[str, List[str]]:
    """Build 4 size-bucketed RAG query files from open_ragbench records."""
    if not records:
        log("No open_ragbench records — using synthetic fallback queries")
        return _synthetic_rag_queries(count)

    while len(records) < count:
        records = records + records
    records = records[:count]

    # General-purpose filler for padding to larger buckets
    FILLER = (
        "This question relates to the documents in the knowledge base. "
        "The answer can be found by examining the relevant source material carefully. "
        "Consider all related context, background information, and supporting evidence "
        "when formulating a comprehensive response to the query above. "
        "The knowledge base contains detailed information from multiple authoritative sources. "
    ) * 30  # ~1800 tokens of filler

    buckets: Dict[str, List[str]] = {b: [] for b in ("short", "medium", "long", "very_long")}

    for rec in records:
        q = rec["question"]

        # short: bare question (naturally 5–30 tokens for most open_ragbench questions)
        buckets["short"].append(_truncate_to_tokens(q, 30))

        # medium: pad to 100–150 tokens
        tmin, tmax = BUCKET_TARGETS["medium"]
        buckets["medium"].append(_pad_to_tokens(q, FILLER, tmin, tmax))

        # long: pad to 350–512 tokens
        tmin, tmax = BUCKET_TARGETS["long"]
        buckets["long"].append(_pad_to_tokens(q, FILLER, tmin, tmax))

        # very_long: pad to 600–1000 tokens
        tmin, tmax = BUCKET_TARGETS["very_long"]
        buckets["very_long"].append(_pad_to_tokens(q, FILLER, tmin, tmax))

    for bucket, qs in buckets.items():
        tokens = [_token_count(q) for q in qs]
        if tokens:
            tmin, tmax = BUCKET_TARGETS[bucket]
            in_range = sum(1 for t in tokens if tmin <= t <= tmax)
            avg = sum(tokens) / len(tokens)
            log(f"  rag/{bucket}: n={len(qs)}, avg_tokens={avg:.0f}, in_range={in_range}/{len(qs)}")

    return buckets


def build_rag_pure_fetch_queries(records: List[Dict], count: int) -> List[str]:
    """Build bare questions for RAG pure-fetch path (5–25 tokens, no padding).

    Purpose: isolate retrieval infrastructure CPU cost (BGE embed → Milvus HNSW →
    SeaweedFS fetch) from query processing cost. The query is short; all context
    comes from RAG retrieval.
    """
    if not records:
        return _synthetic_rag_queries(count).get("short", [])[:count]

    queries = []
    for rec in records:
        q = rec["question"].strip()
        t = _token_count(q)
        # Accept naturally short questions (5–25 tokens)
        if 5 <= t <= 25:
            queries.append(q)
        elif t > 25:
            # Truncate to 25 tokens — keeps the core question
            queries.append(_truncate_to_tokens(q, 25))
        # Skip questions under 5 tokens (likely malformed)

    while len(queries) < count:
        queries = queries + queries
    queries = queries[:count]

    tokens = [_token_count(q) for q in queries]
    avg = sum(tokens) / len(tokens)
    log(f"  rag_pure_fetch: n={len(queries)}, avg_tokens={avg:.0f}")
    return queries


def _synthetic_rag_queries(count: int) -> Dict[str, List[str]]:
    base_questions = [
        "What is the main contribution of this paper?",
        "How does the proposed method compare to baselines?",
        "What datasets were used for evaluation?",
        "What are the limitations of the proposed approach?",
        "What is the computational complexity of the algorithm?",
        "How does the model handle out-of-distribution inputs?",
        "What ablation studies were performed?",
        "What is the training procedure for the model?",
        "How does the method scale with dataset size?",
        "What are the key hyperparameters and their effects?",
    ] * (count // 10 + 1)
    base_questions = base_questions[:count]

    filler = ("The paper presents a novel approach to natural language processing using "
              "transformer architectures. The model achieves state-of-the-art results on "
              "multiple benchmarks. Extensive experiments demonstrate the effectiveness of "
              "the proposed method across diverse tasks and domains. ") * 50

    buckets = {}
    for bucket, (tmin, tmax) in BUCKET_TARGETS.items():
        if bucket in ("pure_fetch", "sc_short", "sc_medium"):
            continue
        qs = []
        for q in base_questions:
            qs.append(_pad_to_tokens(q, filler, tmin, tmax))
        buckets[bucket] = qs
    return buckets


# ── QQP (Semantic Cache dataset) ──────────────────────────────────────────────

def load_qqp(count: int) -> List[Tuple[str, str]]:
    """Load QQP duplicate pairs. Returns list of (question1, question2)."""
    try:
        from datasets import load_dataset
    except ImportError:
        log("ERROR: 'datasets' not installed")
        return []

    log(f"Loading QQP duplicate pairs...")
    try:
        ds = load_dataset("glue", "qqp", split="train", trust_remote_code=True)
    except Exception as e:
        log(f"WARNING: Could not load glue/qqp: {e}")
        return _synthetic_qqp_pairs(count)

    pairs = []
    for item in ds:
        if item.get("label") == 1:
            q1 = str(item.get("question1", "")).strip()
            q2 = str(item.get("question2", "")).strip()
            if q1 and q2 and q1 != q2 and len(q1) > 10 and len(q2) > 10:
                pairs.append((q1, q2))
        if len(pairs) >= count * 3:
            break

    log(f"  loaded {len(pairs)} QQP duplicate pairs")
    return pairs


def _synthetic_qqp_pairs(count: int) -> List[Tuple[str, str]]:
    pairs = [
        ("How do I get better at machine learning?", "What are the best ways to improve in ML?"),
        ("What is the best way to learn Python?", "How should I start learning Python programming?"),
        ("What causes inflation?", "Why does inflation happen in an economy?"),
        ("How does photosynthesis work?", "What is the process of photosynthesis?"),
        ("What are the benefits of exercise?", "Why is regular exercise good for health?"),
        ("How do neural networks learn?", "What is the learning process in neural networks?"),
        ("What is quantum computing?", "How does a quantum computer work?"),
        ("What is the difference between AI and ML?", "How is machine learning different from AI?"),
        ("What are transformers in NLP?", "How do transformer models work in natural language processing?"),
        ("What causes climate change?", "Why is the climate changing?"),
    ] * (count // 10 + 2)
    return pairs[:count]


def build_cache_queries(pairs: List[Tuple[str, str]], count: int) -> Dict[str, Dict[str, List[str]]]:
    """Build SC warm/measure files for short and medium buckets."""
    if not pairs:
        pairs = _synthetic_qqp_pairs(count)

    result: Dict[str, Dict[str, List[str]]] = {}

    sc_buckets = {"short": BUCKET_TARGETS["sc_short"], "medium": BUCKET_TARGETS["sc_medium"]}
    for bucket, (tmin, tmax) in sc_buckets.items():
        warm, measure = [], []

        while len(warm) < count:
            for q1, q2 in pairs:
                if bucket == "short":
                    w_query = _truncate_to_tokens(q1, tmax)
                    m_query = _truncate_to_tokens(q2, tmax)
                else:
                    filler = ("What are the implications and related work in this area? "
                              "Discuss the key concepts and methodologies involved. ") * 10
                    w_query = _pad_to_tokens(q1, filler, tmin, tmax)
                    m_query = _pad_to_tokens(q2, filler, tmin, tmax)

                if _token_count(w_query) >= tmin * 0.6:
                    warm.append(w_query)
                    measure.append(m_query)

                if len(warm) >= count:
                    break

        warm = warm[:count]
        measure = measure[:count]

        w_tokens = [_token_count(q) for q in warm]
        log(f"  cache/{bucket}: n={len(warm)}, avg_tokens={sum(w_tokens)/len(w_tokens):.0f}")
        result[bucket] = {"warm": warm, "measure": measure}

    return result


# ── ShareGPT52K (LLM-direct dataset) ─────────────────────────────────────────

def load_sharegpt(count: int) -> List[Dict]:
    """Load RyokoAI/ShareGPT52K human turns for LLM-direct queries."""
    try:
        from datasets import load_dataset
    except ImportError:
        return []

    log(f"Loading RyokoAI/ShareGPT52K...")
    try:
        ds = load_dataset("RyokoAI/ShareGPT52K", split="train", trust_remote_code=True)
    except Exception as e:
        log(f"WARNING: Could not load ShareGPT52K: {e}")
        return _synthetic_sharegpt(count)

    records = []
    for item in ds:
        convs = item.get("conversations", [])
        if not isinstance(convs, list):
            continue
        for turn in convs:
            if isinstance(turn, dict) and turn.get("from") == "human":
                text = str(turn.get("value", "")).strip()
                if len(text) > 10:
                    records.append({"question": text, "context": ""})
        if len(records) >= count * 5:
            break

    log(f"  loaded {len(records)} ShareGPT human turns")
    return records


def _synthetic_sharegpt(count: int) -> List[Dict]:
    questions = [
        "Explain the difference between supervised and unsupervised learning.",
        "Write a Python function to sort a list of dictionaries by a key.",
        "What is the capital city of Australia?",
        "Who invented the telephone?",
        "What year did World War II end?",
        "How does TCP/IP work?",
        "Explain gradient descent in simple terms.",
        "What is the speed of light in meters per second?",
        "In what year did the Berlin Wall fall?",
        "What is the difference between RAM and ROM?",
    ] * (count // 10 + 2)
    filler = ("This is a question that requires a detailed and comprehensive answer. "
              "Please provide a thorough explanation with relevant examples and context. ") * 100
    return [{"question": q, "context": filler} for q in questions[:count * 3]]


def build_llm_direct_queries(records: List[Dict], count: int) -> Dict[str, List[str]]:
    """Build 4 size-bucketed LLM-direct query files from ShareGPT turns.

    ShareGPT has natural length variation — filter by token count for each bucket
    rather than padding, to preserve the real distribution.
    """
    if not records:
        records = _synthetic_sharegpt(count)

    FILLER = (
        "Please provide a comprehensive and detailed response to the following. "
        "Consider all relevant aspects, background context, edge cases, and practical "
        "implications. Include concrete examples where appropriate. "
        "Draw on relevant knowledge from related domains when helpful. "
        "Organize your response clearly with appropriate structure. "
    ) * 25  # ~600 tokens

    buckets: Dict[str, List[str]] = {b: [] for b in ("short", "medium", "long", "very_long")}

    # First pass: assign records to buckets by natural token count
    for rec in records:
        q = rec["question"]
        t = _token_count(q)
        for bucket, (tmin, tmax) in [
            ("short",     BUCKET_TARGETS["short"]),
            ("medium",    BUCKET_TARGETS["medium"]),
            ("long",      BUCKET_TARGETS["long"]),
            ("very_long", BUCKET_TARGETS["very_long"]),
        ]:
            if tmin <= t <= tmax and len(buckets[bucket]) < count:
                buckets[bucket].append(q)

    # Second pass: fill under-populated buckets via padding
    for bucket, (tmin, tmax) in BUCKET_TARGETS.items():
        if bucket in ("pure_fetch", "sc_short", "sc_medium"):
            continue
        qs = buckets[bucket]
        idx = 0
        while len(qs) < count and idx < len(records):
            q = records[idx]["question"]
            padded = _pad_to_tokens(q, FILLER, tmin, tmax)
            if _token_count(padded) >= tmin:
                qs.append(padded)
            idx += 1
        # Cycle if still short
        while len(qs) < count:
            qs = qs + qs
        qs = qs[:count]
        buckets[bucket] = qs
        tokens = [_token_count(q) for q in qs]
        log(f"  llm_direct/{bucket}: n={len(qs)}, avg_tokens={sum(tokens)/len(tokens):.0f}")

    return buckets


# ── Ingestion ──────────────────────────────────────────────────────────────────

def ingest_rag_pdfs(n_papers: int = 200) -> None:
    """Ingest open_ragbench arXiv corpus into Milvus + SeaweedFS via pre-extracted HF JSON.

    Delegates to scripts/ingest_ragbench_to_milvus.py which downloads the corpus
    sections directly from HuggingFace (no PDF download needed) and ingests into
    the running Milvus + SeaweedFS pods.
    """
    try:
        pod = subprocess.check_output(
            ["kubectl", "get", "pod", "-n", NAMESPACE, "-l", "app=llm-service-kernel",
             "-o", "jsonpath={.items[0].metadata.name}"],
            text=True,
        ).strip()
    except Exception as e:
        log(f"ERROR: kubectl failed: {e}")
        return

    if not pod:
        log(f"ERROR: no llm-service-kernel pod found in namespace {NAMESPACE}")
        return

    log(f"Pod: {pod}")
    log(f"Ingesting {n_papers} ragbench papers into Milvus + SeaweedFS via HF JSON...")

    # Copy the ingest script into the pod and run it there — it has access to
    # the Milvus and SeaweedFS services, and the BGE model is already mounted.
    ingest_script = Path(__file__).parent / "ingest_ragbench_to_milvus.py"
    if not ingest_script.exists():
        log(f"ERROR: {ingest_script} not found")
        return

    # kubectl cp the script into the pod, then exec it
    dest = f"/tmp/ingest_ragbench_to_milvus.py"
    cp_proc = subprocess.run(
        ["kubectl", "cp", "-n", NAMESPACE, str(ingest_script), f"{pod}:{dest}"],
        capture_output=True, text=True,
    )
    if cp_proc.returncode != 0:
        log(f"ERROR: kubectl cp failed: {cp_proc.stderr}")
        return

    cmd = [
        "kubectl", "exec", "-n", NAMESPACE, pod, "--",
        "python3", dest,
        "--n-papers",            str(n_papers),
        "--bge-model-path",      "/app/fastapi_runtime_assets/models/bge-base-en-v1.5",
        "--milvus-uri",          "http://milvus.llm-service.svc.cluster.local:19530",
        "--milvus-token",        "root:Milvus",
        "--collection",          "rag_chunks_seaweed_v2",
        "--s3-endpoint-url",     "http://seaweed-s3.llm-service.svc.cluster.local:8333",
        "--s3-access-key-id",    "llmbenchadmin",
        "--s3-secret-access-key","llmbenchsecretkey123",
        "--s3-bucket",           "llm-rag-store",
        "--tenant",              "tenantA",
        "--queries-dir",         "/app/benchmark_queries/rag",
        "--drop-existing",
    ]
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        log(f"ERROR: ingest failed (exit {proc.returncode})")
    else:
        log("RAG ingest complete — corpus is now in Milvus + SeaweedFS")


# ── Verification ───────────────────────────────────────────────────────────────

def verify_rag_hit_rate(query_file: str, n: int = 20) -> float:
    import requests as req
    queries = Path(query_file).read_text().strip().splitlines()[:n]
    hits = 0
    for q in queries:
        try:
            r = req.post(f"{BASE_URL}/v1/chat/completions",
                         json={"model": os.getenv("BENCHMARK_MODEL", "qwen2.5-0.5b"),
                               "messages": [{"role": "user", "content": q}],
                               "max_tokens": 32, "temperature": 0.0,
                               "bypass_cache": True},
                         timeout=60)
            body = r.json()
            rag = body.get("_rag", {}) or {}
            if rag.get("used"):
                hits += 1
        except Exception:
            pass
    rate = hits / max(len(queries), 1)
    log(f"RAG hit rate: {hits}/{len(queries)} = {rate*100:.0f}%")
    return rate


def verify_sc_hit_rate(warm_file: str, measure_file: str, n: int = 20) -> float:
    import requests as req
    model = os.getenv("BENCHMARK_MODEL", "qwen2.5-0.5b")
    warm_qs    = Path(warm_file).read_text().strip().splitlines()[:n]
    measure_qs = Path(measure_file).read_text().strip().splitlines()[:n]

    log("Warming cache...")
    for q in warm_qs:
        try:
            req.post(f"{BASE_URL}/v1/chat/completions",
                     json={"model": model,
                           "messages": [{"role": "user", "content": q}],
                           "max_tokens": 32, "temperature": 0.0,
                           "bypass_rag": True},
                     timeout=60)
        except Exception:
            pass

    log("Measuring hit rate...")
    hits = 0
    for q in measure_qs:
        try:
            r = req.post(f"{BASE_URL}/v1/chat/completions",
                         json={"model": model,
                               "messages": [{"role": "user", "content": q}],
                               "max_tokens": 32, "temperature": 0.0,
                               "bypass_rag": True},
                         timeout=60)
            body = r.json()
            cache = body.get("_cache", {}) or {}
            if cache.get("semantic_hit"):
                hits += 1
        except Exception:
            pass

    rate = hits / max(len(measure_qs), 1)
    log(f"SC hit rate: {hits}/{len(measure_qs)} = {rate*100:.0f}%")
    return rate


# ── Check ─────────────────────────────────────────────────────────────────────

def check_query_files(n_samples: int = 3) -> None:
    SYNTHETIC_MARKERS = {
        "What is the main contribution of this paper?",
        "How does the proposed method compare to baselines?",
        "What datasets were used for evaluation?",
        "Explain the difference between supervised and unsupervised",
        "What is the capital city of Australia?",
    }

    groups = [
        ("RAG",          "rag",          ["short", "medium", "long", "very_long"]),
        ("RAG pure-fetch","rag_pure_fetch", ["queries"]),
        ("Cache warm",   "cache",        ["short_warm", "medium_warm"]),
        ("Cache meas",   "cache",        ["short_measure", "medium_measure"]),
        ("LLM-direct",   "llm_direct",   ["short", "medium", "long", "very_long"]),
    ]

    for group_name, subdir, buckets in groups:
        print(f"\n{'─'*60}")
        print(f"  {group_name}  ({subdir}/)")
        print(f"{'─'*60}")
        for bucket in buckets:
            path = QUERIES_DIR / subdir / f"{bucket}.txt"
            if not path.exists():
                print(f"  [{bucket:15s}]  MISSING — run --generate")
                continue

            lines = [l for l in path.read_text().splitlines() if l.strip()]
            tokens = [_token_count(l) for l in lines]
            avg    = sum(tokens) / len(tokens) if tokens else 0
            tmin   = min(tokens) if tokens else 0
            tmax_t = max(tokens) if tokens else 0

            is_synthetic = any(
                any(m in line for m in SYNTHETIC_MARKERS) for line in lines[:20]
            )
            source_tag = "  *** SYNTHETIC FALLBACK ***" if is_synthetic else ""

            unique_ratio = len(set(lines)) / len(lines) if lines else 1.0
            if unique_ratio < 0.5 and not is_synthetic:
                source_tag = f"  (heavily cycled: {unique_ratio:.0%} unique)"

            bucket_key = bucket.replace("_warm", "").replace("_measure", "")
            target = BUCKET_TARGETS.get(bucket_key)
            target_str = f"target={target[0]}–{target[1]}" if target else ""

            print(f"  [{bucket:15s}]  n={len(lines):4d}  avg={avg:5.0f} tok  "
                  f"range={tmin}–{tmax_t}  {target_str}{source_tag}")

            for i, line in enumerate(lines[:n_samples]):
                preview = line[:120] + ("…" if len(line) > 120 else "")
                print(f"    [{i+1}] {preview}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare benchmark query files")
    parser.add_argument("--generate", action="store_true", help="Generate query files from datasets")
    parser.add_argument("--ingest",   action="store_true", help="Download PDFs and ingest into RAG")
    parser.add_argument("--verify",   action="store_true", help="Verify RAG and SC hit rates")
    parser.add_argument("--check",    action="store_true", help="Inspect query files on disk")
    parser.add_argument("--samples",  type=int, default=3,  help="Sample lines to show per bucket (--check)")
    parser.add_argument("--count",    type=int, default=400, help="Queries per bucket")
    parser.add_argument("--n-papers", type=int, default=60,  help="Papers to download and ingest")
    args = parser.parse_args()

    if not any([args.generate, args.ingest, args.verify, args.check]):
        parser.print_help()
        sys.exit(0)

    if args.check:
        check_query_files(args.samples)

    if args.ingest:
        ingest_rag_pdfs(args.n_papers)

    if args.generate:
        log(f"Generating query files (count={args.count} per bucket)...")

        # RAG queries from open_ragbench
        log("\n── RAG (vectara/open_ragbench) ──")
        ragbench_records = load_open_ragbench(args.count * 3)
        rag_buckets = build_rag_queries(ragbench_records, args.count)
        for bucket, queries in rag_buckets.items():
            write_query_file(QUERIES_DIR / "rag" / f"{bucket}.txt", queries)

        # RAG pure-fetch path (bare questions)
        # Use records[count:] to avoid overlap with rag/short.txt which uses records[:count]
        log("\n── RAG pure-fetch (bare questions, no padding) ──")
        pure_fetch_queries = build_rag_pure_fetch_queries(ragbench_records[args.count:], args.count)
        write_query_file(QUERIES_DIR / "rag_pure_fetch" / "queries.txt", pure_fetch_queries)

        # SC queries from QQP
        log("\n── Semantic Cache (QQP) ──")
        qqp_pairs = load_qqp(args.count)
        cache_buckets = build_cache_queries(qqp_pairs, args.count)
        for bucket, files in cache_buckets.items():
            write_query_file(QUERIES_DIR / "cache" / f"{bucket}_warm.txt",    files["warm"])
            write_query_file(QUERIES_DIR / "cache" / f"{bucket}_measure.txt", files["measure"])

        # LLM-direct queries from ShareGPT52K
        log("\n── LLM-direct (RyokoAI/ShareGPT52K) ──")
        sharegpt_records = load_sharegpt(args.count)
        llm_buckets = build_llm_direct_queries(sharegpt_records, args.count)
        for bucket, queries in llm_buckets.items():
            write_query_file(QUERIES_DIR / "llm_direct" / f"{bucket}.txt", queries)

        log(f"\nAll query files written to {QUERIES_DIR.relative_to(KERNEL_ROOT)}/")

    if args.verify:
        log("\n── Verification ──")
        rag_file = str(QUERIES_DIR / "rag" / "short.txt")
        if Path(rag_file).exists():
            verify_rag_hit_rate(rag_file, n=20)
        sc_warm = str(QUERIES_DIR / "cache" / "short_warm.txt")
        sc_meas = str(QUERIES_DIR / "cache" / "short_measure.txt")
        if Path(sc_warm).exists() and Path(sc_meas).exists():
            verify_sc_hit_rate(sc_warm, sc_meas, n=20)


if __name__ == "__main__":
    main()
