#!/usr/bin/env python3
"""
Ingest vectara/open_ragbench arXiv corpus directly into Milvus + SeaweedFS.
Bypasses PDF download — uses pre-extracted JSON from HuggingFace.

Usage (inside the ingest job pod):
    python3 scripts/ingest_ragbench_to_milvus.py \
        --n-papers 200 \
        --bge-model-path /app/fastapi_runtime_assets/models/bge-base-en-v1.5 \
        --milvus-uri http://milvus.llm-service.svc.cluster.local:19530 \
        --milvus-token root:Milvus \
        --collection rag_chunks_seaweed_v2 \
        --s3-endpoint-url http://seaweed-s3.llm-service.svc.cluster.local:8333 \
        --s3-access-key-id llmbenchadmin \
        --s3-secret-access-key llmbenchsecretkey123 \
        --s3-bucket llm-rag-store \
        --tenant tenantA \
        --drop-existing
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import boto3
import numpy as np
from botocore.config import Config
from botocore.exceptions import ClientError
from pymilvus import DataType, MilvusClient
from sentence_transformers import SentenceTransformer

HF_BASE = "https://huggingface.co/datasets/vectara/open_ragbench/resolve/main/pdf/arxiv"
HF_QUERIES_URL = f"{HF_BASE}/queries.json"
HF_QRELS_URL   = f"{HF_BASE}/qrels.json"

# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[ragbench-ingest] {msg}", flush=True)

def ok(msg: str) -> None:
    print(f"[ok] {msg}", flush=True)

def warn(msg: str) -> None:
    print(f"[warn] {msg}", flush=True)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    step = max(1, chunk_size - overlap)
    chunks = []
    for start in range(0, len(text), step):
        piece = text[start : start + chunk_size].strip()
        if piece:
            chunks.append(piece)
        if start + chunk_size >= len(text):
            break
    return chunks


# ── HuggingFace corpus listing ────────────────────────────────────────────────

def _required_paper_ids(queries_dir: str) -> set:
    """
    Match benchmark query files against open_ragbench qrels to find which
    arXiv papers the queries actually reference. These are always included first.
    """
    log("Matching benchmark queries to required papers...")
    try:
        with urllib.request.urlopen(HF_QUERIES_URL, timeout=30) as r:
            hf_queries = json.load(r)
        with urllib.request.urlopen(HF_QRELS_URL, timeout=30) as r:
            hf_qrels = json.load(r)
    except Exception as e:
        warn(f"Could not load queries/qrels from HuggingFace: {e} — skipping required matching")
        return set()

    # Build reverse map: query_text → arxiv_id (stripped of version)
    text_to_doc: Dict[str, str] = {}
    for uuid, q in hf_queries.items():
        qrel = hf_qrels.get(uuid)
        if qrel:
            text_to_doc[q["query"].strip()] = re.sub(r"v\d+$", "", qrel["doc_id"])

    # Load benchmark query files from the container
    import glob
    required: set = set()
    pattern = os.path.join(queries_dir, "**", "*.txt")
    for path in glob.glob(pattern, recursive=True):
        try:
            lines = open(path).read().splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Strip context padding added to medium/long bucket queries
            clean = re.split(r"\s{2,}Context:", line)[0].strip()
            doc = text_to_doc.get(clean)
            if doc:
                required.add(doc)

    log(f"Required papers (matched from benchmark queries): {len(required)}")
    return required


def list_corpus_files(n_papers: Optional[int], queries_dir: str = "",
                      n_filler: Optional[int] = 50) -> List[Tuple[str, str]]:
    """
    Return list of (arxiv_id, hf_filename).
    Required papers (matched from benchmark queries) are always included in full.
    Filler papers are capped at n_filler (default 50) unless n_papers is set,
    in which case n_papers is the total cap (legacy behaviour).
    """
    log("Listing corpus files from HuggingFace...")
    from huggingface_hub import list_repo_files

    all_corpus: List[Tuple[str, str]] = []
    seen: set = set()
    for fname in list_repo_files("vectara/open_ragbench", repo_type="dataset"):
        m = re.search(r"corpus/(\d+\.\d+)(v\d+)\.json$", fname)
        if not m:
            continue
        arxiv_id = m.group(1)
        if arxiv_id in seen:
            continue
        seen.add(arxiv_id)
        all_corpus.append((arxiv_id, fname.split("/")[-1]))

    log(f"Total corpus papers available: {len(all_corpus)}")

    required_ids = _required_paper_ids(queries_dir) if queries_dir else set()

    required = [(a, f) for a, f in all_corpus if a in required_ids]
    filler   = [(a, f) for a, f in all_corpus if a not in required_ids]

    import random
    random.seed(42)
    random.shuffle(filler)

    if n_papers is not None:
        # Legacy: cap total
        results = (required + filler)[:n_papers]
        n_actual_filler = len(results) - len(required)
    else:
        # New: always include all required, cap filler independently
        filler_cap = n_filler if n_filler is not None else len(filler)
        results = required + filler[:filler_cap]
        n_actual_filler = len(filler[:filler_cap])

    log(f"Selected {len(results)} papers: {len(required)} required + {n_actual_filler} filler")
    return results


def fetch_corpus_json(hf_filename: str) -> Optional[Dict[str, Any]]:
    """Download one corpus JSON from HuggingFace."""
    url = f"{HF_BASE}/corpus/{hf_filename}"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.load(r)
    except Exception as e:
        warn(f"Failed to fetch {hf_filename}: {e}")
        return None


# ── S3 / SeaweedFS ────────────────────────────────────────────────────────────

def build_s3(endpoint: str, key_id: str, secret: str) -> Any:
    session = boto3.session.Session()
    return session.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
        region_name=os.getenv("RAG_OBJECT_STORE_REGION", "eu-west-2"),
        use_ssl=endpoint.startswith("https"),
        verify=False,
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )


def ensure_bucket(s3: Any, bucket: str) -> None:
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError:
        s3.create_bucket(Bucket=bucket)
        ok(f"Created bucket: {bucket}")


def s3_put(s3: Any, bucket: str, key: str, body: str) -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=body.encode())


# ── Milvus ────────────────────────────────────────────────────────────────────

def ensure_collection(client: MilvusClient, collection: str, dim: int, drop: bool) -> None:
    if client.has_collection(collection):
        if drop:
            warn(f"Dropping existing collection: {collection}")
            client.drop_collection(collection_name=collection)
        else:
            ok(f"Collection '{collection}' already exists — appending")
            return

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("pk",             DataType.VARCHAR, is_primary=True, max_length=128)
    schema.add_field("tenant_id",      DataType.VARCHAR, max_length=128)
    schema.add_field("source",         DataType.VARCHAR, max_length=1024)
    schema.add_field("page",           DataType.INT64)
    schema.add_field("chunk_id",       DataType.VARCHAR, max_length=256)
    schema.add_field("object_key",     DataType.VARCHAR, max_length=2048)
    schema.add_field("pdf_object_key", DataType.VARCHAR, max_length=2048)
    schema.add_field("text_sha256",    DataType.VARCHAR, max_length=64)
    schema.add_field("text",           DataType.VARCHAR, max_length=65535)
    schema.add_field("embedding",      DataType.FLOAT_VECTOR, dim=dim)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="embedding", metric_type="COSINE",
                           index_type="HNSW", params={"M": 16, "efConstruction": 200})

    client.create_collection(collection_name=collection, schema=schema,
                             index_params=index_params)
    ok(f"Created collection: {collection}")


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_batch(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    return model.encode(texts, normalize_embeddings=True,
                        batch_size=32, show_progress_bar=False)


# ── Main ingest ───────────────────────────────────────────────────────────────

def ingest_paper(
    data: Dict[str, Any],
    arxiv_id: str,
    tenant_id: str,
    chunk_size: int,
    overlap: int,
    model: SentenceTransformer,
    s3: Any,
    bucket: str,
    milvus: MilvusClient,
    collection: str,
) -> int:
    from concurrent.futures import ThreadPoolExecutor as _TPE
    title = data.get("title", arxiv_id).strip().replace("\n", " ")
    sections = data.get("sections", [])
    full_text = f"Title: {title}\n\n" + "\n\n".join(
        s.get("text", "") for s in sections
    )

    # Build all chunk rows first (no S3 yet)
    rows = []
    for sec in sections:
        sec_id = sec.get("section_id", 0)
        sec_text = sec.get("text", "").strip()
        if not sec_text:
            continue
        for i, chunk in enumerate(_split_text(sec_text, chunk_size, overlap)):
            rows.append({
                "text": chunk,
                "sha": _sha256(chunk),
                "chunk_key": f"ragbench/{arxiv_id}_s{sec_id}_c{i}.txt",
                "source": f"{arxiv_id}.json",
                "page": int(sec_id),
                "chunk_id": f"{arxiv_id}_s{sec_id}_c{i}",
                "pk_base": f"{arxiv_id}_{sec_id}_{i}",
            })

    if not rows:
        return 0

    # Parallel S3 uploads (paper + all chunks)
    def _upload(key: str, body: str) -> str:
        try:
            s3_put(s3, bucket, key, body)
            return key
        except Exception:
            return ""

    paper_key = f"ragbench/{arxiv_id}.txt"
    upload_tasks = [(paper_key, full_text)] + [(r["chunk_key"], r["text"]) for r in rows]
    with _TPE(max_workers=8) as pool:
        futs = {pool.submit(_upload, k, v): k for k, v in upload_tasks}
        results = {futs[f]: f.result() for f in as_completed(futs)}

    paper_key = results.get(paper_key, "")
    for r in rows:
        r["chunk_key"] = results.get(r["chunk_key"], "")

    # Embed all chunks at once
    embeddings = embed_batch(model, [r["text"] for r in rows])

    milvus_rows = [
        {
            "pk":             _sha256(r["pk_base"])[:64],
            "tenant_id":      tenant_id,
            "source":         r["source"],
            "page":           r["page"],
            "chunk_id":       r["chunk_id"],
            "object_key":     r["chunk_key"],
            "pdf_object_key": paper_key,
            "text_sha256":    r["sha"],
            "text":           r["text"][:65000],
            "embedding":      emb.tolist(),
        }
        for r, emb in zip(rows, embeddings)
    ]

    milvus.insert(collection_name=collection, data=milvus_rows)
    return len(milvus_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-papers",         type=int,  default=None,
                    help="Total paper cap (required + filler). Overrides --n-filler if set.")
    ap.add_argument("--n-filler",         type=int,  default=50,
                    help="Max filler papers (non-query-referenced) added after required papers.")
    ap.add_argument("--tenant",           default="tenantA")
    ap.add_argument("--bge-model-path",   required=True)
    ap.add_argument("--milvus-uri",       required=True)
    ap.add_argument("--milvus-token",     default="root:Milvus")
    ap.add_argument("--collection",       default="rag_chunks_seaweed_v2")
    ap.add_argument("--s3-endpoint-url",  required=True)
    ap.add_argument("--s3-access-key-id", required=True)
    ap.add_argument("--s3-secret-access-key", required=True)
    ap.add_argument("--s3-bucket",        default="llm-rag-store")
    ap.add_argument("--chunk-size",       type=int, default=3000)
    ap.add_argument("--chunk-overlap",    type=int, default=150)
    ap.add_argument("--drop-existing",    action="store_true")
    ap.add_argument("--download-workers", type=int, default=8)
    ap.add_argument("--queries-dir",      default="/app/benchmark_queries",
                    help="Root benchmark_queries dir — all *.txt files are scanned recursively for required-paper matching")
    args = ap.parse_args()

    log(f"tenant={args.tenant} n_filler={args.n_filler} n_papers={args.n_papers} collection={args.collection}")

    # Load BGE model
    log(f"Loading BGE model from {args.bge_model_path}")
    model = SentenceTransformer(args.bge_model_path, device="cpu")
    dim = model.get_sentence_embedding_dimension()
    ok(f"BGE model loaded (dim={dim})")

    # Connect Milvus
    milvus = MilvusClient(uri=args.milvus_uri, token=args.milvus_token)
    ensure_collection(milvus, args.collection, dim, args.drop_existing)

    # Connect S3
    s3 = build_s3(args.s3_endpoint_url, args.s3_access_key_id, args.s3_secret_access_key)
    ensure_bucket(s3, args.s3_bucket)

    # List corpus files — required papers first, then up to n_filler filler papers.
    # --n-papers overrides this if set (legacy behaviour).
    n_filler = args.n_filler if args.n_papers is None else None
    n_papers = args.n_papers  # None when using n_filler mode
    corpus_files = list_corpus_files(n_papers, queries_dir=args.queries_dir, n_filler=n_filler)

    # Download corpus JSONs concurrently
    log(f"Downloading {len(corpus_files)} corpus JSONs (workers={args.download_workers})...")
    papers: List[Tuple[str, Dict]] = []
    with ThreadPoolExecutor(max_workers=args.download_workers) as pool:
        futures = {pool.submit(fetch_corpus_json, fname): arxiv_id
                   for arxiv_id, fname in corpus_files}
        for future in as_completed(futures):
            arxiv_id = futures[future]
            data = future.result()
            if data:
                papers.append((arxiv_id, data))
    ok(f"Downloaded {len(papers)}/{len(corpus_files)} papers")

    # Ingest
    total_chunks = 0
    t0 = time.time()
    for i, (arxiv_id, data) in enumerate(papers, 1):
        n = ingest_paper(
            data, arxiv_id, args.tenant,
            args.chunk_size, args.chunk_overlap,
            model, s3, args.s3_bucket, milvus, args.collection,
        )
        total_chunks += n
        elapsed = time.time() - t0
        rate = total_chunks / elapsed if elapsed > 0 else 0
        eta = (len(papers) - i) * (elapsed / i)
        if i % 10 == 0 or i == len(papers):
            log(f"  {i}/{len(papers)} papers | {total_chunks} chunks | "
                f"{rate*60:.0f} chunks/min | ETA {eta/60:.1f} min")

    ok(f"Ingested {len(papers)} papers → {total_chunks} chunks into '{args.collection}'")

    # Write manifest
    manifest = {
        "tenant_id": args.tenant,
        "source": "vectara/open_ragbench",
        "n_papers": len(papers),
        "n_chunks": total_chunks,
        "collection": args.collection,
        "kb_version": "ragbench-v1",
    }
    log(f"Manifest: {json.dumps(manifest)}")


if __name__ == "__main__":
    main()
