#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)
from sentence_transformers import SentenceTransformer


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_text(text: str, chunk_chars: int, overlap_chars: int) -> List[str]:
    text = " ".join((text or "").split())
    if not text:
        return []

    if len(text) <= chunk_chars:
        return [text]

    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap_chars)
    return chunks


def connect_milvus(uri: str, token: str) -> None:
    kwargs: Dict[str, Any] = {"alias": "default", "uri": uri}
    if token:
        kwargs["token"] = token
    connections.connect(**kwargs)


def ensure_collection(name: str, dim: int, drop_if_exists: bool) -> Collection:
    if utility.has_collection(name):
        if drop_if_exists:
            utility.drop_collection(name)
        else:
            col = Collection(name)
            col.load()
            return col

    fields = [
        FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=128),
        FieldSchema(name="tenant_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="page", dtype=DataType.INT64),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields=fields, description="Benchmark SQuAD RAG corpus")
    col = Collection(name=name, schema=schema)

    col.create_index(
        field_name="embedding",
        index_params={
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200},
        },
    )
    col.load()
    return col


def build_rows(
    corpus_rows: List[Dict[str, Any]],
    model: SentenceTransformer,
    chunk_chars: int,
    overlap_chars: int,
    normalize: bool,
    batch_size: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    all_rows: List[Dict[str, Any]] = []
    tenant_hash_inputs: Dict[str, List[str]] = {"tenantA": [], "tenantB": []}

    texts_to_embed: List[str] = []
    metas: List[Dict[str, Any]] = []

    for row in corpus_rows:
        tenant = str(row["tenant"])
        source = str(row["source"])
        page = int(row.get("page", 1))
        corpus_id = str(row["corpus_id"])
        text = str(row["text"])

        tenant_hash_inputs[tenant].append(corpus_id)
        tenant_hash_inputs[tenant].append(sha256_text(text))

        chunks = chunk_text(text, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
        for idx, chunk in enumerate(chunks):
            chunk_id = sha1_text(f"{tenant}|{corpus_id}|{idx}|{chunk}")[:32]
            pk = sha1_text(f"pk|{tenant}|{corpus_id}|{idx}|{chunk}")[:40]
            metas.append(
                {
                    "pk": pk,
                    "tenant_id": tenant,
                    "source": source,
                    "page": page,
                    "text": chunk,
                    "chunk_id": chunk_id,
                }
            )
            texts_to_embed.append(chunk)

    for start in range(0, len(texts_to_embed), batch_size):
        batch_texts = texts_to_embed[start : start + batch_size]
        batch_meta = metas[start : start + batch_size]
        embs = model.encode(
            batch_texts,
            batch_size=min(batch_size, 64),
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        embs = np.asarray(embs, dtype=np.float32)
        for meta, emb in zip(batch_meta, embs, strict=True):
            row = dict(meta)
            row["embedding"] = emb.tolist()
            all_rows.append(row)

    tenant_versions = {
        tenant: sha1_text("|".join(vals))[:16]
        for tenant, vals in tenant_hash_inputs.items()
    }
    return all_rows, tenant_versions


def insert_rows(col: Collection, rows: List[Dict[str, Any]], insert_batch: int) -> None:
    for start in range(0, len(rows), insert_batch):
        batch = rows[start : start + insert_batch]
        data = [
            [r["pk"] for r in batch],
            [r["tenant_id"] for r in batch],
            [r["source"] for r in batch],
            [r["page"] for r in batch],
            [r["text"] for r in batch],
            [r["chunk_id"] for r in batch],
            [r["embedding"] for r in batch],
        ]
        col.insert(data)
    col.flush()
    col.load()


def write_manifests(manifest_root: Path, collection_name: str, tenant_versions: Dict[str, str], rows: List[Dict[str, Any]]) -> None:
    per_tenant_counts = {"tenantA": 0, "tenantB": 0}
    for row in rows:
        per_tenant_counts[row["tenant_id"]] += 1

    for tenant, kb_version in tenant_versions.items():
        tdir = manifest_root / tenant
        tdir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "tenant": tenant,
            "kb_version": kb_version,
            "collection_name": collection_name,
            "num_chunks": per_tenant_counts.get(tenant, 0),
            "dataset": "squad_benchmark",
        }
        with (tdir / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-jsonl", required=True)
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--collection-name", default="rag_chunks_benchmark")
    parser.add_argument("--milvus-uri", default="http://127.0.0.1:19530")
    parser.add_argument("--milvus-token", default="root:Milvus")
    parser.add_argument("--embed-model-path", required=True)
    parser.add_argument("--chunk-chars", type=int, default=1200)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument("--normalize", type=int, default=1)
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument("--insert-batch-size", type=int, default=256)
    parser.add_argument("--drop-collection", action="store_true")
    args = parser.parse_args()

    corpus_rows = list(read_jsonl(Path(args.corpus_jsonl)))
    if not corpus_rows:
        raise RuntimeError("Empty corpus JSONL")

    model = SentenceTransformer(args.embed_model_path)
    dim = int(model.get_sentence_embedding_dimension())

    connect_milvus(args.milvus_uri, args.milvus_token)
    col = ensure_collection(args.collection_name, dim=dim, drop_if_exists=args.drop_collection)

    rows, tenant_versions = build_rows(
        corpus_rows=corpus_rows,
        model=model,
        chunk_chars=args.chunk_chars,
        overlap_chars=args.chunk_overlap,
        normalize=bool(args.normalize),
        batch_size=args.embed_batch_size,
    )

    insert_rows(col, rows, insert_batch=args.insert_batch_size)
    write_manifests(Path(args.manifest_root), args.collection_name, tenant_versions, rows)

    print(json.dumps(
        {
            "collection_name": args.collection_name,
            "num_input_docs": len(corpus_rows),
            "num_inserted_chunks": len(rows),
            "tenant_versions": tenant_versions,
            "manifest_root": str(Path(args.manifest_root).resolve()),
        },
        indent=2,
        ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
