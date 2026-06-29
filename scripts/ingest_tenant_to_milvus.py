#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import boto3
import numpy as np
from botocore.config import Config
from botocore.exceptions import ClientError, ConnectionClosedError
from pypdf import PdfReader
from pymilvus import DataType, MilvusClient
from sentence_transformers import SentenceTransformer


def _env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_verify(raw: str) -> bool | str:
    value = (raw or "").strip()
    if value.lower() in {"1", "true", "yes", "on"}:
        return True
    if value.lower() in {"0", "false", "no", "off", ""}:
        return False
    return value


def _join_key(prefix: str, key: str) -> str:
    clean_key = str(key or "").strip().lstrip("/")
    clean_prefix = str(prefix or "").strip().strip("/")
    if not clean_prefix:
        return clean_key
    if not clean_key:
        return clean_prefix
    return f"{clean_prefix}/{clean_key}"


def build_s3_client(
    *,
    endpoint_url: str,
    access_key_id: str,
    secret_access_key: str,
    region: str,
    use_ssl: bool,
    verify: bool | str,
):
    session = boto3.session.Session()
    return session.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name=region,
        use_ssl=use_ssl,
        verify=verify,
        config=Config(
            region_name=region,
            signature_version="s3v4",
            s3={
                "addressing_style": "path",
                "payload_signing_enabled": False,
            },
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
            retries={"max_attempts": 3, "mode": "standard"},
            tcp_keepalive=True,
            user_agent_extra="llm-service-kernel-rag-ingest",
        ),
    )


def ensure_bucket(client: Any, bucket: str) -> None:
    """
    SeaweedFS-friendly bucket check.

    Avoid head_bucket() here because some S3-compatible backends can behave
    inconsistently on HEAD even when normal list/get/put operations work.
    """
    try:
        resp = client.list_buckets()
        names = {b.get("Name") for b in resp.get("Buckets", []) if b.get("Name")}
        if bucket in names:
            return
    except Exception:
        pass

    try:
        client.create_bucket(Bucket=bucket)
        return
    except ClientError as exc:
        status = int(exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if status in {200, 409} or code in {"BucketAlreadyOwnedByYou", "BucketAlreadyExists"}:
            return
        raise
    except ConnectionClosedError:
        resp = client.list_buckets()
        names = {b.get("Name") for b in resp.get("Buckets", []) if b.get("Name")}
        if bucket in names:
            return
        raise


def put_json_object(client: Any, *, bucket: str, key: str, payload: Dict[str, Any]) -> None:
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=raw,
        ContentType="application/json",
    )


def put_file_object(client: Any, *, bucket: str, key: str, file_path: Path) -> None:
    content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    with file_path.open("rb") as fh:
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=fh,
            ContentType=content_type,
        )


def compute_kb_version(pdf_files: List[Path], docs_dir: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(pdf_files):
        stat = p.stat()
        h.update(str(p.relative_to(docs_dir)).encode("utf-8"))
        h.update(str(stat.st_size).encode("utf-8"))
        h.update(str(int(stat.st_mtime)).encode("utf-8"))
    return h.hexdigest()[:16]


def _clean_text(text: str) -> str:
    text = (text or "").replace("\x00", " ")
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    text = _clean_text(text)
    if not text:
        return []

    if chunk_size <= 0:
        return [text]

    chunk_overlap = max(0, min(chunk_overlap, chunk_size - 1 if chunk_size > 1 else 0))
    step = max(1, chunk_size - chunk_overlap)

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(n, start + chunk_size)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start += step

    return chunks


def extract_pdf_chunks(
    docs_dir: Path,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for pdf_path in sorted(docs_dir.rglob("*.pdf")):
        rel_pdf = pdf_path.relative_to(docs_dir).as_posix()

        try:
            reader = PdfReader(str(pdf_path))
        except Exception as exc:
            print(f"[warn] failed to open PDF {pdf_path}: {exc}")
            continue

        for page_idx, page in enumerate(reader.pages, start=1):
            try:
                raw_text = page.extract_text() or ""
            except Exception as exc:
                print(f"[warn] failed to extract page {page_idx} from {pdf_path}: {exc}")
                continue

            page_text = _clean_text(raw_text)
            if not page_text:
                continue

            page_chunks = _split_text(
                page_text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            for chunk_idx, chunk_text in enumerate(page_chunks, start=1):
                out.append(
                    {
                        "source": rel_pdf,
                        "page": page_idx,
                        "chunk_ordinal": chunk_idx,
                        "text": chunk_text,
                    }
                )

    return out


def _embed_batch(
    model: SentenceTransformer,
    texts: List[str],
    *,
    normalize: bool,
    batch_size: int,
) -> List[List[float]]:
    if not texts:
        return []
    v = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return v.astype(np.float32).tolist()


def _existing_field_names(client: MilvusClient, collection: str) -> Set[str]:
    desc = client.describe_collection(collection_name=collection)
    fields = desc.get("fields", []) or []
    out: Set[str] = set()
    for f in fields:
        name = f.get("name")
        if name:
            out.add(str(name))
    return out


def ensure_collection(
    client: MilvusClient,
    *,
    collection: str,
    dim: int,
    metric_type: str,
    use_partition_key: bool,
    num_partitions: int,
    partition_key_isolation: bool,
    drop_existing: bool,
) -> None:
    required_fields = {
        "pk",
        "tenant_id",
        "source",
        "page",
        "chunk_id",
        "object_key",
        "pdf_object_key",
        "text_sha256",
        "embedding",
    }

    if client.has_collection(collection):
        if drop_existing:
            print(f"[warn] dropping existing collection: {collection}")
            client.drop_collection(collection_name=collection)
        else:
            existing = _existing_field_names(client, collection)
            missing = sorted(required_fields - existing)
            if missing:
                raise SystemExit(
                    f"existing collection '{collection}' does not match SeaweedFS schema; "
                    f"missing fields: {missing}. Use a new --collection or pass --drop-existing."
                )
            return

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field(field_name="pk", datatype=DataType.VARCHAR, is_primary=True, max_length=128)

    schema.add_field(
        field_name="tenant_id",
        datatype=DataType.VARCHAR,
        max_length=128,
        is_partition_key=bool(use_partition_key),
    )
    schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=1024)
    schema.add_field(field_name="page", datatype=DataType.INT64)
    schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="object_key", datatype=DataType.VARCHAR, max_length=2048)
    schema.add_field(field_name="pdf_object_key", datatype=DataType.VARCHAR, max_length=2048)
    schema.add_field(field_name="text_sha256", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=int(dim))

    if use_partition_key:
        client.create_collection(
            collection_name=collection,
            schema=schema,
            num_partitions=int(num_partitions),
            partition_key_isolation=bool(partition_key_isolation),
        )
    else:
        client.create_collection(collection_name=collection, schema=schema)

    try:
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            index_name="emb_hnsw",
            metric_type=metric_type,
            params={"M": 16, "efConstruction": 200},
        )
        client.create_index(collection_name=collection, index_params=index_params, sync=True)
    except Exception as e:
        print(f"[warn] create_index failed (collection still usable with FLAT): {e}")

    try:
        client.load_collection(collection_name=collection, replica_number=1)
    except Exception as e:
        print(f"[warn] load_collection failed (still usable): {e}")


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--tenant", required=True)
    ap.add_argument("--docs-dir", required=True)
    ap.add_argument("--manifest-root", required=True)

    ap.add_argument("--milvus-uri", default=os.getenv("MILVUS_URI", "http://127.0.0.1:19530"))
    ap.add_argument("--milvus-token", default=os.getenv("MILVUS_TOKEN", "root:Milvus"))
    ap.add_argument("--collection", default=os.getenv("MILVUS_COLLECTION", "rag_chunks_seaweed_v2"))
    ap.add_argument("--metric", default=os.getenv("MILVUS_METRIC_TYPE", "COSINE").upper())

    ap.add_argument("--bge-model-path", default=os.getenv("BGE_MODEL_PATH", "").strip())
    ap.add_argument("--device", default=os.getenv("BGE_DEVICE", "cpu"))
    ap.add_argument("--normalize", type=int, default=int(os.getenv("BGE_NORMALIZE", "1")))
    ap.add_argument("--batch-embed", type=int, default=int(os.getenv("BGE_BATCH", "32")))

    ap.add_argument("--chunk-size", type=int, default=int(os.getenv("RAG_CHUNK_SIZE", "800")))
    ap.add_argument("--chunk-overlap", type=int, default=int(os.getenv("RAG_CHUNK_OVERLAP", "120")))
    ap.add_argument("--text-max-len", type=int, default=int(os.getenv("RAG_OBJECT_TEXT_MAX_LEN", "4096")))
    ap.add_argument("--insert-batch", type=int, default=int(os.getenv("RAG_INGEST_BATCH", "128")))

    ap.add_argument("--no-partition-key", action="store_true")
    ap.add_argument("--num-partitions", type=int, default=int(os.getenv("MILVUS_NUM_PARTITIONS", "64")))
    ap.add_argument("--partition-key-isolation", type=int, default=int(os.getenv("MILVUS_PARTITION_KEY_ISOLATION", "1")))
    ap.add_argument("--drop-existing", action="store_true")

    ap.add_argument("--s3-endpoint-url", default=os.getenv("RAG_OBJECT_STORE_ENDPOINT_URL", "http://127.0.0.1:8333"))
    ap.add_argument("--s3-access-key-id", default=os.getenv("RAG_OBJECT_STORE_ACCESS_KEY_ID", "").strip())
    ap.add_argument("--s3-secret-access-key", default=os.getenv("RAG_OBJECT_STORE_SECRET_ACCESS_KEY", "").strip())
    ap.add_argument("--s3-bucket", default=os.getenv("RAG_OBJECT_STORE_BUCKET", "llm-rag-store").strip())
    ap.add_argument("--s3-region", default=os.getenv("RAG_OBJECT_STORE_REGION", "us-east-1").strip())
    ap.add_argument("--s3-use-ssl", type=int, default=int(os.getenv("RAG_OBJECT_STORE_USE_SSL", "0")))
    ap.add_argument("--s3-verify", default=os.getenv("RAG_OBJECT_STORE_VERIFY", "false"))
    ap.add_argument("--s3-base-prefix", default=os.getenv("RAG_OBJECT_STORE_PREFIX", "rag").strip())
    ap.add_argument("--s3-chunks-subprefix", default=os.getenv("RAG_OBJECT_STORE_CHUNKS_SUBPREFIX", "chunks").strip())
    ap.add_argument("--s3-pdfs-subprefix", default=os.getenv("RAG_OBJECT_STORE_PDFS_SUBPREFIX", "pdfs").strip())

    args = ap.parse_args()

    tenant = args.tenant.strip()
    docs_dir = Path(args.docs_dir)
    if not docs_dir.exists():
        raise SystemExit(f"docs dir does not exist: {docs_dir}")

    bge_path = args.bge_model_path
    if not bge_path:
        raise SystemExit("Missing --bge-model-path (or set BGE_MODEL_PATH env var)")
    bge_dir = Path(bge_path)
    if not bge_dir.exists():
        raise SystemExit(f"BGE model path does not exist: {bge_dir}")

    if not args.s3_access_key_id or not args.s3_secret_access_key:
        raise SystemExit(
            "Missing SeaweedFS S3 credentials. Set RAG_OBJECT_STORE_ACCESS_KEY_ID and "
            "RAG_OBJECT_STORE_SECRET_ACCESS_KEY or pass --s3-access-key-id / --s3-secret-access-key."
        )

    metric = str(args.metric).upper()

    pdf_files = sorted(docs_dir.rglob("*.pdf"))
    if not pdf_files:
        raise SystemExit(f"no PDFs found in {docs_dir}")

    kb_version = compute_kb_version(pdf_files, docs_dir)
    print(f"[ingest] tenant={tenant} pdfs={len(pdf_files)} kb_version={kb_version}")

    chunks = extract_pdf_chunks(
        docs_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    if not chunks:
        raise SystemExit(f"no extractable text chunks found in {docs_dir}")

    print(f"[ingest] extracted chunks={len(chunks)}")

    model = SentenceTransformer(str(bge_dir), device=args.device)
    dim = int(model.get_sentence_embedding_dimension())
    normalize = bool(args.normalize)

    s3_client = build_s3_client(
        endpoint_url=args.s3_endpoint_url,
        access_key_id=args.s3_access_key_id,
        secret_access_key=args.s3_secret_access_key,
        region=args.s3_region,
        use_ssl=bool(args.s3_use_ssl),
        verify=_parse_verify(args.s3_verify),
    )
    ensure_bucket(s3_client, args.s3_bucket)

    client = MilvusClient(uri=args.milvus_uri, token=args.milvus_token)

    ensure_collection(
        client,
        collection=args.collection,
        dim=dim,
        metric_type=metric,
        use_partition_key=(not args.no_partition_key),
        num_partitions=int(args.num_partitions),
        partition_key_isolation=bool(args.partition_key_isolation),
        drop_existing=bool(args.drop_existing),
    )

    pdf_object_keys_by_source: Dict[str, str] = {}
    uploaded_pdfs = 0

    for pdf_path in pdf_files:
        rel_pdf = pdf_path.relative_to(docs_dir).as_posix()
        pdf_relative_key = _join_key(
            f"{args.s3_pdfs_subprefix}/{tenant}/{kb_version}",
            rel_pdf,
        )
        pdf_full_key = _join_key(args.s3_base_prefix, pdf_relative_key)

        put_file_object(
            s3_client,
            bucket=args.s3_bucket,
            key=pdf_full_key,
            file_path=pdf_path,
        )
        uploaded_pdfs += 1

        pdf_object_keys_by_source[pdf_path.name] = pdf_relative_key
        pdf_object_keys_by_source[rel_pdf] = pdf_relative_key

    rows: List[Dict[str, Any]] = []
    texts: List[str] = []
    inserted = 0
    uploaded_chunks = 0

    for item in chunks:
        text = (item.get("text") or "").strip()
        if not text:
            continue

        if args.text_max_len > 0:
            text = text[: args.text_max_len]

        src = str(item["source"])
        page = int(item["page"])
        pdf_object_key = pdf_object_keys_by_source.get(src, "")

        text_sha256 = _sha256_text(text)
        chunk_id = hashlib.sha256(f"{tenant}|{src}|{page}|{text}".encode("utf-8")).hexdigest()[:32]
        source_hash = hashlib.sha256(str(src).encode("utf-8")).hexdigest()[:16]

        chunk_relative_key = f"{args.s3_chunks_subprefix}/{tenant}/{kb_version}/{source_hash}/p{page}/{chunk_id}.json"
        chunk_full_key = _join_key(args.s3_base_prefix, chunk_relative_key)

        chunk_payload = {
            "tenant_id": tenant,
            "source": src,
            "page": int(page),
            "chunk_id": chunk_id,
            "text": text,
            "text_sha256": text_sha256,
            "kb_version": kb_version,
            "pdf_object_key": pdf_object_key,
        }
        put_json_object(
            s3_client,
            bucket=args.s3_bucket,
            key=chunk_full_key,
            payload=chunk_payload,
        )
        uploaded_chunks += 1

        rows.append(
            {
                "pk": chunk_id,
                "tenant_id": tenant,
                "source": src,
                "page": int(page),
                "chunk_id": chunk_id,
                "object_key": chunk_relative_key,
                "pdf_object_key": pdf_object_key,
                "text_sha256": text_sha256,
                "text": text[:65535],
            }
        )
        texts.append(text)

        if len(rows) >= args.insert_batch:
            vecs = _embed_batch(model, texts, normalize=normalize, batch_size=args.batch_embed)
            for r, v in zip(rows, vecs):
                r["embedding"] = v
            client.insert(collection_name=args.collection, data=rows)
            inserted += len(rows)
            rows.clear()
            texts.clear()

    if rows:
        vecs = _embed_batch(model, texts, normalize=normalize, batch_size=args.batch_embed)
        for r, v in zip(rows, vecs):
            r["embedding"] = v
        client.insert(collection_name=args.collection, data=rows)
        inserted += len(rows)

    print(f"[ok] uploaded pdf objects={uploaded_pdfs} to bucket={args.s3_bucket}")
    print(f"[ok] uploaded chunk objects={uploaded_chunks} to bucket={args.s3_bucket}")
    print(f"[ok] inserted rows={inserted} into collection={args.collection}")

    manifest_dir = Path(args.manifest_root) / tenant
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "tenant": tenant,
        "docs_dir": str(docs_dir),
        "kb_version": kb_version,
        "collection": args.collection,
        "metric": metric,
        "dim": dim,
        "chunk_size": int(args.chunk_size),
        "chunk_overlap": int(args.chunk_overlap),
        "bge_model_path": str(bge_dir),
        "normalize": normalize,
        "inserted_rows": inserted,
        "uploaded_pdfs": uploaded_pdfs,
        "uploaded_chunk_objects": uploaded_chunks,
        "partition_key": (not args.no_partition_key),
        "num_partitions": int(args.num_partitions) if not args.no_partition_key else None,
        "object_store": {
            "type": "seaweedfs_s3",
            "endpoint_url": args.s3_endpoint_url,
            "bucket": args.s3_bucket,
            "base_prefix": args.s3_base_prefix,
            "chunks_subprefix": args.s3_chunks_subprefix,
            "pdfs_subprefix": args.s3_pdfs_subprefix,
        },
        "collection_fields": [
            "tenant_id",
            "source",
            "page",
            "chunk_id",
            "object_key",
            "pdf_object_key",
            "text_sha256",
            "embedding",
        ],
    }
    (manifest_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[ok] wrote manifest: {manifest_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
