from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, Optional

from pymilvus import MilvusClient

from src.service.embeddings.bge import BGEEmbedder
from src.service.rag.milvus_rag import MilvusTenantRAG
from src.service.rag.types import IRAG


class TenantRAGRouter:
    """
    RAG provider:
      - milvus: a single Milvus collection, tenant isolation via filter tenant_id == "<tenant>"
      - local:  per-tenant local stores under RAG_STORE_ROOT_DIR/<tenant> (kept as fallback)

    Important:
      - Do NOT eagerly load Milvus collections at startup.
      - Do NOT import local RAG dependencies unless local backend is actually used.
      - Keep startup lightweight and non-blocking.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        backend: str,
        top_k: int,
        kb_version_fallback: str,
        manifest_root_dir: str,
        # local backend
        store_root_dir: str,
        embed_model_path_local: str = "",
        fallback_tenant: Optional[str] = None,
        max_loaded: int = 64,
        # milvus backend
        milvus_uri: str = "",
        milvus_token: str = "",
        milvus_collection: str = "rag_chunks",
        milvus_vector_field: str = "embedding",
        milvus_tenant_field: str = "tenant_id",
        bge_model_path: str = "",
        bge_device: str = "cpu",
        bge_normalize: bool = True,
    ):
        self.enabled = enabled
        self.backend = backend
        self.top_k = int(top_k)
        self.kb_version_fallback = kb_version_fallback
        self.manifest_root_dir = Path(manifest_root_dir)

        self.store_root = Path(store_root_dir)
        self.embed_model_path_local = embed_model_path_local
        self.fallback_tenant = fallback_tenant
        self.max_loaded = int(max_loaded)

        self._lock = threading.Lock()
        self._rags: Dict[str, IRAG] = {}

        self._milvus = None
        self._milvus_collection = milvus_collection
        self._milvus_vector_field = milvus_vector_field
        self._milvus_tenant_field = milvus_tenant_field

        self._embedder = None
        self._init_error: Optional[str] = None

        if not self.enabled:
            return

        if self.backend == "milvus":
            try:
                print("RAG INIT: before MilvusClient", flush=True)
                self._milvus = MilvusClient(uri=milvus_uri, token=milvus_token)
                print("RAG INIT: after MilvusClient", flush=True)

                print(f"RAG INIT: before BGEEmbedder path={bge_model_path}", flush=True)
                self._embedder = BGEEmbedder(
                    bge_model_path,
                    device=bge_device,
                    normalize=bge_normalize,
                    batch_size=32,
                )
                print("RAG INIT: after BGEEmbedder", flush=True)

                print(
                    f"RAG INIT: skipping eager load_collection for {self._milvus_collection}",
                    flush=True,
                )
            except Exception as e:
                self._init_error = str(e)
                self.enabled = False
                self._milvus = None
                self._embedder = None
                print(f"RAG INIT ERROR: {e}", flush=True)

    @property
    def init_error(self) -> Optional[str]:
        return self._init_error

    def _load_kb_version(self, tenant_id: str) -> str:
        mp = self.manifest_root_dir / tenant_id / "manifest.json"
        if not mp.exists():
            return self.kb_version_fallback
        try:
            j = json.loads(mp.read_text(encoding="utf-8"))
            return str(j.get("kb_version", self.kb_version_fallback))
        except Exception:
            return self.kb_version_fallback

    def get(self, tenant_id: str) -> Optional[IRAG]:
        if not self.enabled:
            return None

        tid = tenant_id.strip()
        if not tid and self.fallback_tenant:
            tid = self.fallback_tenant

        with self._lock:
            if tid in self._rags:
                return self._rags[tid]

        rag: Optional[IRAG] = None
        kb_version = self._load_kb_version(tid)

        if self.backend == "milvus":
            if self._milvus is None or self._embedder is None:
                return None

            r = MilvusTenantRAG(
                tenant_id=tid,
                milvus=self._milvus,
                collection=self._milvus_collection,
                embedder=self._embedder,
                top_k=self.top_k,
                vector_field=self._milvus_vector_field,
                tenant_field=self._milvus_tenant_field,
            )
            r.kb_version = kb_version
            rag = r
        else:
            try:
                from src.service.rag.local_rag import LocalRAG
            except Exception as e:
                self._init_error = f"local_rag_import_error: {e}"
                return None

            rag_dir = self.store_root / tid
            if not (rag_dir / "manifest.json").exists():
                if self.fallback_tenant and tid != self.fallback_tenant:
                    return self.get(self.fallback_tenant)
                return None

            r = LocalRAG(
                rag_store_dir=str(rag_dir),
                embed_model_path=self.embed_model_path_local,
                top_k=self.top_k,
            )
            rag = r

        if rag is None:
            return None

        with self._lock:
            if len(self._rags) >= self.max_loaded:
                self._rags.pop(next(iter(self._rags.keys())))
            self._rags[tid] = rag
        return rag
