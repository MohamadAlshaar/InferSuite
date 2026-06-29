import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

from src.service.utils.hashing import sha256_text


class LocalRAG:
    def __init__(self, rag_store_dir: str, embed_model_path: str, top_k: int):
        self.rag_store_dir = Path(rag_store_dir)
        self.embed_model_path = str(embed_model_path)
        self.top_k = int(top_k)

        manifest_path = self.rag_store_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"RAG manifest not found: {manifest_path}. Run scripts/build_rag_index.py first."
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.kb_version = self.manifest.get("kb_version", "unknown")

        Settings.embed_model = HuggingFaceEmbedding(
            model_name=str(self.embed_model_path),
            device="cpu",
            embed_batch_size=8,
        )

        vector_store = FaissVectorStore.from_persist_dir(str(self.rag_store_dir))
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=str(self.rag_store_dir),
        )
        self.index = load_index_from_storage(storage_context=storage_context)
        self.retriever = self.index.as_retriever(similarity_top_k=self.top_k)

    def retrieve(self, query: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        t0 = time.perf_counter()
        nodes = self.retriever.retrieve(query)
        retrieve_ms = (time.perf_counter() - t0) * 1000.0

        out: List[Dict[str, Any]] = []
        for rank, node in enumerate(nodes, start=1):
            metadata = dict(getattr(node, "metadata", {}) or {})
            text = node.get_text().strip()
            out.append(
                {
                    "rank": rank,
                    "score": float(getattr(node, "score", 0.0) or 0.0),
                    "text": text,
                    "metadata": metadata,
                }
            )

        # Fingerprint retrieved context so cache keys are correct when KB/retrieval changes.
        fp_parts: List[str] = []
        for item in out:
            meta = item["metadata"]
            source = meta.get("file_name") or meta.get("filename") or meta.get("source") or "unknown"
            page = str(meta.get("page_label") or meta.get("page") or "?")
            fp_parts.append(f"{source}:{page}:{sha256_text(item['text'])}")
        context_fingerprint = sha256_text("|".join(fp_parts)) if fp_parts else sha256_text("no_context")

        meta = {
            "retrieve_ms": retrieve_ms,
            "num_chunks": len(out),
            # For the FAISS IndexFlatL2 build: score is L2 distance -> lower is better
            "top_score": float(out[0]["score"]) if out else 0.0,
            "context_fingerprint": context_fingerprint,
        }
        return out, meta

    def format_context(self, items: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for item in items:
            meta = item["metadata"]
            source = meta.get("file_name") or meta.get("filename") or meta.get("source") or "unknown"
            page = meta.get("page_label") or meta.get("page") or "?"
            parts.append(f"[Source {item['rank']}: {source}, page {page}]\n{item['text']}")
        return "\n\n".join(parts)
