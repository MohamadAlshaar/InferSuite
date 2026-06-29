from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from pymilvus import Collection, connections
from pymongo import ASCENDING, MongoClient
from pymongo.collection import Collection as MongoCollection
from pymongo.errors import PyMongoError
from sentence_transformers import SentenceTransformer

from src.service.cache.semantic_schema import (
    SEMANTIC_CACHE_ACCEPT_KEY_FIELD,
    SEMANTIC_CACHE_VECTOR_FIELD,
    get_semantic_cache_insert_payload,
    get_semantic_cache_output_fields,
    open_semantic_cache_collection,
)


def _parse_milvus_uri(uri: str) -> Tuple[str, bool]:
    uri = (uri or "").strip()
    if not uri:
        return "http://127.0.0.1:19530", False

    if "://" not in uri:
        return f"http://{uri}", False

    u = urlparse(uri)
    secure = u.scheme.lower() == "https"
    normalized = f"{u.scheme}://{u.hostname}:{u.port or 19530}"
    return normalized, secure


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    return None


class SemanticCache:
    """
    Direct semantic cache implementation:
      - vector store: Milvus
      - scalar store: MongoDB

    Runtime behavior in this version:
      - connects only to existing Milvus collection
      - does not create Milvus collections at app startup
      - non-vector payload truth lives in MongoDB
      - Milvus stores only pk, accept_key, embedding
    """

    def __init__(
        self,
        *,
        enabled: bool,
        similarity_threshold: float,
        ttl_sec: int,
        embed_model: str,
        embed_model_path: str,
        mongo_uri: str,
        mongo_db: str,
        mongo_collection: str,
        mongo_connect_timeout_ms: int,
        milvus_uri: str,
        milvus_user: str,
        milvus_password: str,
        milvus_secure: bool,
        milvus_collection: str,
        vector_dim: int,
        top_k: int = 5,
        normalize_embeddings: bool = True,
    ):
        self.enabled = bool(enabled)
        self.ttl_sec = int(ttl_sec)
        self.similarity_threshold = float(similarity_threshold)
        self.mongo_uri = str(mongo_uri or "").strip()
        self.mongo_db_name = str(mongo_db or "").strip()
        self.mongo_collection_name = str(mongo_collection or "").strip()
        self.mongo_connect_timeout_ms = int(mongo_connect_timeout_ms)
        self.milvus_collection_name = str(milvus_collection or "").strip()
        self.vector_dim = int(vector_dim)
        self.top_k = int(top_k)
        self.normalize_embeddings = bool(normalize_embeddings)

        self._model: Optional[SentenceTransformer] = None
        self._collection: Optional[Collection] = None
        self._mongo_client: Optional[MongoClient] = None
        self._mongo_collection: Optional[MongoCollection] = None
        self._init_error: Optional[str] = None
        self._conn_alias = f"semcache_{uuid.uuid4().hex[:8]}"
        self._collection_load_attempted = False
        self._collection_loaded = False

        if not self.enabled:
            return

        try:
            model_source = (
                embed_model_path.strip()
                if embed_model_path.strip()
                else embed_model.strip()
            )
            if not model_source:
                raise ValueError("missing SEM_CACHE_EMBED_MODEL / SEM_CACHE_EMBED_MODEL_PATH")

            if not self.mongo_uri:
                raise ValueError("missing SEM_CACHE_MONGO_URI")
            if not self.mongo_db_name:
                raise ValueError("missing SEM_CACHE_MONGO_DB")
            if not self.mongo_collection_name:
                raise ValueError("missing SEM_CACHE_MONGO_COLLECTION")
            if not self.milvus_collection_name:
                raise ValueError("missing SEM_CACHE_MILVUS_COLLECTION")
            if self.vector_dim <= 0:
                raise ValueError("invalid SEM_CACHE_VECTOR_DIM")

            self._model = SentenceTransformer(model_source)

            self._init_mongo()

            normalized_uri, secure_from_uri = _parse_milvus_uri(milvus_uri)
            connections.connect(
                alias=self._conn_alias,
                uri=normalized_uri,
                user=milvus_user or "",
                password=milvus_password or "",
                secure=bool(milvus_secure) or secure_from_uri,
            )

            self._collection = self._open_existing_milvus_collection(
                collection_name=self.milvus_collection_name
            )

        except Exception as e:
            self.enabled = False
            self._init_error = str(e)

    @property
    def init_error(self) -> Optional[str]:
        return self._init_error

    def close(self) -> None:
        if self._mongo_client is not None:
            try:
                self._mongo_client.close()
            except Exception:
                pass

        try:
            connections.disconnect(self._conn_alias)
        except Exception:
            pass

    def _init_mongo(self) -> None:
        self._mongo_client = MongoClient(
            self.mongo_uri,
            serverSelectionTimeoutMS=self.mongo_connect_timeout_ms,
            tz_aware=True,
            appname="llm-service-kernel-semcache",
        )

        self._mongo_client.admin.command("ping")

        db = self._mongo_client[self.mongo_db_name]
        self._mongo_collection = db[self.mongo_collection_name]

        self._mongo_collection.create_index(
            [("accept_key", ASCENDING)],
            name="idx_semantic_accept_key",
        )
        self._mongo_collection.create_index(
            [("expires_at", ASCENDING)],
            name="idx_semantic_expires_at_ttl",
            expireAfterSeconds=0,
        )

    def _open_existing_milvus_collection(self, collection_name: str) -> Collection:
        return open_semantic_cache_collection(
            alias=self._conn_alias,
            collection_name=collection_name,
            expected_vector_dim=self.vector_dim,
        )

    def _try_load_collection(self) -> bool:
        if not self.enabled or self._collection is None:
            return False

        if self._collection_loaded:
            return True

        if self._collection_load_attempted:
            return False

        self._collection_load_attempted = True

        try:
            self._collection.load(timeout=10)
            self._collection_loaded = True
            return True
        except Exception as e:
            self.enabled = False
            self._init_error = f"milvus_load_error: {e}"
            return False

    def _embed(self, text: str) -> List[float]:
        if self._model is None:
            raise RuntimeError("Semantic cache model is not initialized")
        vec = self._model.encode(
            [str(text)],
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )[0]
        return vec.tolist()

    @staticmethod
    def _extract_hits(search_result: Any) -> List[Any]:
        if isinstance(search_result, list) and search_result:
            if isinstance(search_result[0], list):
                return search_result[0]
            return search_result
        return []

    @staticmethod
    def _extract_hit_id_and_score(hit: Any) -> Tuple[Optional[str], float]:
        if isinstance(hit, dict):
            entry_id = hit.get("id") or hit.get("pk")
            entity = hit.get("entity") or {}
            if not entry_id and isinstance(entity, dict):
                entry_id = entity.get("pk")
            score = hit.get("score", hit.get("distance", 0.0))
            return (str(entry_id) if entry_id else None, float(score or 0.0))

        entry_id = getattr(hit, "id", None)
        score = getattr(hit, "score", getattr(hit, "distance", 0.0))

        entity = getattr(hit, "entity", None)
        if entry_id is None and entity is not None:
            try:
                entry_id = entity.get("pk")
            except Exception:
                entry_id = None

        return (str(entry_id) if entry_id else None, float(score or 0.0))

    def get(self, cache_text: str, expected_accept_key: str):
        """
        Returns:
          (payload or None, reason or None)

        reasons:
          disabled, not_found, below_threshold, ttl_expired,
          mongo_error, vector_error, bad_json
        """
        self._last_get_timings: dict = {"embed_ms": 0.0, "milvus_ms": 0.0, "mongo_ms": 0.0}

        if not self.enabled or self._collection is None or self._mongo_collection is None:
            return None, "disabled"

        if not self._try_load_collection():
            return None, "disabled"

        try:
            _te0 = time.perf_counter()
            query_vec = self._embed(cache_text)
            self._last_get_timings["embed_ms"] = (time.perf_counter() - _te0) * 1000.0
        except Exception:
            return None, "vector_error"

        try:
            _tm0 = time.perf_counter()
            search_result = self._collection.search(
                data=[query_vec],
                anns_field=SEMANTIC_CACHE_VECTOR_FIELD,
                param={"metric_type": "COSINE", "params": {"ef": 64}},
                limit=max(1, self.top_k),
                expr=f'{SEMANTIC_CACHE_ACCEPT_KEY_FIELD} == "{expected_accept_key}"',
                output_fields=get_semantic_cache_output_fields(),
                timeout=5,
            )
        except Exception:
            return None, "vector_error"

        self._last_get_timings["milvus_ms"] = (time.perf_counter() - _tm0) * 1000.0

        hits = self._extract_hits(search_result)
        if not hits:
            return None, "not_found"

        _, top_score = self._extract_hit_id_and_score(hits[0])
        if top_score < self.similarity_threshold:
            return None, "below_threshold"

        now = _utc_now()

        try:
            _tmo0 = time.perf_counter()
            for hit in hits:
                entry_id, score = self._extract_hit_id_and_score(hit)
                if score < self.similarity_threshold:
                    continue
                if not entry_id:
                    continue

                doc = self._mongo_collection.find_one(
                    {"_id": entry_id, "accept_key": expected_accept_key},
                    projection={
                        "_id": 1,
                        "payload_json": 1,
                        "expires_at": 1,
                    },
                )

                if doc is None:
                    continue

                expires_at = _normalize_dt(doc.get("expires_at"))
                if expires_at is not None and expires_at <= now:
                    continue

                try:
                    payload = json.loads(str(doc.get("payload_json") or ""))
                except json.JSONDecodeError:
                    return None, "bad_json"

                if not isinstance(payload, dict):
                    return None, "bad_json"

                return payload, None

        except PyMongoError:
            return None, "mongo_error"
        except Exception:
            return None, "mongo_error"
        finally:
            self._last_get_timings["mongo_ms"] = (time.perf_counter() - _tmo0) * 1000.0

        return None, "ttl_expired"

    def put(self, cache_text: str, payload: Dict[str, Any], accept_key: str) -> None:
        if not self.enabled or self._collection is None or self._mongo_collection is None:
            return

        if not self._try_load_collection():
            return

        try:
            vector = self._embed(cache_text)
        except Exception as e:
            print(f"SemanticCache put embed error: {e}", flush=True)
            return

        entry_id = _sha1_text(f"{accept_key}|{cache_text}|{time.time_ns()}")[:40]
        created_at = _utc_now()
        expires_at = created_at + timedelta(seconds=int(self.ttl_sec)) if self.ttl_sec > 0 else None

        try:
            payload_json = json.dumps(payload, ensure_ascii=False)
        except Exception as e:
            print(f"SemanticCache payload json encode error: {e}", flush=True)
            return

        try:
            doc = {
                "_id": entry_id,
                "accept_key": str(accept_key),
                "query_text": str(cache_text),
                "payload_json": payload_json,
                "created_at": created_at,
                "ttl_sec": int(self.ttl_sec),
                "expires_at": expires_at,
            }
            self._mongo_collection.replace_one(
                {"_id": entry_id},
                doc,
                upsert=True,
            )
        except Exception as e:
            print(f"SemanticCache mongo put error: {e}", flush=True)
            return

        try:
            self._collection.insert(
                get_semantic_cache_insert_payload(
                    entry_id=entry_id,
                    accept_key=accept_key,
                    vector=vector,
                ),
                timeout=5,
            )
            # No per-request flush — Milvus auto-flushes on its own schedule.
            # Calling flush() per insert forces a ~1500ms segment seal operation.
        except Exception as e:
            print(f"SemanticCache milvus put error: {e}", flush=True)
            try:
                self._mongo_collection.delete_one({"_id": entry_id})
            except Exception:
                pass
