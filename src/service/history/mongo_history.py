from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional

from pymongo import MongoClient
from pymongo.errors import PyMongoError

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="history-write")


class MessageHistoryWriter:
    def __init__(
        self,
        *,
        mongo_uri: str,
        mongo_db: str = "llm_service",
        mongo_collection: str = "messages",
        connect_timeout_ms: int = 3000,
    ):
        self._client = MongoClient(
            mongo_uri,
            serverSelectionTimeoutMS=connect_timeout_ms,
            connectTimeoutMS=connect_timeout_ms,
        )
        self._col = self._client[mongo_db][mongo_collection]
        self._col.create_index("session_id")
        self._col.create_index("tenant_id")
        self._col.create_index("ts")

    def _write(
        self,
        *,
        session_id: str,
        tenant_id: str,
        user_content: str,
        assistant_content: str,
        path: Optional[str],
        latency_ms: float,
        cache_hit: bool,
    ) -> None:
        ts = datetime.now(timezone.utc)
        try:
            self._col.insert_many(
                [
                    {
                        "session_id": session_id,
                        "tenant_id": tenant_id,
                        "role": "user",
                        "content": user_content,
                        "ts": ts,
                    },
                    {
                        "session_id": session_id,
                        "tenant_id": tenant_id,
                        "role": "assistant",
                        "content": assistant_content,
                        "ts": ts,
                        "latency_ms": latency_ms,
                        "cache_hit": cache_hit,
                        "path": path,
                    },
                ]
            )
        except PyMongoError as e:
            logger.warning("history write failed: %s", e)

    def write_turn_async(
        self,
        *,
        session_id: str,
        tenant_id: str,
        user_content: str,
        assistant_content: str,
        path: Optional[str] = None,
        latency_ms: float = 0.0,
        cache_hit: bool = False,
    ) -> None:
        loop = asyncio.get_event_loop()
        loop.run_in_executor(
            _executor,
            lambda: self._write(
                session_id=session_id,
                tenant_id=tenant_id,
                user_content=user_content,
                assistant_content=assistant_content,
                path=path,
                latency_ms=latency_ms,
                cache_hit=cache_hit,
            ),
        )

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass
