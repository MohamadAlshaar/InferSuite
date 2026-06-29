from __future__ import annotations

import json
from typing import Any, Optional

from redis.asyncio import Redis


class ExactCache:
    def __init__(self, url: str):
        self._redis = Redis.from_url(url, decode_responses=True)

    async def get(self, key: str) -> Optional[Any]:
        raw = await self._redis.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    async def put(self, key: str, payload: Any, ttl_sec: int) -> None:
        raw = json.dumps(payload, ensure_ascii=False)
        if int(ttl_sec) > 0:
            await self._redis.set(key, raw, ex=int(ttl_sec))
        else:
            await self._redis.set(key, raw)

    async def delete(self, key: str) -> None:
        await self._redis.delete(key)

    async def clear(self) -> None:
        await self._redis.flushdb()

    async def close(self) -> None:
        try:
            await self._redis.aclose()
        except AttributeError:
            await self._redis.close()
