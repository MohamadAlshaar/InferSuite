import hashlib
import json
from typing import Any, Dict


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_json(payload: Dict[str, Any]) -> str:
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return sha256_text(s)
