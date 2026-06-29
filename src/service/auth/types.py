from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class AuthContext:
    tenant_id: str
    subject: Optional[str] = None
    scopes: Optional[List[str]] = None
    roles: Optional[List[str]] = None
    claims: Optional[Dict[str, Any]] = None
