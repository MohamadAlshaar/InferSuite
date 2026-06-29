from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

import httpx
import jwt  # PyJWT
from fastapi import HTTPException, status

from src.service.auth.types import AuthContext


class KeycloakJWTVerifier:
    """
    Verifies JWTs using Keycloak's JWKS endpoint.
    - Caches JWKS in-memory for jwks_cache_ttl_sec
    - Verifies signature + issuer + exp + (optional) audience
    """

    def __init__(
        self,
        *,
        issuer: str,
        jwks_url: str,
        audience: Optional[str],
        tenant_claim: str,
        jwks_cache_ttl_sec: int = 300,
        allowed_algs: Optional[list[str]] = None,
        http_timeout_s: float = 10.0,
    ):
        self.issuer = issuer.rstrip("/")
        self.jwks_url = jwks_url
        self.audience = audience
        self.tenant_claim = tenant_claim
        self.jwks_cache_ttl_sec = int(jwks_cache_ttl_sec)
        self.allowed_algs = allowed_algs or ["RS256", "RS384", "RS512"]
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(http_timeout_s))

        self._jwks: Optional[Dict[str, Any]] = None
        self._jwks_fetched_at: float = 0.0

    async def close(self) -> None:
        await self._http.aclose()

    async def _get_jwks(self) -> Dict[str, Any]:
        now = time.time()
        if self._jwks is not None and (now - self._jwks_fetched_at) < self.jwks_cache_ttl_sec:
            return self._jwks

        try:
            r = await self._http.get(self.jwks_url)
            r.raise_for_status()
            jwks = r.json()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to fetch JWKS from Keycloak: {e}",
            )

        if not isinstance(jwks, dict) or "keys" not in jwks:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Invalid JWKS payload from Keycloak",
            )

        self._jwks = jwks
        self._jwks_fetched_at = now
        return jwks

    @staticmethod
    def _extract_scopes(claims: Dict[str, Any]) -> Optional[list[str]]:
        scope = claims.get("scope")
        if isinstance(scope, str):
            return [s for s in scope.split() if s]
        return None

    def _extract_roles(self, claims: Dict[str, Any]) -> Optional[list[str]]:
        # Common Keycloak layout: realm_access.roles
        ra = claims.get("realm_access")
        if isinstance(ra, dict):
            roles = ra.get("roles")
            if isinstance(roles, list):
                return [str(r) for r in roles]
        return None

    def _extract_tenant_id(self, claims: Dict[str, Any]) -> str:
        v = claims.get(self.tenant_claim)

        # allow nested path like "tenant.id" if we want later
        if v is None and "." in self.tenant_claim:
            cur: Any = claims
            for part in self.tenant_claim.split("."):
                if not isinstance(cur, dict):
                    cur = None
                    break
                cur = cur.get(part)
            v = cur

        if v is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Token missing tenant claim '{self.tenant_claim}'",
            )
        if not isinstance(v, str) or not v.strip():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Invalid tenant claim '{self.tenant_claim}'",
            )
        return v.strip()

    async def verify_bearer_token(self, token: str) -> AuthContext:
        # 1) read header to find kid/alg
        try:
            header = jwt.get_unverified_header(token)
        except Exception:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid JWT header")

        alg = header.get("alg")
        kid = header.get("kid")
        if alg not in self.allowed_algs:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Disallowed JWT alg '{alg}'")
        if not kid:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="JWT missing kid")

        # 2) fetch jwks + find matching key
        jwks = await self._get_jwks()
        keys = jwks.get("keys", [])
        jwk = None
        for k in keys:
            if isinstance(k, dict) and k.get("kid") == kid:
                jwk = k
                break
        if jwk is None:
            # force refresh once in case Keycloak rotated keys
            self._jwks = None
            jwks = await self._get_jwks()
            for k in jwks.get("keys", []):
                if isinstance(k, dict) and k.get("kid") == kid:
                    jwk = k
                    break
        if jwk is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No matching JWKS key for kid")

        # 3) build public key and verify token
        try:
            public_key = jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(jwk))
            options = {"verify_aud": self.audience is not None}
            claims = jwt.decode(
                token,
                key=public_key,
                algorithms=[alg],
                issuer=self.issuer,
                audience=self.audience if self.audience is not None else None,
                options=options,
            )
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="JWT expired")
        except jwt.InvalidIssuerError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid JWT issuer")
        except jwt.InvalidAudienceError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid JWT audience")
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"JWT verification failed: {e}")

        if not isinstance(claims, dict):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid JWT claims")

        tenant_id = self._extract_tenant_id(claims)
        subject = claims.get("sub") if isinstance(claims.get("sub"), str) else None

        return AuthContext(
            tenant_id=tenant_id,
            subject=subject,
            scopes=self._extract_scopes(claims),
            roles=self._extract_roles(claims),
            claims=claims,
        )
