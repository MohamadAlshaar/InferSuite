from fastapi import HTTPException, Request, status

from src.service.auth.types import AuthContext


async def get_auth_context(request: Request) -> AuthContext:
    settings = request.app.state.settings
    verifier = request.app.state.auth_verifier  # may be None

    # M2-lite: if auth is disabled, take tenant from header, else fallback to DEV_TENANT_ID
    if not settings.AUTH_ENABLED:
        hdr = request.headers.get("X-Tenant-Id", "").strip()
        tenant = hdr if hdr else settings.DEV_TENANT_ID
        return AuthContext(tenant_id=tenant)

    #  Keycloak verification
    if verifier is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Auth enabled but verifier not configured",
        )

    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Bearer token")

    token = auth[len("Bearer ") :].strip()
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Empty Bearer token")

    return await verifier.verify_bearer_token(token)
