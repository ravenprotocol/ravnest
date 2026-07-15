"""
ravnest.security.auth — API key authentication for the Ravnest gateway.

Usage
-----
    from ravnest.security.auth import APIKeyStore, require_api_key
    from ravnest.gateway       import GatewayServer, Orchestrator

    store = APIKeyStore()
    key   = store.add_key("my-client", scopes=["generate", "rag"])
    print("API key:", key)   # keep this secret

    # Attach to GatewayServer
    orch   = Orchestrator()
    server = GatewayServer(orch, port=8080)
    server.add_middleware(require_api_key(store, header="X-API-Key"))
    server.run()

    # Client call:
    # curl -H "X-API-Key: <key>" http://localhost:8080/chat ...

Scope enforcement
-----------------
When scopes are set on a key, ``require_api_key`` checks that the
requested route is in the key's scope list.  Route → scope mapping:

    /chat, /v1/chat/completions  → "generate"
    /query                       → "query"
    /rag                         → "rag"
    /pipeline                    → "pipeline"
    /health, /nodes, /           → (no scope required — always allowed)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .crypto import generate_token, hash_key

logger = logging.getLogger(__name__)

# Route → required scope
_ROUTE_SCOPES: Dict[str, str] = {
    "/chat":                 "generate",
    "/v1/chat/completions":  "generate",
    "/query":                "query",
    "/rag":                  "rag",
    "/pipeline":             "pipeline",
}


@dataclass
class KeyInfo:
    """Metadata stored alongside a hashed API key."""
    name:        str
    key_hash:    str
    scopes:      Set[str]         = field(default_factory=set)
    created_at:  float            = field(default_factory=time.time)
    last_used:   Optional[float]  = None
    use_count:   int              = 0
    metadata:    Dict[str, Any]   = field(default_factory=dict)
    enabled:     bool             = True

    def has_scope(self, scope: str) -> bool:
        """True if this key grants the requested scope (or has wildcard "*")."""
        return not self.scopes or scope in self.scopes or "*" in self.scopes

    def to_dict(self) -> dict:
        return {
            "name":       self.name,
            "scopes":     sorted(self.scopes),
            "created_at": self.created_at,
            "last_used":  self.last_used,
            "use_count":  self.use_count,
            "enabled":    self.enabled,
            "metadata":   self.metadata,
        }


class APIKeyStore:
    """
    In-memory store of hashed API keys.

    API keys are never stored in plaintext — only their SHA-256 hash is kept.
    The caller receives the plaintext key once at creation time.

    Thread-safe for asyncio (single event loop); wrap with a threading.Lock
    if used from multiple threads.

    Args:
        allow_anonymous:  If True, requests without an API key are accepted
                          (useful for development / open instances).
    """

    def __init__(self, allow_anonymous: bool = False):
        self._keys:            Dict[str, KeyInfo] = {}  # key_hash → KeyInfo
        self._allow_anonymous: bool               = allow_anonymous

    # ── key management ────────────────────────────────────────────────────

    def add_key(
        self,
        name:     str,
        scopes:   Optional[List[str]] = None,
        metadata: Optional[dict]      = None,
        key:      Optional[str]       = None,
    ) -> str:
        """
        Create a new API key and return the plaintext token.

        Args:
            name:     Human-readable label (e.g. "production-client").
            scopes:   List of allowed route scopes.  Empty = all scopes.
            metadata: Arbitrary extra info stored with the key.
            key:      Override the generated key (useful in tests).

        Returns:
            The plaintext API key — store it securely, it cannot be recovered.
        """
        token     = key or generate_token("rav")
        kh        = hash_key(token)
        info      = KeyInfo(
            name      = name,
            key_hash  = kh,
            scopes    = set(scopes or []),
            metadata  = metadata or {},
        )
        self._keys[kh] = info
        logger.info("[APIKeyStore] Added key '%s' (hash=%s…)", name, kh[:8])
        return token

    def revoke_key(self, token: str) -> bool:
        """Revoke (delete) a key by its plaintext value. Returns True if found."""
        kh = hash_key(token)
        if kh in self._keys:
            del self._keys[kh]
            logger.info("[APIKeyStore] Revoked key (hash=%s…)", kh[:8])
            return True
        return False

    def disable_key(self, token: str) -> bool:
        """Disable a key without deleting it. Returns True if found."""
        info = self._lookup(token)
        if info:
            info.enabled = False
            return True
        return False

    def enable_key(self, token: str) -> bool:
        info = self._lookup(token)
        if info:
            info.enabled = True
            return True
        return False

    def list_keys(self) -> List[dict]:
        """Return metadata for all keys (no plaintext values)."""
        return [info.to_dict() for info in self._keys.values()]

    # ── verification ──────────────────────────────────────────────────────

    def verify(self, token: str) -> Optional[KeyInfo]:
        """
        Verify a plaintext API key.

        Returns the ``KeyInfo`` on success, or ``None`` if the key is invalid,
        revoked, or disabled.  Updates use statistics on success.
        """
        if not token:
            return None
        info = self._lookup(token)
        if info is None or not info.enabled:
            return None
        info.last_used = time.time()
        info.use_count += 1
        return info

    @property
    def allow_anonymous(self) -> bool:
        return self._allow_anonymous

    # ── private ───────────────────────────────────────────────────────────

    def _lookup(self, token: str) -> Optional[KeyInfo]:
        return self._keys.get(hash_key(token))


# ── aiohttp middleware ────────────────────────────────────────────────────────

def require_api_key(
    store:  APIKeyStore,
    header: str = "X-API-Key",
) -> Any:
    """
    Return an aiohttp middleware that enforces API key authentication.

    Requests to ``/health``, ``/nodes``, and ``/`` are always allowed
    regardless of key presence or scope.

    Authenticated key info is stored in ``request["auth"]`` for downstream
    handlers.

    Args:
        store:  The ``APIKeyStore`` to validate keys against.
        header: HTTP header name carrying the API key.
    """
    _OPEN_ROUTES = {"/health", "/nodes", "/"}

    try:
        from aiohttp import web

        @web.middleware
        async def _middleware(request, handler):
            # Always allow health/info endpoints
            if request.path in _OPEN_ROUTES:
                request["auth"] = None
                return await handler(request)

            token = request.headers.get(header, "").strip()

            if not token:
                if store.allow_anonymous:
                    request["auth"] = None
                    return await handler(request)
                return web.json_response(
                    {"error": f"Missing {header} header",
                     "hint":  "Provide a valid API key"},
                    status=401,
                )

            info = store.verify(token)
            if info is None:
                logger.warning("[auth] Invalid API key from %s", request.remote)
                return web.json_response(
                    {"error": "Invalid or revoked API key"}, status=401
                )

            # Scope check
            required_scope = _ROUTE_SCOPES.get(request.path)
            if required_scope and not info.has_scope(required_scope):
                logger.warning(
                    "[auth] Key '%s' lacks scope '%s'", info.name, required_scope
                )
                return web.json_response(
                    {"error": f"Key does not have scope '{required_scope}'"},
                    status=403,
                )

            request["auth"] = info
            return await handler(request)

    except (ImportError, ModuleNotFoundError):
        async def _middleware(request, handler):
            return await handler(request)

    return _middleware


def require_hmac(
    key:           str,
    header:        str = "X-Ravnest-Signature",
    max_age:       int = 300,
) -> Any:
    """
    Return an aiohttp middleware that verifies HMAC-signed request bodies.

    The sender must:
    1. Add ``_ts`` (Unix timestamp) to the JSON body.
    2. Compute HMAC-SHA256 over the JSON body (keys sorted, no spaces).
    3. Set the ``X-Ravnest-Signature`` header to the base64url signature.

    Use :func:`ravnest.security.crypto.sign_message_dict` to produce
    compliant signed bodies.
    """
    from .crypto import verify_message_dict

    try:
        from aiohttp import web

        @web.middleware
        async def _middleware(request, handler):
            sig = request.headers.get(header, "").strip()
            if not sig:
                return web.json_response(
                    {"error": f"Missing {header} header"}, status=401
                )
            try:
                import json
                body = await request.json()
            except Exception:
                return web.json_response({"error": "Invalid JSON body"}, status=400)

            if not verify_message_dict({**body, "_sig": sig}, key,
                                       max_age_seconds=max_age):
                logger.warning("[hmac] Signature verification failed from %s",
                               request.remote)
                return web.json_response(
                    {"error": "Invalid or expired request signature"}, status=401
                )
            return await handler(request)

    except (ImportError, ModuleNotFoundError):
        async def _middleware(request, handler):
            return await handler(request)

    return _middleware
