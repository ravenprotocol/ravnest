"""
ravnest.security — Security & trust primitives for the Ravnest mesh.

Components
----------
crypto
    HMAC-SHA256 signing / verification for inter-node messages.
    No external dependencies.

auth
    API key store and aiohttp gateway middleware for authentication
    and per-route scope enforcement.

rate_limiter
    Token-bucket rate limiter with aiohttp middleware.

sandbox
    ToolSandbox (allowlist + timeout) and AgentSandbox (subprocess
    isolation) for constraining agent tool execution.

tls
    Self-signed CA + node cert generation and mTLS SSLContext builders.
    Requires ``pip install cryptography``.

Quick example — secure gateway in 10 lines
------------------------------------------
    from ravnest.gateway          import GatewayServer, Orchestrator
    from ravnest.security.auth    import APIKeyStore, require_api_key
    from ravnest.security.rate_limiter import rate_limit_by_api_key

    store = APIKeyStore()
    key   = store.add_key("my-client", scopes=["generate", "rag"])

    orch   = Orchestrator()
    server = GatewayServer(orch, port=8080)
    server.add_middleware(require_api_key(store))
    server.add_middleware(rate_limit_by_api_key(rate=30, burst=50))
    server.run()
"""

from .crypto import (
    generate_key,
    generate_token,
    sign,
    verify,
    hash_key,
    make_timestamp,
    check_timestamp,
    sign_message_dict,
    verify_message_dict,
)

from .auth import (
    KeyInfo,
    APIKeyStore,
    require_api_key,
    require_hmac,
)

from .rate_limiter import (
    RateLimiter,
    MultiLimiter,
    rate_limit,
    rate_limit_by_api_key,
)

from .sandbox import (
    ToolSandbox,
    AgentSandbox,
    InputValidator,
    ToolNotAllowed,
    ToolLimitExceeded,
    ToolTimeout,
)

__all__ = [
    # ── crypto ────────────────────────────────────────────────────────────
    "generate_key",
    "generate_token",
    "sign",
    "verify",
    "hash_key",
    "make_timestamp",
    "check_timestamp",
    "sign_message_dict",
    "verify_message_dict",
    # ── auth ──────────────────────────────────────────────────────────────
    "KeyInfo",
    "APIKeyStore",
    "require_api_key",
    "require_hmac",
    # ── rate limiting ─────────────────────────────────────────────────────
    "RateLimiter",
    "MultiLimiter",
    "rate_limit",
    "rate_limit_by_api_key",
    # ── sandbox ───────────────────────────────────────────────────────────
    "ToolSandbox",
    "AgentSandbox",
    "InputValidator",
    "ToolNotAllowed",
    "ToolLimitExceeded",
    "ToolTimeout",
    # ── tls ───────────────────────────────────────────────────────────────
    # (CertBundle etc. loaded lazily — import from ravnest.security.tls directly)
]
