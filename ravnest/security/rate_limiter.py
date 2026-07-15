"""
ravnest.security.rate_limiter — Token-bucket rate limiting for the gateway.

Provides both a standalone async ``RateLimiter`` and an aiohttp middleware
factory ``rate_limit()``.

Usage
-----
    from ravnest.security.rate_limiter import RateLimiter, rate_limit

    # Standalone async use
    limiter = RateLimiter(rate=10, burst=20)  # 10 req/s, burst of 20
    allowed = await limiter.consume("client-ip")

    # As aiohttp middleware
    middleware = rate_limit(rate=10, burst=20, key_func=lambda r: r.remote)
    server.add_middleware(middleware)

Algorithm
---------
Token bucket with continuous refill.  Each "bucket" starts full (``burst``
tokens).  Tokens refill at ``rate`` per second.  A request consumes one token.
If the bucket is empty the request is rejected with HTTP 429.

Buckets are lazily created and evicted after ``ttl`` seconds of inactivity to
prevent unbounded memory growth.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class _Bucket:
    """Single token bucket for one key."""

    __slots__ = ("tokens", "last_refill")

    def __init__(self, burst: float):
        self.tokens      = burst
        self.last_refill = time.monotonic()

    def consume(self, rate: float, burst: float, tokens: float = 1.0) -> bool:
        now    = time.monotonic()
        delta  = now - self.last_refill
        self.tokens      = min(burst, self.tokens + delta * rate)
        self.last_refill = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


class RateLimiter:
    """
    Async token-bucket rate limiter.

    Args:
        rate:          Token refill rate (tokens per second).
        burst:         Maximum burst size (bucket capacity).
        ttl:           Seconds of inactivity before a bucket is evicted.
        key:           Default key when ``consume()`` is called without one.
    """

    def __init__(
        self,
        rate:  float = 60.0,
        burst: float = 100.0,
        ttl:   float = 3600.0,
        key:   str   = "default",
    ):
        self._rate    = rate
        self._burst   = burst
        self._ttl     = ttl
        self._default = key
        self._buckets: Dict[str, Tuple[_Bucket, float]] = {}
        self._lock    = asyncio.Lock()

    async def consume(self, key: Optional[str] = None,
                      tokens: float = 1.0) -> bool:
        """
        Try to consume ``tokens`` from the bucket for ``key``.

        Returns True if allowed, False if rate-limited.
        """
        k = key or self._default
        async with self._lock:
            bucket, _ = self._buckets.get(k, (None, 0))
            if bucket is None:
                bucket = _Bucket(self._burst)
            allowed = bucket.consume(self._rate, self._burst, tokens)
            self._buckets[k] = (bucket, time.monotonic())
            return allowed

    def consume_sync(self, key: Optional[str] = None,
                     tokens: float = 1.0) -> bool:
        """Synchronous variant (single-threaded contexts only)."""
        k = key or self._default
        bucket, _ = self._buckets.get(k, (None, 0))
        if bucket is None:
            bucket = _Bucket(self._burst)
        allowed = bucket.consume(self._rate, self._burst, tokens)
        self._buckets[k] = (bucket, time.monotonic())
        return allowed

    async def evict_stale(self) -> int:
        """Remove buckets that have been idle for longer than ``ttl`` seconds."""
        cutoff  = time.monotonic() - self._ttl
        removed = 0
        async with self._lock:
            stale = [k for k, (_, ts) in self._buckets.items() if ts < cutoff]
            for k in stale:
                del self._buckets[k]
                removed += 1
        return removed

    def stats(self) -> Dict[str, Any]:
        """Return current limiter statistics."""
        return {
            "rate":        self._rate,
            "burst":       self._burst,
            "active_keys": len(self._buckets),
        }


class MultiLimiter:
    """
    Multiple independent rate limiters stacked in tiers.

    Common pattern: per-IP limiter (loose) + per-key limiter (strict).

        limiter = MultiLimiter([
            RateLimiter(rate=100, burst=200),   # per-IP (key = IP)
            RateLimiter(rate=10,  burst=20),    # per-API-key
        ])
        allowed = await limiter.consume([ip, api_key])
    """

    def __init__(self, limiters: list[RateLimiter]):
        self._limiters = limiters

    async def consume(self, keys: list[Optional[str]],
                      tokens: float = 1.0) -> bool:
        """
        Consume from all limiters in order.  Returns False if any is exceeded.

        On rejection, no tokens are consumed from subsequent limiters in the
        chain (fail-fast).
        """
        consumed = []
        for limiter, key in zip(self._limiters, keys):
            allowed = await limiter.consume(key, tokens)
            if not allowed:
                return False
            consumed.append((limiter, key, tokens))
        return True


# ── aiohttp middleware ────────────────────────────────────────────────────────

def rate_limit(
    rate:     float                          = 60.0,
    burst:    float                          = 100.0,
    key_func: Optional[Callable]             = None,
    ttl:      float                          = 3600.0,
    limiter:  Optional[RateLimiter]          = None,
) -> Any:
    """
    Return an aiohttp middleware that enforces rate limiting.

    Args:
        rate:      Tokens refilled per second (default 60).
        burst:     Bucket capacity / max burst (default 100).
        key_func:  Callable(request) → str key.  Defaults to client IP.
        ttl:       Idle TTL for bucket eviction (seconds).
        limiter:   Bring-your-own ``RateLimiter`` (overrides rate/burst/ttl).

    The middleware adds ``Retry-After`` and ``X-RateLimit-*`` headers to
    429 responses.
    """
    _limiter  = limiter or RateLimiter(rate=rate, burst=burst, ttl=ttl)
    _key_func = key_func or (lambda req: req.remote or "unknown")

    try:
        from aiohttp import web

        @web.middleware
        async def _middleware(request, handler):
            key = _key_func(request)
            if not await _limiter.consume(key):
                retry_after = max(1, int(1.0 / _limiter._rate))
                logger.warning("[rate_limit] Throttled key=%s", key)
                return web.json_response(
                    {"error": "Rate limit exceeded", "retry_after": retry_after},
                    status=429,
                    headers={
                        "Retry-After":           str(retry_after),
                        "X-RateLimit-Limit":     str(int(_limiter._burst)),
                        "X-RateLimit-Remaining": "0",
                    },
                )
            response = await handler(request)
            return response

    except (ImportError, ModuleNotFoundError):
        async def _middleware(request, handler):
            return await handler(request)

    return _middleware


def rate_limit_by_api_key(
    rate:  float = 60.0,
    burst: float = 100.0,
) -> Any:
    """
    Rate limit keyed on the authenticated API key name (from ``request["auth"]``).

    Must be placed AFTER the ``require_api_key`` middleware in the stack.
    Falls back to client IP if no auth info is present.
    """

    def _key_func(request):
        auth = request.get("auth")
        if auth is not None:
            return f"key:{auth.name}"
        return f"ip:{request.remote or 'unknown'}"

    return rate_limit(rate=rate, burst=burst, key_func=_key_func)
