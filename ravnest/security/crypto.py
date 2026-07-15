"""
ravnest.security.crypto — HMAC-SHA256 signing for inter-node messages.

No external dependencies — stdlib only (hmac, hashlib, secrets, base64).

Usage
-----
    from ravnest.security.crypto import generate_key, sign, verify, hash_key

    # Shared secret (store in env-var / secrets manager — never in code)
    key = generate_key()          # "abc123..." (64-char hex string)

    # Sign a request body
    body  = b'{"prompt": "hello"}'
    sig   = sign(body, key)
    valid = verify(body, key, sig)  # True

    # Hash an API key before storing it
    stored = hash_key(key)
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import time
from typing import Optional, Tuple


def generate_key(length: int = 32) -> str:
    """Return a cryptographically random hex key of ``length`` bytes."""
    return secrets.token_hex(length)


def sign(payload: bytes, key: str) -> str:
    """
    Compute an HMAC-SHA256 signature for ``payload`` using ``key``.

    Args:
        payload: Raw bytes to sign (e.g. JSON-encoded request body).
        key:     Hex-encoded secret key (as returned by ``generate_key``).

    Returns:
        Base64url-encoded signature string.
    """
    raw_key = bytes.fromhex(key)
    mac     = hmac.new(raw_key, payload, hashlib.sha256)
    return base64.urlsafe_b64encode(mac.digest()).decode()


def verify(payload: bytes, key: str, signature: str) -> bool:
    """
    Verify a signature produced by :func:`sign`.

    Uses ``hmac.compare_digest`` to prevent timing attacks.
    Returns False (rather than raising) if the signature is malformed.
    """
    try:
        expected = sign(payload, key)
        return hmac.compare_digest(expected, signature)
    except Exception:
        return False


def hash_key(key: str) -> str:
    """
    One-way hash of an API key for safe storage.

    The stored hash can be used to look up the key without revealing
    the plaintext.  Not suitable as a password hash (use bcrypt/argon2
    for that) — appropriate for high-entropy random API keys.
    """
    return hashlib.sha256(key.encode()).hexdigest()


def make_timestamp() -> str:
    """Return the current UTC time as a decimal string (Unix epoch)."""
    return str(int(time.time()))


def check_timestamp(ts: str, max_age_seconds: int = 300) -> bool:
    """
    Return True if ``ts`` (Unix epoch string) is within ``max_age_seconds``
    of the current time.

    Prevents replay attacks when used alongside message signing.
    """
    try:
        t = int(ts)
        return abs(time.time() - t) <= max_age_seconds
    except (ValueError, TypeError):
        return False


def sign_message_dict(body: dict, key: str) -> Tuple[dict, str]:
    """
    Sign a dict payload and return (signed_dict, signature).

    Adds ``_ts`` (timestamp) to the dict before signing so that the
    receiver can detect replayed requests with ``check_timestamp``.

    The full body including ``_ts`` is JSON-encoded and signed.
    The signature is stored in ``_sig`` for convenience.

    Returns both the modified dict and the signature string.
    """
    import json
    body = dict(body)
    body["_ts"] = make_timestamp()
    payload     = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    sig         = sign(payload, key)
    body["_sig"] = sig
    return body, sig


def verify_message_dict(body: dict, key: str,
                        max_age_seconds: int = 300) -> bool:
    """
    Verify a dict that was signed with :func:`sign_message_dict`.

    Extracts and removes ``_sig``, checks the timestamp, then
    recomputes the HMAC over the remaining fields.

    Returns True only when signature is valid AND timestamp is fresh.
    """
    import json
    body = dict(body)
    sig  = body.pop("_sig", None)
    if sig is None:
        return False
    ts = body.get("_ts")
    if not check_timestamp(str(ts) if ts is not None else "", max_age_seconds):
        return False
    payload = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return verify(payload, key, sig)


def generate_token(prefix: str = "rav", length: int = 32) -> str:
    """
    Generate a human-readable API token like ``rav_<hex>``.

    Args:
        prefix: Short identifier prepended to the token.
        length: Number of random bytes.
    """
    return f"{prefix}_{secrets.token_hex(length)}"
