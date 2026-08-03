"""
examples/security/demo.py
=========================
Smoke test for Phase-7 — Security & Trust.

What this covers
----------------
1.  crypto — generate_key, sign/verify, hash_key
2.  crypto — timestamp generation + replay detection
3.  crypto — sign_message_dict / verify_message_dict
4.  crypto — generate_token format
5.  auth — APIKeyStore: add, verify, revoke, disable
6.  auth — APIKeyStore: scope checking
7.  auth — APIKeyStore: list_keys / use_count tracking
8.  rate_limiter — RateLimiter basic consume
9.  rate_limiter — RateLimiter burst enforcement
10. rate_limiter — MultiLimiter stacked enforcement
11. rate_limiter — evict_stale (no error path)
12. sandbox — ToolSandbox allowlist enforcement
13. sandbox — ToolSandbox denylist takes priority
14. sandbox — ToolSandbox call-count cap
15. sandbox — ToolSandbox timeout enforcement
16. sandbox — ToolSandbox.wrap() produces callable
17. sandbox — InputValidator happy path
18. sandbox — InputValidator prompt length violation
19. sandbox — InputValidator message count violation
20. tls — generate_ca (skip if cryptography not installed)
21. tls — generate_node_cert (skip if cryptography not installed)
22. tls — server ssl.SSLContext (skip if cryptography not installed)
23. tls — CertBundle.save / CertBundle.load round-trip (skip if missing)

Running
-------
python3 examples/security/demo.py
python3 examples/security/demo.py --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import pathlib
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

PASS  = "PASS"
FAIL  = "FAIL"
SKIP  = "SKIP"
_results: list[tuple[str, str]] = []
_verbose = False


def _sep(title: str = "") -> None:
    width = 64
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─' * pad} {title} {'─' * pad}")
    else:
        print("─" * width)


def check(name: str, ok: bool, detail: str = "") -> None:
    status = PASS if ok else FAIL
    _results.append((name, status))
    icon   = "✓" if ok else "✗"
    suffix = f"  ({detail})" if (detail and _verbose) else ""
    print(f"  [{icon}] {name}{suffix}")


def skip(name: str, reason: str = "") -> None:
    _results.append((name, SKIP))
    print(f"  [-] SKIP  {name}  ({reason})")


# ─────────────────────────────────────────────────────────────────────────────
# 1–4  crypto
# ─────────────────────────────────────────────────────────────────────────────

def test_crypto_basic():
    _sep("crypto — sign / verify")

    from ravnest.security.crypto import generate_key, sign, verify, hash_key

    key = generate_key()
    check("generate_key length=64 hex", len(key) == 64)
    check("generate_key hex chars",     all(c in "0123456789abcdef" for c in key))

    payload = b'{"prompt": "hello", "model": "llama3.2"}'
    sig     = sign(payload, key)
    check("sign returns string",        isinstance(sig, str))
    check("verify correct sig → True",  verify(payload, key, sig))
    check("verify wrong payload → False",
          not verify(b"tampered!", key, sig))
    check("verify wrong key → False",
          not verify(payload, generate_key(), sig))
    check("verify empty sig → False",   not verify(payload, key, ""))
    check("verify garbage → False",     not verify(payload, key, "NOTBASE64!@#"))

    # hash_key is deterministic
    h1 = hash_key("my-api-key")
    h2 = hash_key("my-api-key")
    h3 = hash_key("different-key")
    check("hash_key deterministic",     h1 == h2)
    check("hash_key different inputs",  h1 != h3)
    check("hash_key length=64",         len(h1) == 64)


def test_crypto_timestamp():
    _sep("crypto — timestamps / replay guard")

    from ravnest.security.crypto import make_timestamp, check_timestamp

    ts = make_timestamp()
    check("make_timestamp is digit string", ts.isdigit())
    check("check_timestamp fresh → True",   check_timestamp(ts, max_age_seconds=60))
    check("check_timestamp stale → False",
          not check_timestamp("0", max_age_seconds=60))
    check("check_timestamp garbage → False",
          not check_timestamp("not-a-number"))


def test_crypto_sign_dict():
    _sep("crypto — sign_message_dict / verify_message_dict")

    from ravnest.security.crypto import generate_key, sign_message_dict, verify_message_dict

    key  = generate_key()
    body = {"prompt": "hello", "model": "llama3.2"}

    signed, sig = sign_message_dict(body, key)
    check("signed dict has _ts",  "_ts"  in signed)
    check("signed dict has _sig", "_sig" in signed)
    check("original body unchanged", "model" in body and "_sig" not in body)

    check("verify_message_dict valid → True",
          verify_message_dict(signed, key))
    check("verify_message_dict wrong key → False",
          not verify_message_dict(signed, generate_key()))

    # Tampered field
    tampered = {**signed, "model": "evil-model"}
    check("verify_message_dict tampered → False",
          not verify_message_dict(tampered, key))

    # Missing _sig
    no_sig = {k: v for k, v in signed.items() if k != "_sig"}
    check("verify_message_dict missing sig → False",
          not verify_message_dict(no_sig, key))


def test_crypto_token():
    _sep("crypto — generate_token")

    from ravnest.security.crypto import generate_token

    tok = generate_token("rav")
    check("token has prefix",      tok.startswith("rav_"))
    check("token long enough",     len(tok) >= 20)

    t1 = generate_token()
    t2 = generate_token()
    check("tokens are unique",     t1 != t2)


# ─────────────────────────────────────────────────────────────────────────────
# 5–7  auth
# ─────────────────────────────────────────────────────────────────────────────

def test_auth_keystore():
    _sep("auth — APIKeyStore")

    from ravnest.security.auth import APIKeyStore

    store = APIKeyStore()

    key1 = store.add_key("client-a", scopes=["generate"])
    key2 = store.add_key("client-b", scopes=["rag", "query"])

    # Verify valid keys
    info1 = store.verify(key1)
    info2 = store.verify(key2)
    check("verify key1 → KeyInfo",    info1 is not None)
    check("verify key2 → KeyInfo",    info2 is not None)
    check("key1 name correct",        info1.name == "client-a")
    check("verify unknown → None",    store.verify("bad-key") is None)
    check("verify empty → None",      store.verify("") is None)

    # use_count increments
    info1_again = store.verify(key1)
    check("use_count increments",     info1.use_count == 2)

    # Disable
    store.disable_key(key1)
    check("disabled key → None",      store.verify(key1) is None)
    store.enable_key(key1)
    check("re-enabled key → KeyInfo", store.verify(key1) is not None)

    # Revoke
    store.revoke_key(key2)
    check("revoked key → None",       store.verify(key2) is None)
    check("revoke returns True",      not store.revoke_key("no-such-key"))


def test_auth_scopes():
    _sep("auth — scope enforcement")

    from ravnest.security.auth import APIKeyStore

    store = APIKeyStore()
    full  = store.add_key("admin",    scopes=[])            # empty = all
    gen   = store.add_key("gen-only", scopes=["generate"])
    wild  = store.add_key("wildcard", scopes=["*"])

    info_full = store.verify(full)
    info_gen  = store.verify(gen)
    info_wild = store.verify(wild)

    check("empty scopes → has_scope('generate')",  info_full.has_scope("generate"))
    check("empty scopes → has_scope('rag')",       info_full.has_scope("rag"))
    check("gen-only → has_scope('generate')",      info_gen.has_scope("generate"))
    check("gen-only → NOT has_scope('rag')",       not info_gen.has_scope("rag"))
    check("wildcard → has_scope('anything')",      info_wild.has_scope("anything"))


def test_auth_list():
    _sep("auth — list_keys / anonymous mode")

    from ravnest.security.auth import APIKeyStore

    store = APIKeyStore(allow_anonymous=True)
    store.add_key("x")
    store.add_key("y")

    keys = store.list_keys()
    check("list_keys returns 2",     len(keys) == 2)
    check("list_keys has name",      all("name" in k for k in keys))
    check("list_keys no plaintext",  all("key" not in k for k in keys))
    check("allow_anonymous True",    store.allow_anonymous is True)


# ─────────────────────────────────────────────────────────────────────────────
# 8–11  rate_limiter
# ─────────────────────────────────────────────────────────────────────────────

async def test_rate_limiter_basic():
    _sep("rate_limiter — basic consume")

    from ravnest.security.rate_limiter import RateLimiter

    limiter = RateLimiter(rate=100.0, burst=5.0)

    # First 5 requests should succeed
    results = [await limiter.consume("test-ip") for _ in range(5)]
    check("burst allows 5 requests",  all(results))

    # 6th should fail
    r6 = await limiter.consume("test-ip")
    check("burst exhausted → False",  r6 is False)

    # Different key has its own bucket
    r_new = await limiter.consume("other-ip")
    check("different key has own bucket", r_new is True)


async def test_rate_limiter_refill():
    _sep("rate_limiter — refill over time")

    from ravnest.security.rate_limiter import RateLimiter

    # Very fast refill: 1000 tokens/s, burst=2
    limiter = RateLimiter(rate=1000.0, burst=2.0)

    r1 = await limiter.consume("k")
    r2 = await limiter.consume("k")
    r3 = await limiter.consume("k")   # should fail initially
    check("burst of 2 exhausted on 3rd", r3 is False)

    # Wait 5ms → should refill ~5 tokens at 1000/s
    await asyncio.sleep(0.005)
    r4 = await limiter.consume("k")
    check("refills after sleep",      r4 is True)


async def test_multi_limiter():
    _sep("rate_limiter — MultiLimiter")

    from ravnest.security.rate_limiter import RateLimiter, MultiLimiter

    loose  = RateLimiter(rate=100.0, burst=10.0)
    strict = RateLimiter(rate=1.0,   burst=2.0)
    multi  = MultiLimiter([loose, strict])

    # Should pass (both have tokens)
    r1 = await multi.consume(["ip-a", "key-a"])
    r2 = await multi.consume(["ip-a", "key-a"])
    check("multi: first 2 allowed",   r1 and r2)

    # Strict limiter exhausted
    r3 = await multi.consume(["ip-a", "key-a"])
    check("multi: 3rd blocked by strict limiter", r3 is False)

    # Loose limiter still has tokens for a different key combo
    r4 = await multi.consume(["ip-a", "key-b"])
    check("multi: different key combo uses own bucket", r4 is True)


async def test_evict_stale():
    _sep("rate_limiter — evict_stale")

    from ravnest.security.rate_limiter import RateLimiter

    limiter = RateLimiter(rate=10.0, burst=10.0, ttl=0.01)
    await limiter.consume("evict-me")
    check("bucket exists before eviction", len(limiter._buckets) == 1)

    await asyncio.sleep(0.02)   # TTL expires
    removed = await limiter.evict_stale()
    check("evict_stale removes idle bucket", removed == 1)
    check("buckets dict empty after evict",  len(limiter._buckets) == 0)


# ─────────────────────────────────────────────────────────────────────────────
# 12–16  sandbox
# ─────────────────────────────────────────────────────────────────────────────

def test_sandbox_allowlist():
    _sep("sandbox — ToolSandbox allowlist")

    from ravnest.security.sandbox import ToolSandbox, ToolNotAllowed

    sandbox = ToolSandbox(allowed_tools={"web_search", "calculator"})

    check("allowed tool passes",     sandbox.is_allowed("web_search"))
    check("allowed tool passes",     sandbox.is_allowed("calculator"))
    check("unknown tool blocked",    not sandbox.is_allowed("bash"))
    check("unknown tool blocked",    not sandbox.is_allowed("python_repl"))

    # check_tool raises for disallowed
    raised = False
    try:
        sandbox.check_tool("bash")
    except ToolNotAllowed:
        raised = True
    check("check_tool raises ToolNotAllowed", raised)

    # check_tool passes for allowed
    no_raise = True
    try:
        sandbox.check_tool("web_search")
    except ToolNotAllowed:
        no_raise = False
    check("check_tool passes for allowed", no_raise)


def test_sandbox_denylist():
    _sep("sandbox — denylist takes priority")

    from ravnest.security.sandbox import ToolSandbox

    sandbox = ToolSandbox(
        allowed_tools={"bash", "python_repl", "web_search"},
        blocked_tools={"bash", "python_repl"},
    )
    check("blocked despite allowlist: bash",         not sandbox.is_allowed("bash"))
    check("blocked despite allowlist: python_repl",  not sandbox.is_allowed("python_repl"))
    check("not-blocked passes: web_search",           sandbox.is_allowed("web_search"))


def test_sandbox_call_count():
    _sep("sandbox — call count cap")

    from ravnest.security.sandbox import ToolSandbox, ToolLimitExceeded

    def echo_executor(name, args):
        return f"result from {name}"

    sandbox = ToolSandbox(max_calls=3)
    wrapped = sandbox.wrap(echo_executor)

    results = []
    for _ in range(3):
        results.append(wrapped("web_search", {"q": "x"}))
    check("3 calls allowed",  len(results) == 3)

    exceeded = False
    try:
        wrapped("web_search", {"q": "one too many"})
    except ToolLimitExceeded:
        exceeded = True
    check("4th call raises ToolLimitExceeded", exceeded)


def test_sandbox_timeout():
    _sep("sandbox — tool timeout")

    from ravnest.security.sandbox import ToolSandbox, ToolTimeout
    import time

    def slow_tool(name, args):
        time.sleep(10)   # way longer than timeout
        return "should not reach"

    sandbox = ToolSandbox(timeout=0.1)
    wrapped = sandbox.wrap(slow_tool)

    timed_out = False
    try:
        wrapped("slow_tool", {})
    except ToolTimeout:
        timed_out = True
    check("slow tool raises ToolTimeout", timed_out)


def test_sandbox_wrap():
    _sep("sandbox — wrap / describe")

    from ravnest.security.sandbox import ToolSandbox

    def my_exec(name, args):
        return {"tool": name, "result": "ok"}

    sandbox = ToolSandbox(
        allowed_tools={"search"},
        blocked_tools={"bash"},
        max_calls=5,
        timeout=10.0,
    )
    wrapped = sandbox.wrap(my_exec)
    result  = wrapped("search", {"q": "test"})

    check("wrapped executor returns result", result["result"] == "ok")

    desc = sandbox.describe()
    check("describe has allowed_tools", "allowed_tools" in desc)
    check("describe has max_calls",     desc["max_calls"] == 5)


def test_input_validator():
    _sep("sandbox — InputValidator")

    from ravnest.security.sandbox import InputValidator
    from ravnest.gateway.base     import GatewayRequest

    validator = InputValidator(max_prompt_length=100, max_messages=3)

    # Happy path
    ok_req = GatewayRequest(prompt="hello")
    try:
        validator.validate(ok_req)
        check("valid request passes", True)
    except ValueError:
        check("valid request passes", False)

    # Prompt too long
    long_req = GatewayRequest(prompt="x" * 200)
    raised = False
    try:
        validator.validate(long_req)
    except ValueError:
        raised = True
    check("long prompt raises ValueError", raised)

    # Too many messages
    msgs_req = GatewayRequest(messages=[
        {"role": "user", "content": f"msg {i}"} for i in range(5)
    ])
    raised = False
    try:
        validator.validate(msgs_req)
    except ValueError:
        raised = True
    check("too many messages raises ValueError", raised)


# ─────────────────────────────────────────────────────────────────────────────
# 20–23  TLS
# ─────────────────────────────────────────────────────────────────────────────

def test_tls():
    _sep("tls — generate_ca / node_cert / SSLContext")

    try:
        import cryptography  # noqa
    except ImportError:
        for name in ["generate_ca", "generate_node_cert",
                     "server ssl.SSLContext", "CertBundle.save / load"]:
            skip(name, "cryptography not installed")
        return

    from ravnest.security.tls import generate_ca, generate_node_cert, CertBundle

    ca = generate_ca(cn="test-ca", days=1, key_size=2048)
    check("ca is CertBundle",     isinstance(ca, CertBundle))
    check("ca.is_ca is True",     ca.is_ca is True)
    check("ca has cert_pem",      ca.cert_pem.startswith(b"-----BEGIN CERTIFICATE"))
    check("ca has key_pem",       ca.key_pem.startswith(b"-----BEGIN RSA PRIVATE KEY"))

    node = generate_node_cert(ca, "node-1", days=1, key_size=2048,
                              san_ips=["127.0.0.1"])
    check("node is CertBundle",   isinstance(node, CertBundle))
    check("node.is_ca is False",  node.is_ca is False)
    check("node.cn correct",      node.cn == "node-1")

    # SSLContext
    import ssl
    ctx = node.server_ssl_context(ca_bundle=ca)
    check("server_ssl_context → SSLContext", isinstance(ctx, ssl.SSLContext))

    cctx = node.client_ssl_context(ca_bundle=ca)
    check("client_ssl_context → SSLContext", isinstance(cctx, ssl.SSLContext))

    # save / load round-trip
    with tempfile.TemporaryDirectory() as d:
        cert_path, key_path = ca.save(d, "ca")
        loaded = CertBundle.load(cert_path, key_path, is_ca=True)
        check("loaded cert_pem matches", loaded.cert_pem == ca.cert_pem)
        check("loaded key_pem matches",  loaded.key_pem  == ca.key_pem)
        check("loaded cn matches",       loaded.cn       == ca.cn)


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

async def _run_async():
    await test_rate_limiter_basic()
    await test_rate_limiter_refill()
    await test_multi_limiter()
    await test_evict_stale()


def main():
    global _verbose
    p = argparse.ArgumentParser(description="Security phase-7 smoke test")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    _verbose = args.verbose

    print("=" * 64)
    print(" Ravnest Security — Phase 7 smoke test")
    print("=" * 64)

    test_crypto_basic()
    test_crypto_timestamp()
    test_crypto_sign_dict()
    test_crypto_token()
    test_auth_keystore()
    test_auth_scopes()
    test_auth_list()

    asyncio.run(_run_async())

    test_sandbox_allowlist()
    test_sandbox_denylist()
    test_sandbox_call_count()
    test_sandbox_timeout()
    test_sandbox_wrap()
    test_input_validator()
    test_tls()

    print()
    _sep("Summary")
    passed  = sum(1 for _, s in _results if s == PASS)
    failed  = sum(1 for _, s in _results if s == FAIL)
    skipped = sum(1 for _, s in _results if s == SKIP)
    print(f"  Passed: {passed}   Failed: {failed}   Skipped: {skipped}")
    if failed:
        print("\nFailed tests:")
        for name, status in _results:
            if status == FAIL:
                print(f"    ✗  {name}")
        sys.exit(1)
    else:
        print("\nAll tests passed.")


if __name__ == "__main__":
    main()
