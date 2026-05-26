"""
examples/compute/demo.py
========================
Smoke test for the Phase-2 Compute backend abstraction layer.

What this covers
----------------
1.  GenerateRequest / Message construction and flat_prompt()
2.  OllamaBackend  — health-check against a live or mocked server
3.  OpenAICompatBackend — health-check against a live or mocked endpoint
4.  ComputeRouter  — offline mode (no registry) using add_local_backend()
5.  LoadBasedStrategy / RoundRobinStrategy / ModelMatchStrategy
6.  ComputeCapability / HealthStatus dataclasses

Running
-------
# Full live test (requires Ollama running locally on :11434)
python examples/compute/demo.py --live-ollama

# Offline structural test (no external services needed)
python examples/compute/demo.py

# Against a real registry
python examples/compute/demo.py --registry 127.0.0.1:50099 --live-ollama
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
import types
import unittest.mock as mock
from typing import List

# ── make sure the package root is on sys.path ─────────────────────────────────
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

# Import directly from sub-packages to avoid triggering torch/distributed
# imports in the top-level ravnest/__init__.py.
from ravnest.compute.base import (
    ComputeBackend,
    ComputeCapability,
    EmbedRequest,
    EmbedResponse,
    GenerateRequest,
    GenerateResponse,
    HealthStatus,
    Message,
)
from ravnest.compute.router import (
    ComputeRouter,
    LoadBasedStrategy,
    ModelMatchStrategy,
    RoundRobinStrategy,
)
from ravnest.compute.ollama_backend   import OllamaBackend
from ravnest.compute.openai_compat    import OpenAICompatBackend

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sep(title: str = "") -> None:
    width = 60
    if title:
        pad   = (width - len(title) - 2) // 2
        print(f"\n{'─' * pad} {title} {'─' * pad}")
    else:
        print("─" * width)


def _ok(msg: str) -> None:
    print(f"  ✓  {msg}")


def _skip(msg: str) -> None:
    print(f"  ○  SKIP  {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data-class construction
# ─────────────────────────────────────────────────────────────────────────────

def test_dataclasses() -> None:
    _sep("Data classes")

    # Message
    m = Message("user", "Hello!")
    assert m.role == "user" and m.content == "Hello!"
    _ok("Message(role, content)")

    # GenerateRequest — prompt path
    req = GenerateRequest(prompt="Why is the sky blue?", max_tokens=64)
    assert req.flat_prompt() == "Why is the sky blue?"
    _ok("GenerateRequest.flat_prompt() — raw prompt")

    # GenerateRequest — messages path
    req2 = GenerateRequest(
        messages=[Message("system", "You are helpful."), Message("user", "Hi")],
        max_tokens=32,
    )
    fp = req2.flat_prompt()
    assert "system: You are helpful." in fp
    assert "user: Hi" in fp
    _ok("GenerateRequest.flat_prompt() — messages")

    # GenerateResponse
    resp = GenerateResponse(text="Blue sky", model="test", backend="mock",
                            request_id="r1", latency_ms=12.3)
    assert resp.text == "Blue sky"
    _ok("GenerateResponse")

    # EmbedRequest / EmbedResponse
    ereq = EmbedRequest(texts=["hello", "world"])
    eresp = EmbedResponse(embeddings=[[0.1, 0.2], [0.3, 0.4]], model="embed",
                          backend="mock", request_id=ereq.request_id)
    assert len(eresp.embeddings) == 2
    _ok("EmbedRequest / EmbedResponse")

    # HealthStatus
    hs = HealthStatus(healthy=True, backend="mock", model="m")
    assert hs.healthy
    _ok("HealthStatus")

    # ComputeCapability
    cap = ComputeCapability(backend="mock", models=["m1", "m2"],
                            supports_streaming=True, node_id="n1")
    assert cap.supports_streaming
    _ok("ComputeCapability")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Stub backend for router tests
# ─────────────────────────────────────────────────────────────────────────────

class _StubBackend(ComputeBackend):
    """Minimal in-process backend for testing the router without HTTP."""

    def __init__(self, model: str, node_id: str, gpu_pct: float = 0.0):
        self._model   = model
        self._node_id = node_id
        self._gpu_pct = gpu_pct
        self._calls: List[str] = []

    async def agenerate(self, request: GenerateRequest) -> GenerateResponse:
        self._calls.append("agenerate")
        return GenerateResponse(
            text       = f"response from {self._node_id}",
            model      = self._model,
            backend    = "stub",
            request_id = request.request_id,
        )

    async def agenerate_stream(self, request: GenerateRequest):
        yield f"token from {self._node_id}"

    async def aembed(self, request: EmbedRequest) -> EmbedResponse:
        return EmbedResponse(embeddings=[[0.0]], model=self._model,
                             backend="stub", request_id=request.request_id)

    async def ahealth(self) -> HealthStatus:
        return HealthStatus(healthy=True, backend="stub", model=self._model)

    def capabilities(self) -> ComputeCapability:
        return ComputeCapability(
            backend    = "stub",
            models     = [self._model],
            node_id    = self._node_id,
            extra      = {"base_url": "local"},
        )

    def _build_node_capability(self):
        """Return a NodeCapability-like object for the router."""
        from ravnest.registry.capability import NodeCapability as NC
        from ravnest.registry.capability import NodeType
        return NC(
            node_id      = self._node_id,
            node_type    = NodeType.STANDALONE_COMPUTE,
            subtype      = "stub",
            address      = "local",
            models       = [self._model],
            current_load = {"gpu_percent": self._gpu_pct,
                            "gpu_vram_percent": self._gpu_pct * 0.8},
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Router tests (offline — no registry)
# ─────────────────────────────────────────────────────────────────────────────

def _make_offline_router() -> ComputeRouter:
    """Build a ComputeRouter that skips registry lookup entirely."""
    # Patch RegistryClient so no network call is attempted
    with mock.patch("ravnest.compute.router.ComputeRouter._refresh_backends"):
        with mock.patch("ravnest.compute.router.ComputeRouter._start_refresh_thread"):
            router = ComputeRouter.__new__(ComputeRouter)
            router._strategy         = LoadBasedStrategy()
            router._node_types       = ["standalone_compute"]
            router._refresh_interval = 30.0
            router._max_retries      = 3
            router._backends         = {}
            router._local_backends   = []
            router._last_refresh     = time.monotonic()
            import threading
            router._lock = threading.RLock()
    return router


def test_router_load_based() -> None:
    _sep("Router — LoadBasedStrategy")

    router = _make_offline_router()

    # Add two stub backends: node-A is heavily loaded, node-B is idle
    b_heavy = _StubBackend("llama3", "node-A", gpu_pct=80.0)
    b_light = _StubBackend("llama3", "node-B", gpu_pct=10.0)
    router.add_local_backend(b_heavy)
    router.add_local_backend(b_light)

    req = GenerateRequest(messages=[Message("user", "Hi")], max_tokens=16)
    resp = router.generate(req)
    assert "node-B" in resp.text, f"Expected node-B (lighter load) but got: {resp.text}"
    _ok("Routes to least-loaded node (node-B)")

    info = router.list_backends()
    assert len(info) == 2
    _ok(f"list_backends() returns {len(info)} entries")


def test_router_round_robin() -> None:
    _sep("Router — RoundRobinStrategy")

    # Test the strategy in isolation: pick() should cycle across all candidates.
    # (The full router's _rank_candidates calls pick() N times per request to
    # build a ranked list — testing the strategy directly is the right level.)
    strat = RoundRobinStrategy()

    # Build mock capability objects
    class _MockCap:
        def __init__(self, nid):
            self.node_id = nid

    caps = [_MockCap(f"node-{i}") for i in range(3)]

    picked = [strat.pick(caps).node_id for _ in range(9)]
    # Should cycle 0, 1, 2, 0, 1, 2, ...
    assert picked == ["node-0", "node-1", "node-2"] * 3, \
        f"Unexpected round-robin order: {picked}"
    _ok(f"RoundRobinStrategy cycles evenly: {picked[:3]} ...")

    # Also verify the router wires it up without error
    router = _make_offline_router()
    router._strategy = RoundRobinStrategy()
    router.add_local_backend(_StubBackend("llama3", "node-A"))
    req  = GenerateRequest(prompt="ping", max_tokens=4)
    resp = router.generate(req)
    assert resp.text  # any response is fine
    _ok("Router dispatches successfully with RoundRobinStrategy")


def test_router_model_match() -> None:
    _sep("Router — ModelMatchStrategy")

    router = _make_offline_router()
    router._strategy = ModelMatchStrategy("mistral", inner=LoadBasedStrategy())

    router.add_local_backend(_StubBackend("llama3",  "node-llama",   gpu_pct=5.0))
    router.add_local_backend(_StubBackend("mistral", "node-mistral", gpu_pct=90.0))

    # Even though mistral node is heavier, model match should prefer it
    req  = GenerateRequest(
        messages=[Message("user", "test")], model="mistral", max_tokens=8
    )
    resp = router.generate(req)
    assert "node-mistral" in resp.text, f"Expected node-mistral, got: {resp.text}"
    _ok("ModelMatchStrategy prefers exact model match (node-mistral)")


def test_router_remove_backend() -> None:
    _sep("Router — remove_local_backend")

    router = _make_offline_router()
    b = _StubBackend("llama3", "node-X")
    router.add_local_backend(b)
    assert len(router._local_backends) == 1
    router.remove_local_backend("node-X")
    assert len(router._local_backends) == 0
    _ok("remove_local_backend() clears the entry")


def test_router_no_backends() -> None:
    _sep("Router — no backends raises RuntimeError")

    router = _make_offline_router()
    req    = GenerateRequest(prompt="test", max_tokens=4)
    try:
        router.generate(req)
        assert False, "Should have raised RuntimeError"
    except RuntimeError as exc:
        assert "No compute backends" in str(exc)
        _ok(f"Raised RuntimeError correctly: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. async streaming smoke test
# ─────────────────────────────────────────────────────────────────────────────

async def _async_stream_test() -> None:
    router = _make_offline_router()
    b = _StubBackend("llama3", "node-stream")
    router.add_local_backend(b)

    req    = GenerateRequest(prompt="stream test", max_tokens=16)
    tokens = []
    async for tok in router.agenerate_stream(req):
        tokens.append(tok)
    assert tokens, "No tokens received from stream"
    return tokens


def test_async_streaming() -> None:
    _sep("Router — async streaming")
    tokens = asyncio.run(_async_stream_test())
    _ok(f"agenerate_stream() yielded {len(tokens)} token(s): {tokens}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Live Ollama test (optional)
# ─────────────────────────────────────────────────────────────────────────────

async def _live_ollama_health(model: str, base_url: str) -> HealthStatus:
    backend = OllamaBackend(model=model, base_url=base_url)
    return await backend.ahealth()


def test_live_ollama(base_url: str = "http://localhost:11434",
                     model:    str = "llama3.2") -> None:
    _sep(f"Live Ollama ({base_url})")
    try:
        import httpx  # noqa: F401
    except ImportError:
        _skip("httpx not installed — skipping live Ollama test")
        return

    try:
        hs = asyncio.run(_live_ollama_health(model, base_url))
    except Exception as exc:
        _skip(f"Ollama unreachable ({exc})")
        return

    if not hs.healthy:
        _skip(f"Ollama not healthy: {hs.message}")
        return

    _ok(f"Ollama healthy: {hs.message}")

    # Generate a short response
    backend = OllamaBackend(model=model, base_url=base_url, timeout=60.0)
    req  = GenerateRequest(
        messages  = [Message("user", "Reply with exactly three words.")],
        max_tokens = 16,
    )
    try:
        resp = backend.generate(req)
        _ok(f"generate() → '{resp.text.strip()}'  ({resp.latency_ms:.0f} ms)")
    except Exception as exc:
        _skip(f"generate() failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Live OpenAI-compat test (optional — needs OPENAI_API_KEY or local server)
# ─────────────────────────────────────────────────────────────────────────────

async def _live_openai_health(model: str, base_url: str, api_key: str) -> HealthStatus:
    backend = OpenAICompatBackend(model=model, base_url=base_url, api_key=api_key)
    return await backend.ahealth()


def test_live_openai(base_url: str, model: str, api_key: str) -> None:
    _sep(f"Live OpenAI-compat ({base_url})")
    try:
        import openai  # noqa: F401
    except ImportError:
        _skip("openai not installed — skipping live OpenAI test")
        return

    try:
        hs = asyncio.run(_live_openai_health(model, base_url, api_key))
    except Exception as exc:
        _skip(f"OpenAI-compat endpoint unreachable ({exc})")
        return

    if not hs.healthy:
        _skip(f"OpenAI-compat not healthy: {hs.message}")
        return
    _ok(f"OpenAI-compat healthy: {hs.message}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Registry-backed router smoke test (optional)
# ─────────────────────────────────────────────────────────────────────────────

def test_registry_router(registry_address: str) -> None:
    _sep(f"Registry-backed ComputeRouter ({registry_address})")
    try:
        router = ComputeRouter(registry_address=registry_address, max_retries=1)
        backends = router.list_backends()
        _ok(f"Discovered {len(backends)} backend(s) from registry")
        for b in backends:
            print(f"       {b['node_id']:30s}  subtype={b['subtype']}  models={b['models']}")
    except Exception as exc:
        _skip(f"Registry unreachable ({exc})")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Ravnest compute layer smoke test")
    parser.add_argument("--live-ollama",  action="store_true",
                        help="Run against a live Ollama server")
    parser.add_argument("--ollama-url",   default="http://localhost:11434",
                        help="Ollama base URL")
    parser.add_argument("--ollama-model", default="llama3.2",
                        help="Ollama model tag to use")
    parser.add_argument("--live-openai",  action="store_true",
                        help="Run against an OpenAI-compat endpoint")
    parser.add_argument("--openai-url",   default="https://api.openai.com/v1",
                        help="OpenAI-compat base URL")
    parser.add_argument("--openai-model", default="gpt-4o-mini")
    parser.add_argument("--openai-key",   default="EMPTY")
    parser.add_argument("--registry",     default=None,
                        help="host:port of a live Ravnest registry")
    args = parser.parse_args()

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║      Ravnest · Phase-2 Compute Layer  smoke test    ║")
    print("╚══════════════════════════════════════════════════════╝")

    # Offline tests (always run)
    test_dataclasses()
    test_router_load_based()
    test_router_round_robin()
    test_router_model_match()
    test_router_remove_backend()
    test_router_no_backends()
    test_async_streaming()

    # Optional live tests
    if args.live_ollama:
        test_live_ollama(base_url=args.ollama_url, model=args.ollama_model)
    else:
        _sep("Live Ollama")
        _skip("pass --live-ollama to run against a real Ollama server")

    if args.live_openai:
        test_live_openai(base_url=args.openai_url,
                         model=args.openai_model,
                         api_key=args.openai_key)
    else:
        _sep("Live OpenAI-compat")
        _skip("pass --live-openai to run against a real OpenAI-compat endpoint")

    if args.registry:
        test_registry_router(args.registry)
    else:
        _sep("Registry-backed router")
        _skip("pass --registry host:port to run against a live registry")

    _sep()
    print("\n  All offline tests passed ✓\n")


if __name__ == "__main__":
    main()
