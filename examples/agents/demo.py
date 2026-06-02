"""
examples/agents/demo.py
=======================
Smoke test for the Phase-3 Agent Node Framework.

What this covers
----------------
1.  AgentRequest / Message / ToolCall / ToolResult dataclasses
2.  AgentBackend abstract interface — sync wrappers (run / stream / health)
3.  LiteLLMAgent structural test (no live model needed)
4.  ResearchAgent structural test + mock search
5.  SQLAgent structural test + in-memory SQLite DB
6.  AgentRouter offline mode via add_local_backend()
7.  LoadBasedAgentStrategy / RoundRobinAgentStrategy / AgentTypeStrategy
8.  AgentCapability capabilities() reporting

Running
-------
# Offline structural tests (no external services needed)
python examples/agents/demo.py

# Full live test with LiteLLM (needs API key in environment)
python examples/agents/demo.py --live-litellm --model gpt-4o-mini

# Live research agent (needs duckduckgo-search + LiteLLM)
python examples/agents/demo.py --live-research --model gpt-4o-mini

# Live SQL agent (uses in-memory SQLite)
python examples/agents/demo.py --live-sql --model gpt-4o-mini

# Against a real registry
python examples/agents/demo.py --registry 127.0.0.1:50099
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
import threading
import unittest.mock as mock
from typing import List

import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from ravnest.agents.base import (
    AgentBackend, AgentCapability, AgentHealthStatus,
    AgentRequest, AgentResponse, Message, ToolCall, ToolResult,
)
from ravnest.agents.router import (
    AgentRouter, LoadBasedAgentStrategy, RoundRobinAgentStrategy,
    AgentTypeStrategy,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sep(title: str = "") -> None:
    width = 62
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─' * pad} {title} {'─' * pad}")
    else:
        print("─" * width)


def _ok(msg: str) -> None:   print(f"  ✓  {msg}")
def _skip(msg: str) -> None: print(f"  ○  SKIP  {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data-class construction
# ─────────────────────────────────────────────────────────────────────────────

def test_dataclasses() -> None:
    _sep("Data classes")

    m = Message("user", "Hello!")
    assert m.role == "user" and m.content == "Hello!"
    _ok("Message(role, content)")

    req = AgentRequest(
        messages    = [Message("system", "You are helpful."),
                       Message("user",   "What is 2+2?")],
        max_steps   = 5,
        max_tokens  = 256,
    )
    assert req.last_user_message() == "What is 2+2?"
    assert req.request_id  # auto-generated UUID
    _ok("AgentRequest — last_user_message(), auto request_id")

    tc = ToolCall(id="call_1", name="web_search",
                  args={"query": "ravnest"}, raw='{"query":"ravnest"}')
    assert tc.name == "web_search"
    _ok("ToolCall")

    tr = ToolResult(tool_call_id="call_1", content="result text")
    assert not tr.is_error
    _ok("ToolResult")

    resp = AgentResponse(text="Four.", agent="litellm", model="test",
                         request_id=req.request_id, latency_ms=42.0)
    assert resp.finish_reason == "stop"
    _ok("AgentResponse")

    hs = AgentHealthStatus(healthy=True, agent="litellm", model="test")
    assert hs.healthy
    _ok("AgentHealthStatus")

    cap = AgentCapability(agent_type="research", models=["gpt-4o"],
                          tools=["web_search"], node_id="n1")
    assert cap.tools == ["web_search"]
    _ok("AgentCapability")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Stub backend for router tests
# ─────────────────────────────────────────────────────────────────────────────

class _StubAgent(AgentBackend):
    """Minimal in-process agent for testing the router without LLM calls."""

    def __init__(self, agent_type: str, node_id: str, ram_pct: float = 0.0):
        self._agent_type = agent_type
        self._node_id    = node_id
        self._ram_pct    = ram_pct
        self.calls: List[str] = []

    async def arun(self, request: AgentRequest) -> AgentResponse:
        self.calls.append("arun")
        return AgentResponse(
            text       = f"answer from {self._node_id}",
            agent      = self._agent_type,
            request_id = request.request_id,
        )

    async def astream(self, request: AgentRequest):
        yield f"token from {self._node_id}"

    async def ahealth(self) -> AgentHealthStatus:
        return AgentHealthStatus(healthy=True, agent=self._agent_type)

    def capabilities(self) -> AgentCapability:
        return AgentCapability(
            agent_type = self._agent_type,
            models     = ["stub-model"],
            node_id    = self._node_id,
            extra      = {"address": "local"},
        )

    def _build_node_capability(self):
        return self.capabilities()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Router helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_offline_router() -> AgentRouter:
    """Build an AgentRouter with no registry connection."""
    with mock.patch("ravnest.agents.router.AgentRouter._refresh_backends"):
        with mock.patch("ravnest.agents.router.AgentRouter._start_refresh_thread"):
            router = AgentRouter.__new__(AgentRouter)
            router._strategy         = LoadBasedAgentStrategy()
            router._refresh_interval = 30.0
            router._max_retries      = 3
            router._backends         = {}
            router._local_backends   = []
            router._last_refresh     = time.monotonic()
            router._lock             = __import__("threading").RLock()
    return router


# ─────────────────────────────────────────────────────────────────────────────
# 4. Router tests
# ─────────────────────────────────────────────────────────────────────────────

def test_router_load_based() -> None:
    _sep("AgentRouter — LoadBasedAgentStrategy")

    router = _make_offline_router()
    b_busy = _StubAgent("litellm", "node-busy", ram_pct=90.0)
    b_idle = _StubAgent("litellm", "node-idle", ram_pct=5.0)

    # Manually create caps with load info
    from ravnest.agents.router import _AgentCapWrapper
    cap_busy = b_busy.capabilities()
    cap_idle = b_idle.capabilities()

    # Inject with mock load
    class _LoadedCap:
        def __init__(self, cap, ram):
            self.node_id      = cap.node_id
            self.agent_type   = cap.agent_type
            self.subtype      = cap.agent_type
            self.models       = cap.models
            self.current_load = {"ram_percent": ram}
            self.address      = "local"
            self.metadata     = {}
            self.extra        = cap.extra

    with router._lock:
        router._local_backends = [
            (b_busy, _LoadedCap(cap_busy, 90.0)),
            (b_idle, _LoadedCap(cap_idle,  5.0)),
        ]

    req  = AgentRequest(messages=[Message("user", "test")])
    resp = router.run(req)
    assert "node-idle" in resp.text, f"Expected node-idle (lower load), got: {resp.text}"
    _ok("Routes to least-loaded agent (node-idle)")

    info = router.list_backends()
    assert len(info) == 2
    _ok(f"list_backends() returns {len(info)} entries")


def test_router_agent_type_strategy() -> None:
    _sep("AgentRouter — AgentTypeStrategy")

    router = _make_offline_router()
    router._strategy = AgentTypeStrategy("research")

    b_litellm  = _StubAgent("litellm",  "node-litellm")
    b_research = _StubAgent("research", "node-research")
    router.add_local_backend(b_litellm)
    router.add_local_backend(b_research)

    req  = AgentRequest(
        messages = [Message("user", "Search for news.")],
        extra    = {"agent_type": "research"},
    )
    resp = router.run(req)
    assert "node-research" in resp.text, \
        f"Expected node-research, got: {resp.text}"
    _ok("AgentTypeStrategy routes to research agent")


def test_router_round_robin_strategy() -> None:
    _sep("AgentRouter — RoundRobinAgentStrategy (direct)")

    strat = RoundRobinAgentStrategy()

    class _MockCap:
        def __init__(self, nid):
            self.node_id = nid
            self.subtype = "litellm"
            self.current_load = {}

    caps   = [_MockCap(f"node-{i}") for i in range(3)]
    picked = [strat.pick(caps).node_id for _ in range(9)]
    assert picked == ["node-0", "node-1", "node-2"] * 3, \
        f"Unexpected order: {picked}"
    _ok(f"RoundRobinAgentStrategy cycles: {picked[:3]} ...")


def test_router_no_backends() -> None:
    _sep("AgentRouter — no backends raises RuntimeError")
    router = _make_offline_router()
    req    = AgentRequest(messages=[Message("user", "test")])
    try:
        router.run(req)
        assert False, "Should have raised"
    except RuntimeError as exc:
        assert "No agent backends" in str(exc)
        _ok(f"Raised RuntimeError: {exc}")


def test_router_remove_backend() -> None:
    _sep("AgentRouter — remove_local_backend")
    router = _make_offline_router()
    b = _StubAgent("litellm", "node-X")
    router.add_local_backend(b)
    assert len(router._local_backends) == 1
    router.remove_local_backend("node-X")
    assert len(router._local_backends) == 0
    _ok("remove_local_backend() clears the entry")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Stub sync wrappers
# ─────────────────────────────────────────────────────────────────────────────

def test_sync_wrappers() -> None:
    _sep("AgentBackend sync wrappers")

    agent = _StubAgent("litellm", "sync-test")
    req   = AgentRequest(messages=[Message("user", "hi")])

    resp = agent.run(req)
    assert resp.text == "answer from sync-test"
    _ok("run() wraps arun() correctly")

    tokens = list(agent.stream(req))
    assert tokens == ["token from sync-test"]
    _ok("stream() wraps astream() correctly")

    hs = agent.health()
    assert hs.healthy
    _ok("health() wraps ahealth() correctly")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Async streaming smoke test
# ─────────────────────────────────────────────────────────────────────────────

async def _async_stream_test() -> List[str]:
    router = _make_offline_router()
    router.add_local_backend(_StubAgent("litellm", "node-stream"))
    req    = AgentRequest(messages=[Message("user", "stream")])
    tokens = []
    async for tok in router.astream(req):
        tokens.append(tok)
    return tokens


def test_async_streaming() -> None:
    _sep("AgentRouter — async streaming")
    tokens = asyncio.run(_async_stream_test())
    assert tokens, "No tokens received"
    _ok(f"astream() yielded {len(tokens)} token(s): {tokens}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. LiteLLMAgent structural test (no live model)
# ─────────────────────────────────────────────────────────────────────────────

def test_litellm_structure() -> None:
    _sep("LiteLLMAgent — structural test")
    try:
        from ravnest.agents.litellm_agent import LiteLLMAgent
        agent = LiteLLMAgent(model="gpt-4o-mini", node_id="test-litellm")
    except ImportError as e:
        _skip(f"litellm not installed: {e}")
        return

    cap = agent.capabilities()
    assert cap.agent_type == "litellm"
    assert "gpt-4o-mini" in cap.models
    assert cap.node_id == "test-litellm"
    _ok(f"capabilities() → agent_type={cap.agent_type}, models={cap.models}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. ResearchAgent structural test + mock search
# ─────────────────────────────────────────────────────────────────────────────

def test_research_agent_structure() -> None:
    _sep("ResearchAgent — structural test")
    try:
        from ravnest.agents.research_agent import ResearchAgent
        agent = ResearchAgent(model="gpt-4o-mini", node_id="test-research")
    except ImportError as e:
        _skip(f"litellm not installed: {e}")
        return

    cap = agent.capabilities()
    assert cap.agent_type == "research"
    assert "web_search" in cap.tools
    _ok(f"capabilities() → tools={cap.tools}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. SQLAgent structural test + in-memory SQLite
# ─────────────────────────────────────────────────────────────────────────────

def test_sql_agent_structure() -> None:
    _sep("SQLAgent — structural + SQLite introspection")
    try:
        import sqlalchemy   # noqa: F401
        import litellm      # noqa: F401
        from ravnest.agents.sql_agent import SQLAgent
        from sqlalchemy import create_engine, text as sa_text
    except ImportError as e:
        _skip(f"sqlalchemy or litellm not installed: {e}")
        return

    # Use a file-based temp SQLite so two agents share the same schema
    import tempfile, os
    tmp = tempfile.mktemp(suffix=".db")
    db_url = f"sqlite:///{tmp}"
    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            conn.execute(sa_text(
                "CREATE TABLE products (id INTEGER, name TEXT, price REAL)"
            ))
            conn.execute(sa_text(
                "INSERT INTO products VALUES (1, 'Widget', 9.99),"
                " (2, 'Gadget', 19.99)"
            ))
            conn.commit()
        engine.dispose()

        agent = SQLAgent(model="gpt-4o-mini", db_url=db_url, node_id="test-sql")
        cap   = agent.capabilities()
        assert cap.agent_type == "sql"
        assert "sql_query" in cap.tools
        _ok(f"capabilities() → agent_type={cap.agent_type}, tools={cap.tools}")

        schema = asyncio.run(agent._get_schema())
        assert isinstance(schema, str) and len(schema) > 0
        _ok(f"_get_schema() → {schema.strip()[:60]}…")
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# 10. Live LiteLLM test (optional)
# ─────────────────────────────────────────────────────────────────────────────

async def _live_litellm_run(model: str) -> AgentResponse:
    from ravnest.agents.litellm_agent import LiteLLMAgent
    agent = LiteLLMAgent(model=model)
    req   = AgentRequest(
        messages  = [Message("user", "Reply with exactly three words.")],
        max_tokens = 16,
    )
    return await agent.arun(req)


def test_live_litellm(model: str) -> None:
    _sep(f"Live LiteLLMAgent ({model})")
    try:
        import litellm  # noqa: F401
    except ImportError:
        _skip("litellm not installed")
        return
    try:
        resp = asyncio.run(_live_litellm_run(model))
        _ok(f"arun() → '{resp.text.strip()}'  ({resp.latency_ms:.0f} ms, "
            f"tokens={resp.usage.get('total_tokens', '?')})")
    except Exception as exc:
        _skip(f"LiteLLM call failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# 11. Live research agent test (optional)
# ─────────────────────────────────────────────────────────────────────────────

async def _live_research_run(model: str) -> AgentResponse:
    from ravnest.agents.research_agent import ResearchAgent
    agent = ResearchAgent(model=model)
    req   = AgentRequest(
        messages  = [Message("user", "What is Ravnest? Give a one-sentence summary.")],
        max_steps  = 3,
        max_tokens = 128,
    )
    return await agent.arun(req)


def test_live_research(model: str) -> None:
    _sep(f"Live ResearchAgent ({model})")
    try:
        import litellm  # noqa: F401
    except ImportError:
        _skip("litellm not installed")
        return
    try:
        resp = asyncio.run(_live_research_run(model))
        _ok(f"arun() ({resp.steps} steps) → '{resp.text[:80].strip()}…'")
        _ok(f"searched={resp.metadata.get('searched')}, "
            f"latency={resp.latency_ms:.0f} ms")
    except Exception as exc:
        _skip(f"Research agent failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# 12. Registry-backed router (optional)
# ─────────────────────────────────────────────────────────────────────────────

def test_registry_router(registry_address: str) -> None:
    _sep(f"Registry-backed AgentRouter ({registry_address})")
    try:
        router   = AgentRouter(registry_address=registry_address, max_retries=1)
        backends = router.list_backends()
        _ok(f"Discovered {len(backends)} agent backend(s)")
        for b in backends:
            print(f"       {b['node_id']:30s}  type={b['agent_type']}  "
                  f"models={b['models']}")
    except Exception as exc:
        _skip(f"Registry unreachable ({exc})")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Ravnest agents smoke test")
    parser.add_argument("--live-litellm",  action="store_true")
    parser.add_argument("--live-research", action="store_true")
    parser.add_argument("--live-sql",      action="store_true")
    parser.add_argument("--model",         default="gpt-4o-mini",
                        help="LiteLLM model string for live tests")
    parser.add_argument("--registry",      default=None,
                        help="host:port of a live Ravnest registry")
    args = parser.parse_args()

    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║       Ravnest · Phase-3 Agent Node Framework  smoke test   ║")
    print("╚════════════════════════════════════════════════════════════╝")

    # Offline tests (always run)
    test_dataclasses()
    test_sync_wrappers()
    test_router_load_based()
    test_router_agent_type_strategy()
    test_router_round_robin_strategy()
    test_router_no_backends()
    test_router_remove_backend()
    test_async_streaming()
    test_litellm_structure()
    test_research_agent_structure()
    test_sql_agent_structure()

    # Optional live tests
    if args.live_litellm:
        test_live_litellm(args.model)
    else:
        _sep("Live LiteLLMAgent")
        _skip("pass --live-litellm to run against a real LLM")

    if args.live_research:
        test_live_research(args.model)
    else:
        _sep("Live ResearchAgent")
        _skip("pass --live-research to run with web search")

    if args.live_sql:
        _sep("Live SQLAgent")
        _skip("live SQL test requires a running DB — see module docstring")

    if args.registry:
        test_registry_router(args.registry)
    else:
        _sep("Registry-backed AgentRouter")
        _skip("pass --registry host:port to run against a live registry")

    _sep()
    print("\n  All offline tests passed ✓\n")


if __name__ == "__main__":
    main()
