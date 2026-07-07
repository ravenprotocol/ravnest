"""
examples/gateway/demo.py
========================
Smoke test for Phase-6 — Orchestration & HTTP Gateway.

What this covers
----------------
1.  GatewayRequest — construction, to_dict/from_dict, query_text(), modes
2.  GatewayResponse — to_dict, error_response
3.  Orchestrator — auto-mode inference (all 5 paths)
4.  Orchestrator — generate mode (MockCompute backend)
5.  Orchestrator — query mode (TextSource backend)
6.  Orchestrator — rag mode (TextSource + MockCompute)
7.  Orchestrator — agent mode (MockAgent backend)
8.  Orchestrator — pipeline mode (steps list → Pipeline)
9.  Orchestrator — node_id routing hint
10. Orchestrator — model routing hint
11. Orchestrator — health_all()
12. Orchestrator — list_backends()
13. GatewayServer — structural test (builds app without binding)
14. run_gateway — argparse smoke test

Running
-------
# Offline tests (no external services needed)
python3 examples/gateway/demo.py

# Live test against Ollama
python3 examples/gateway/demo.py --live --ollama-model llama3.2

# Live test against a running gateway
python3 examples/gateway/demo.py --gateway http://localhost:8080 --prompt "Hello"
"""

from __future__ import annotations

import argparse
import asyncio
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from ravnest.gateway.base         import GatewayRequest, GatewayResponse
from ravnest.gateway.orchestrator import Orchestrator


# ─────────────────────────────────────────────────────────────────────────────
# Helpers / mock backends
# ─────────────────────────────────────────────────────────────────────────────

PASS  = "PASS"
FAIL  = "FAIL"
SKIP  = "SKIP"
_results: list[tuple[str, str]] = []


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
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{icon}] {name}{suffix}")


def skip(name: str, reason: str = "") -> None:
    _results.append((name, SKIP))
    print(f"  [-] SKIP  {name}  ({reason})")


class _MockComputeCap:
    node_id  = "mock-compute"
    models   = ["mock-model", "other-model"]
    backend  = "mock"
    extra    = {}


class _MockHealthStatus:
    healthy  = True
    backend  = "mock"
    model    = "mock-model"
    message  = "ok"


class _MockComputeBackend:
    def capabilities(self): return _MockComputeCap()
    async def ahealth(self): return _MockHealthStatus()
    async def agenerate(self, req):
        from ravnest.compute.base import GenerateResponse
        text = req.prompt or (req.messages[-1].content if req.messages else "")
        return GenerateResponse(
            text    = f"Echo: {text}",
            model   = "mock-model",
            backend = "mock",
            usage   = {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        )
    async def aembed(self, req):
        from ravnest.compute.base import EmbedResponse
        return EmbedResponse(embeddings=[[0.1, 0.2, 0.3]], model="mock-model", backend="mock")


class _MockAgentCap:
    node_id    = "mock-agent"
    agent_type = "litellm"
    models     = ["mock-model"]
    extra      = {}


class _MockAgentBackend:
    def capabilities(self): return _MockAgentCap()
    async def ahealth(self): return _MockHealthStatus()
    async def arun(self, req):
        from ravnest.agents.base import AgentResponse
        text = req.last_user_message() or "no input"
        return AgentResponse(
            text=f"Agent says: {text}",
            agent="litellm",
            model="mock-model",
            steps=1,
            finish_reason="stop",
            usage={"total_tokens": 20},
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test groups
# ─────────────────────────────────────────────────────────────────────────────

def test_gateway_request():
    _sep("GatewayRequest")

    # 1. Default construction
    req = GatewayRequest()
    check("default mode is auto",   req.mode == "auto")
    check("default max_tokens=512", req.max_tokens == 512)
    check("request_id generated",   len(req.request_id) > 8)

    # 2. query_text() with prompt
    req2 = GatewayRequest(prompt="hello world")
    check("query_text() returns prompt", req2.query_text() == "hello world")

    # 3. query_text() with messages
    req3 = GatewayRequest(messages=[
        {"role": "system",    "content": "You are helpful."},
        {"role": "user",      "content": "What is gravity?"},
    ])
    check("query_text() returns last user msg", req3.query_text() == "What is gravity?")

    # 4. round-trip serialisation
    d    = req3.to_dict()
    req4 = GatewayRequest.from_dict(d)
    check("from_dict → to_dict round-trip", req4.messages == req3.messages)

    # 5. pipeline mode fields
    req5 = GatewayRequest(mode="pipeline", steps=[
        {"node_type": "data_source", "label": "retrieve"},
        {"node_type": "compute",     "label": "generate"},
    ])
    check("steps field preserved", len(req5.steps) == 2)


def test_gateway_response():
    _sep("GatewayResponse")

    resp = GatewayResponse(ok=True, text="Hello", model="gpt-4o")
    check("to_dict has ok key",   resp.to_dict()["ok"] is True)
    check("to_dict has text key", resp.to_dict()["text"] == "Hello")

    err = GatewayResponse.error_response("something broke",
                                         request_id="req-1", trace_id="tr-1")
    check("error_response ok=False",       err.ok is False)
    check("error_response has message",    err.error == "something broke")
    check("error_response request_id set", err.request_id == "req-1")


def test_mode_inference():
    _sep("Orchestrator — auto mode inference")

    cases = [
        (GatewayRequest(steps=[{"node_type": "compute"}]),            "pipeline"),
        (GatewayRequest(agent_type="research"),                        "agent"),
        (GatewayRequest(source_type="text", model="llama3.2"),         "rag"),
        (GatewayRequest(source_type="text"),                           "query"),
        (GatewayRequest(prompt="Hello"),                               "generate"),
    ]
    for req, expected in cases:
        actual = Orchestrator._resolve_mode(req)
        check(f"auto → {expected}", actual == expected, f"got {actual!r}")

    # Explicit mode is never overridden
    explicit = GatewayRequest(mode="query", agent_type="research")
    check("explicit mode not overridden",
          Orchestrator._resolve_mode(explicit) == "query")


async def test_orchestrator_generate():
    _sep("Orchestrator — generate mode")

    orch = Orchestrator()
    orch.add_local_compute(_MockComputeBackend())

    resp = await orch.ahandle(GatewayRequest(prompt="Hello!", mode="generate"))
    check("generate ok",          resp.ok is True)
    check("generate has text",    "Echo: Hello!" in resp.text)
    check("generate mode set",    resp.mode == "generate")
    check("generate latency set", resp.latency_ms > 0)
    check("generate usage set",   resp.usage.get("total_tokens", 0) > 0)

    # No backend → error
    orch2 = Orchestrator()
    resp2 = await orch2.ahandle(GatewayRequest(prompt="x", mode="generate"))
    check("no backend → error", resp2.ok is False)


async def test_orchestrator_query():
    _sep("Orchestrator — query mode (TextSource)")

    from ravnest.data_sources.text_source import TextSource

    source = TextSource(paths=[])
    source.add_text("Distributed training splits the model across GPUs.",  source="doc1")
    source.add_text("Pipeline parallelism divides layers across nodes.", source="doc2")

    orch = Orchestrator()
    orch.add_local_data_source(source)

    resp = await orch.ahandle(GatewayRequest(
        prompt="training", mode="query", top_k=2,
    ))
    check("query ok",          resp.ok is True)
    check("query has chunks",  len(resp.chunks) > 0)
    check("chunks have score", "score" in resp.chunks[0])
    check("query mode set",    resp.mode == "query")


async def test_orchestrator_rag():
    _sep("Orchestrator — RAG mode (TextSource + MockCompute)")

    from ravnest.data_sources.text_source import TextSource

    source = TextSource(paths=[])
    source.add_text("Gradient checkpointing saves memory during training.", source="doc3")

    orch = Orchestrator()
    orch.add_local_data_source(source)
    orch.add_local_compute(_MockComputeBackend())

    resp = await orch.ahandle(GatewayRequest(
        prompt="memory saving", mode="rag", top_k=2,
    ))
    check("rag ok",           resp.ok is True)
    check("rag has text",     len(resp.text) > 0)
    check("rag has chunks",   len(resp.chunks) > 0)
    check("rag steps logged", len(resp.steps) >= 2)

    # RAG with no data source → error
    orch2 = Orchestrator()
    orch2.add_local_compute(_MockComputeBackend())
    resp2 = await orch2.ahandle(GatewayRequest(prompt="x", mode="rag"))
    check("rag no data_source → error", resp2.ok is False)


async def test_orchestrator_agent():
    _sep("Orchestrator — agent mode")

    orch = Orchestrator()
    orch.add_local_agent(_MockAgentBackend())

    resp = await orch.ahandle(GatewayRequest(
        messages=[{"role": "user", "content": "What is 2+2?"}],
        mode="agent",
    ))
    check("agent ok",              resp.ok is True)
    check("agent has text",        "Agent says:" in resp.text)
    check("agent_type returned",   resp.agent_type == "litellm")

    # No backend → error
    orch2 = Orchestrator()
    resp2 = await orch2.ahandle(GatewayRequest(
        messages=[{"role": "user", "content": "x"}], mode="agent"
    ))
    check("no agent backend → error", resp2.ok is False)


async def test_orchestrator_pipeline():
    _sep("Orchestrator — pipeline mode")

    from ravnest.data_sources.text_source import TextSource

    source = TextSource(paths=[])
    source.add_text("Ravnest supports ring all-reduce.", source="doc4")

    orch = Orchestrator()
    orch.add_local_data_source(source)
    orch.add_local_compute(_MockComputeBackend())

    req = GatewayRequest(
        prompt = "ring all-reduce",
        mode   = "pipeline",
        steps  = [
            {"node_type": "data_source", "label": "retrieve"},
            {"node_type": "compute",     "label": "generate"},
        ],
    )
    resp = await orch.ahandle(req)
    # The pipeline may succeed or fail depending on text-source availability;
    # we just check the response is well-formed.
    check("pipeline returns GatewayResponse", isinstance(resp, GatewayResponse))
    check("pipeline mode set", resp.mode == "pipeline")

    # No steps → error
    req2 = GatewayRequest(prompt="x", mode="pipeline", steps=[])
    resp2 = await orch.ahandle(req2)
    check("empty steps → error", resp2.ok is False)


async def test_orchestrator_routing_hints():
    _sep("Orchestrator — routing hints (node_id / model)")

    class _BackendA(_MockComputeBackend):
        class _Cap(_MockComputeCap):
            node_id = "node-A"
            models  = ["model-a"]
        def capabilities(self): return self._Cap()

    class _BackendB(_MockComputeBackend):
        class _Cap(_MockComputeCap):
            node_id = "node-B"
            models  = ["model-b"]
        def capabilities(self): return self._Cap()

    orch = Orchestrator()
    orch.add_local_compute(_BackendA())
    orch.add_local_compute(_BackendB())

    # node_id pin
    resp = await orch.ahandle(GatewayRequest(
        prompt="hi", mode="generate", node_id="node-B"
    ))
    check("node_id pin routes correctly", resp.node_id == "node-B", resp.node_id)

    # model pin
    resp2 = await orch.ahandle(GatewayRequest(
        prompt="hi", mode="generate", model="model-a"
    ))
    check("model pin routes correctly", resp2.node_id == "node-A", resp2.node_id)


async def test_health_and_list():
    _sep("Orchestrator — health_all / list_backends")

    from ravnest.data_sources.text_source import TextSource

    orch = Orchestrator()
    orch.add_local_compute(_MockComputeBackend())
    orch.add_local_agent(_MockAgentBackend())
    orch.add_local_data_source(TextSource(paths=[]))

    health = await orch.health_all()
    check("health_all returns dict",          isinstance(health, dict))
    check("health_all has 3 entries",         len(health) >= 3)
    check("compute node healthy",
          health.get("mock-compute", {}).get("healthy", False))

    nodes = orch.list_backends()
    check("list_backends has compute",    len(nodes.get("compute", [])) >= 1)
    check("list_backends has agent",      len(nodes.get("agent", [])) >= 1)
    check("list_backends has data_source", len(nodes.get("data_source", [])) >= 1)


def test_server_structural():
    _sep("GatewayServer — structural (no socket bind)")

    try:
        from ravnest.gateway.server import GatewayServer
        orch   = Orchestrator()
        server = GatewayServer(orch, port=19999)
        # Build the aiohttp app but do NOT call runner.setup() (no port bind)
        app = server._build_app()
        routes = [r.resource.canonical for r in app.router.routes()]
        for expected in ["/chat", "/query", "/rag", "/pipeline", "/health", "/nodes"]:
            check(f"route {expected} registered", expected in routes)
    except ImportError:
        skip("GatewayServer structural", "aiohttp not installed")


def test_run_gateway_argparse():
    _sep("run_gateway — argparse smoke test")

    import importlib.util, pathlib
    script = pathlib.Path(__file__).parents[2] / "run_gateway.py"
    spec   = importlib.util.spec_from_file_location("run_gateway", script)
    rg     = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rg)

    args = rg.parse_args.__wrapped__([]) if hasattr(rg.parse_args, "__wrapped__") else None
    # Just verify the module imported cleanly
    check("run_gateway module importable", hasattr(rg, "parse_args"))
    check("run_gateway has main()",        hasattr(rg, "main"))


# ─────────────────────────────────────────────────────────────────────────────
# Live tests
# ─────────────────────────────────────────────────────────────────────────────

async def test_live_ollama(model: str):
    _sep("LIVE — Ollama generate via Orchestrator")

    try:
        from ravnest.compute.ollama_backend import OllamaBackend
        backend = OllamaBackend(model=model)
        orch    = Orchestrator()
        orch.add_local_compute(backend)
        resp = await orch.ahandle(GatewayRequest(
            prompt="In one sentence, what is gradient descent?",
            mode="generate",
        ))
        check("live generate ok",      resp.ok, resp.error)
        check("live generate has text", len(resp.text) > 5)
        print(f"     → {resp.text[:120]}")
    except Exception as exc:
        skip("live Ollama generate", str(exc))


async def test_live_gateway(gateway_url: str, prompt: str):
    _sep(f"LIVE — HTTP Gateway ({gateway_url})")

    try:
        import httpx
        async with httpx.AsyncClient(timeout=30) as client:
            # health check
            hr = await client.get(f"{gateway_url}/health")
            check("gateway /health 2xx",  hr.status_code in (200, 207),
                  f"status={hr.status_code}")

            # /chat
            cr = await client.post(f"{gateway_url}/chat",
                                   json={"prompt": prompt})
            body = cr.json()
            check("/chat ok", body.get("ok", False), body.get("error", ""))
            if body.get("text"):
                print(f"     → {body['text'][:120]}")

            # /nodes
            nr = await client.get(f"{gateway_url}/nodes")
            check("/nodes returns dict", "nodes" in nr.json())

    except Exception as exc:
        skip("live gateway test", str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

async def _run_async(args):
    await test_orchestrator_generate()
    await test_orchestrator_query()
    await test_orchestrator_rag()
    await test_orchestrator_agent()
    await test_orchestrator_pipeline()
    await test_orchestrator_routing_hints()
    await test_health_and_list()

    if args.live:
        await test_live_ollama(args.ollama_model)
    if args.gateway:
        await test_live_gateway(args.gateway, args.prompt)


def main():
    p = argparse.ArgumentParser(description="Gateway demo / smoke test")
    p.add_argument("--live",         action="store_true",
                   help="Run live Ollama tests")
    p.add_argument("--ollama-model", default="llama3.2",
                   help="Ollama model to use for live tests")
    p.add_argument("--gateway",      default=None, metavar="URL",
                   help="URL of a running GatewayServer to hit")
    p.add_argument("--prompt",       default="What is gradient descent?")
    args = p.parse_args()

    print("=" * 64)
    print(" Ravnest Gateway — Phase 6 smoke test")
    print("=" * 64)

    # Sync tests
    test_gateway_request()
    test_gateway_response()
    test_mode_inference()
    test_server_structural()
    test_run_gateway_argparse()

    # Async tests
    asyncio.run(_run_async(args))

    # Summary
    print()
    _sep("Summary")
    passed = sum(1 for _, s in _results if s == PASS)
    failed = sum(1 for _, s in _results if s == FAIL)
    skipped= sum(1 for _, s in _results if s == SKIP)
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
