"""
examples/mesh/demo.py
=====================
Smoke test for the Phase-5 Unified Communication Protocol (mesh layer).

What this covers
----------------
1.  NodeMessage — construction, serialisation, convenience constructors
2.  NodeResponse — serialisation, error_response, helper methods
3.  PipelineStep / PipelineResult — dataclass construction and helpers
4.  Pipeline (local backends, no HTTP server needed):
      a. Single data_source step
      b. Single compute step
      c. data_source → compute   (RAG-style two-step)
      d. Custom transform between steps
5.  NodeServer dispatcher — offline (no HTTP port), direct _dispatch() test
6.  NodeClient — structural test (no live server)
7.  Pipeline with agent backend

Running
-------
# Offline tests (no external services needed)
python examples/mesh/demo.py

# Live RAG pipeline against a real Ollama server
python examples/mesh/demo.py --live --ollama-model llama3.2

# Against a live NodeServer
python examples/mesh/demo.py --server http://localhost:8765
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from ravnest.mesh.base import NodeMessage, NodeResponse, PipelineStep, PipelineResult
from ravnest.mesh.pipeline import Pipeline
from ravnest.mesh.node_server import NodeServer
from ravnest.mesh.node_client import NodeClient
from ravnest.data_sources.text_source import TextSource
from ravnest.data_sources.base import DataRequest


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sep(title: str = "") -> None:
    width = 64
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─' * pad} {title} {'─' * pad}")
    else:
        print("─" * width)


def _ok(msg: str) -> None:   print(f"  ✓  {msg}")
def _skip(msg: str) -> None: print(f"  ○  SKIP  {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. NodeMessage & NodeResponse dataclasses
# ─────────────────────────────────────────────────────────────────────────────

def test_node_message() -> None:
    _sep("NodeMessage — construction & serialisation")

    # Basic construction
    msg = NodeMessage(node_type="compute", action="generate",
                      payload={"prompt": "Hello"})
    assert msg.message_id and msg.trace_id   # auto UUIDs
    assert msg.payload["prompt"] == "Hello"
    _ok("NodeMessage basic construction")

    # Round-trip
    d    = msg.to_dict()
    msg2 = NodeMessage.from_dict(d)
    assert msg2.node_type == "compute"
    assert msg2.message_id == msg.message_id
    _ok("NodeMessage to_dict / from_dict round-trip")

    # Convenience constructors
    m1 = NodeMessage.generate(prompt="test", max_tokens=64)
    assert m1.node_type == "compute" and m1.action == "generate"
    assert m1.payload["max_tokens"] == 64
    _ok("NodeMessage.generate() convenience constructor")

    m2 = NodeMessage.agent_run(messages=[{"role": "user", "content": "hi"}],
                               agent_type="research")
    assert m2.node_type == "agent" and m2.action == "run"
    assert m2.source_type == "research"
    _ok("NodeMessage.agent_run() convenience constructor")

    m3 = NodeMessage.data_query("distributed training", top_k=3,
                                source_type="text")
    assert m3.node_type == "data_source" and m3.payload["top_k"] == 3
    _ok("NodeMessage.data_query() convenience constructor")


def test_node_response() -> None:
    _sep("NodeResponse — construction & helpers")

    r = NodeResponse(ok=True, result={"text": "hello"}, message_id="m1",
                     trace_id="t1", latency_ms=12.5)
    assert r.ok and r.result["text"] == "hello"
    _ok("NodeResponse basic construction")

    d  = r.to_dict()
    r2 = NodeResponse.from_dict(d)
    assert r2.latency_ms == 12.5
    _ok("NodeResponse to_dict / from_dict round-trip")

    err = NodeResponse.error_response("something went wrong", "m1", "t1")
    assert not err.ok and err.error == "something went wrong"
    _ok("NodeResponse.error_response()")


def test_pipeline_types() -> None:
    _sep("PipelineStep / PipelineResult — dataclasses")

    step = PipelineStep(node_type="compute", model="llama3.2", label="gen")
    assert step.default_action() == "generate"

    step2 = PipelineStep(node_type="agent")
    assert step2.default_action() == "run"

    step3 = PipelineStep(node_type="data_source")
    assert step3.default_action() == "query"

    _ok("PipelineStep.default_action() per node_type")

    nr = NodeResponse(ok=True, result={"text": "answer", "chunks": []})
    res = PipelineResult(
        steps      = [("retrieval", nr), ("generation", nr)],
        final      = nr,
        trace_id   = "t1",
        latency_ms = 50.0,
        ok         = True,
    )
    assert res.text() == "answer"
    assert res.step_result("retrieval") is nr
    assert res.step_result("missing") is None
    _ok("PipelineResult.text() / step_result() / chunks()")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Pipeline with local data_source backend
# ─────────────────────────────────────────────────────────────────────────────

def _make_text_source() -> TextSource:
    src = TextSource(paths=[], node_id="demo-text")
    src.add_text(
        "Ravnest is a decentralized distributed ML framework that supports "
        "compute, agent, and data source nodes.",
        source="intro.txt",
    )
    src.add_text(
        "Pipeline parallelism splits a model across multiple devices, with "
        "each device holding a subset of the model's layers.",
        source="concepts.txt",
    )
    src.add_text(
        "vLLM uses PagedAttention for high-throughput LLM inference with "
        "minimal memory waste.",
        source="vllm.txt",
    )
    return src


def test_pipeline_data_source_only() -> None:
    _sep("Pipeline — single data_source step")

    src = _make_text_source()
    p   = Pipeline()
    p.step(PipelineStep(node_type="data_source", label="retrieve",
                        extra={"top_k": 2}))
    p.add_local_data_source(src)

    result = p.run("Ravnest framework")
    assert result.ok, f"Pipeline failed: {result.final.error}"
    assert len(result.chunks()) > 0
    _ok(f"data_source step → {len(result.chunks())} chunk(s) retrieved")
    _ok(f"Top chunk: score={result.chunks()[0]['score']}  "
        f"content='{result.chunks()[0]['content'][:50]}…'")


def test_pipeline_two_step_rag() -> None:
    _sep("Pipeline — data_source → compute (RAG)")

    src = _make_text_source()

    # Use a stub compute backend instead of a real LLM
    from ravnest.compute.base import (
        ComputeBackend, ComputeCapability, EmbedRequest, EmbedResponse,
        GenerateRequest, GenerateResponse, HealthStatus,
    )

    class _EchoBackend(ComputeBackend):
        async def agenerate(self, req: GenerateRequest) -> GenerateResponse:
            return GenerateResponse(
                text          = f"[ECHO] {(req.prompt or '')[:80]}",
                model         = "echo",
                backend       = "echo",
                request_id    = req.request_id,
                finish_reason = "stop",
            )
        async def agenerate_stream(self, req):
            yield "[stream]"
        async def aembed(self, req): ...
        async def ahealth(self):
            return HealthStatus(healthy=True, backend="echo", model="echo")
        def capabilities(self):
            return ComputeCapability(backend="echo", models=["echo"],
                                     node_id="echo-node")

    p = Pipeline()
    p.step(PipelineStep(node_type="data_source", label="retrieve",
                        extra={"top_k": 2}))
    p.step(PipelineStep(node_type="compute",     label="generate",
                        extra={"max_tokens": 64}))
    p.add_local_data_source(src)
    p.add_local_compute(_EchoBackend())

    result = p.run("pipeline parallelism")
    assert result.ok, f"Pipeline failed: {result.final.error}"
    assert result.text().startswith("[ECHO]")
    _ok(f"RAG 2-step OK: '{result.text()[:60]}…'")
    _ok(f"Step labels: {[lbl for lbl, _ in result.steps]}")
    _ok(f"Total latency: {result.latency_ms:.1f} ms")


def test_pipeline_custom_transform() -> None:
    _sep("Pipeline — custom inter-step transform")

    src = _make_text_source()

    from ravnest.compute.base import (
        ComputeBackend, ComputeCapability, GenerateRequest, GenerateResponse,
        HealthStatus, EmbedRequest,
    )
    class _UpperBackend(ComputeBackend):
        async def agenerate(self, req: GenerateRequest) -> GenerateResponse:
            return GenerateResponse(
                text=req.prompt.upper(), model="upper", backend="upper",
                request_id=req.request_id)
        async def agenerate_stream(self, req): yield ""
        async def aembed(self, req): ...
        async def ahealth(self):
            return HealthStatus(healthy=True, backend="upper", model="upper")
        def capabilities(self):
            return ComputeCapability(backend="upper", models=["upper"],
                                     node_id="upper-node")

    # Custom transform: take top chunk and upper-case it
    def my_transform(prev_result: dict) -> NodeMessage:
        chunks = prev_result.get("chunks", [])
        text   = chunks[0]["content"] if chunks else "nothing found"
        return NodeMessage(
            node_type = "compute",
            action    = "generate",
            payload   = {"prompt": text[:80], "max_tokens": 32},
        )

    p = Pipeline()
    p.step(PipelineStep(node_type="data_source", label="retrieve",
                        extra={"top_k": 1}))
    p.step(PipelineStep(node_type="compute", label="upper",
                        transform=my_transform))
    p.add_local_data_source(src)
    p.add_local_compute(_UpperBackend())

    result = p.run("vllm")
    assert result.ok, f"Pipeline failed: {result.final.error}"
    assert result.text() == result.text().upper()
    _ok(f"Custom transform + upper backend: '{result.text()[:50]}…'")


def test_pipeline_failed_step() -> None:
    _sep("Pipeline — step failure propagation")

    from ravnest.compute.base import (
        ComputeBackend, ComputeCapability, GenerateRequest, GenerateResponse,
        HealthStatus, EmbedRequest,
    )
    class _BrokenBackend(ComputeBackend):
        async def agenerate(self, req: GenerateRequest) -> GenerateResponse:
            raise RuntimeError("Backend exploded!")
        async def agenerate_stream(self, req): yield ""
        async def aembed(self, req): ...
        async def ahealth(self):
            return HealthStatus(healthy=False, backend="broken", model="x")
        def capabilities(self):
            return ComputeCapability(backend="broken", models=["x"],
                                     node_id="broken-node")

    p = Pipeline()
    p.step(PipelineStep(node_type="compute", label="broken"))
    p.add_local_compute(_BrokenBackend())

    result = p.run("test input")
    assert not result.ok
    assert "Backend exploded!" in result.final.error
    _ok(f"Failed step captured: ok={result.ok}, error='{result.final.error}'")


# ─────────────────────────────────────────────────────────────────────────────
# 3. NodeServer dispatcher (offline — no HTTP port)
# ─────────────────────────────────────────────────────────────────────────────

async def _test_server_dispatch_async() -> None:
    server = NodeServer(port=19999, node_id="test-server")
    server.add_data_source(_make_text_source())

    # Direct dispatch (bypasses HTTP)
    msg  = NodeMessage.data_query("Ravnest", top_k=2)
    resp = await server._dispatch(msg)
    assert resp.ok, f"Dispatch failed: {resp.error}"
    assert len(resp.result["chunks"]) > 0
    _ok(f"NodeServer._dispatch(data_source/query) → {len(resp.result['chunks'])} chunk(s)")

    # Unknown node_type
    bad  = NodeMessage(node_type="unknown", action="test")
    resp2 = await server._dispatch(bad)
    assert not resp2.ok
    _ok(f"Unknown node_type → error: '{resp2.error}'")

    # Health action on data_source
    msg3 = NodeMessage(node_type="data_source", action="health")
    resp3 = await server._dispatch(msg3)
    assert resp3.ok and resp3.result["healthy"]
    _ok(f"NodeServer._dispatch(data_source/health) → healthy={resp3.result['healthy']}")


def test_server_dispatch() -> None:
    _sep("NodeServer — offline dispatcher")
    asyncio.run(_test_server_dispatch_async())


# ─────────────────────────────────────────────────────────────────────────────
# 4. NodeClient structural test
# ─────────────────────────────────────────────────────────────────────────────

def test_node_client_structural() -> None:
    _sep("NodeClient — structural test")
    try:
        import httpx  # noqa: F401
    except ImportError:
        _skip("httpx not installed — pip install httpx")
        return

    client = NodeClient("http://localhost:19999", timeout=1.0, retries=0)
    assert client._base_url == "http://localhost:19999"
    _ok("NodeClient constructed with base_url and timeout")

    # Send to a non-running server — should return error response
    msg  = NodeMessage.data_query("test")
    resp = client.send(msg)
    assert not resp.ok   # server not running
    _ok(f"Failed send returns error NodeResponse: ok={resp.ok}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Pipeline with agent backend
# ─────────────────────────────────────────────────────────────────────────────

def test_pipeline_agent_step() -> None:
    _sep("Pipeline — agent step")

    from ravnest.agents.base import (
        AgentBackend, AgentCapability, AgentHealthStatus,
        AgentRequest, AgentResponse,
    )

    class _EchoAgent(AgentBackend):
        async def arun(self, req: AgentRequest) -> AgentResponse:
            q = req.last_user_message() or ""
            return AgentResponse(
                text   = f"[AGENT] {q[:60]}",
                agent  = "echo",
                model  = "echo",
                request_id = req.request_id,
            )
        async def astream(self, req):
            yield "[stream]"
        async def ahealth(self):
            return AgentHealthStatus(healthy=True, agent="echo")
        def capabilities(self):
            return AgentCapability(agent_type="echo", models=["echo"],
                                   node_id="echo-agent")

    p = Pipeline()
    p.step(PipelineStep(node_type="agent", label="answer"))
    p.add_local_agent(_EchoAgent())

    result = p.run("What is Ravnest?")
    assert result.ok, f"Pipeline failed: {result.final.error}"
    assert result.text().startswith("[AGENT]")
    _ok(f"Agent step: '{result.text()}'")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Live tests (optional)
# ─────────────────────────────────────────────────────────────────────────────

def test_live_rag(ollama_model: str) -> None:
    _sep(f"Live RAG pipeline ({ollama_model})")
    try:
        from ravnest.compute.ollama_backend import OllamaBackend
        import httpx  # noqa: F401
    except ImportError as e:
        _skip(f"Missing dependency: {e}")
        return

    src = _make_text_source()
    llm = OllamaBackend(model=ollama_model, timeout=30.0)

    hs = llm.health()
    if not hs.healthy:
        _skip(f"Ollama not healthy: {hs.message}")
        return

    p = Pipeline()
    p.step(PipelineStep(node_type="data_source", label="retrieve",
                        extra={"top_k": 2}))
    p.step(PipelineStep(node_type="compute",     label="generate",
                        extra={"max_tokens": 128}))
    p.add_local_data_source(src)
    p.add_local_compute(llm)

    result = p.run("What is Ravnest and how does it support ML workloads?")
    if result.ok:
        _ok(f"Live RAG answer ({result.latency_ms:.0f} ms): "
            f"'{result.text()[:80].strip()}…'")
    else:
        _skip(f"Pipeline failed: {result.final.error}")


def test_live_server(server_url: str) -> None:
    _sep(f"Live NodeServer at {server_url}")
    try:
        import httpx  # noqa: F401
    except ImportError:
        _skip("httpx not installed")
        return

    client = NodeClient(server_url, timeout=10.0)
    try:
        caps = client.capabilities()
        _ok(f"node_id={caps.get('node_id')}, "
            f"backends={caps.get('capabilities', [])}")
        health = client.health()
        _ok(f"all healthy={health.get('ok')}")
    except Exception as exc:
        _skip(f"Server unreachable: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Ravnest mesh smoke test")
    parser.add_argument("--live",         action="store_true",
                        help="Run live RAG pipeline (requires Ollama)")
    parser.add_argument("--ollama-model", default="llama3.2")
    parser.add_argument("--server",       default=None,
                        help="URL of a live NodeServer to test against")
    args = parser.parse_args()

    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║    Ravnest · Phase-5 Unified Comm Protocol  smoke test     ║")
    print("╚════════════════════════════════════════════════════════════╝")

    # Offline tests (always run)
    test_node_message()
    test_node_response()
    test_pipeline_types()
    test_pipeline_data_source_only()
    test_pipeline_two_step_rag()
    test_pipeline_custom_transform()
    test_pipeline_failed_step()
    test_server_dispatch()
    test_node_client_structural()
    test_pipeline_agent_step()

    # Optional live tests
    if args.live:
        test_live_rag(args.ollama_model)
    else:
        _sep("Live RAG pipeline")
        _skip("pass --live to run against a real Ollama server")

    if args.server:
        test_live_server(args.server)
    else:
        _sep("Live NodeServer")
        _skip("pass --server http://host:port to test a live NodeServer")

    _sep()
    print("\n  All offline tests passed ✓\n")


if __name__ == "__main__":
    main()
