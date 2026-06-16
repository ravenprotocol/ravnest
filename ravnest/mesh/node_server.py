"""
NodeServer — HTTP server that wraps any Ravnest backend.

A single ``NodeServer`` can host one or more backends of different types
(compute, agent, data_source) and exposes them all over a unified JSON
REST API.  The server:

  POST /message        — handle a NodeMessage, return a NodeResponse
  GET  /health         — health check for all registered backends
  GET  /capabilities   — list capabilities of all registered backends
  GET  /               — welcome / version info

Install:  pip install aiohttp  (or fastapi + uvicorn)
The server auto-detects which ASGI/WSGI library is available.

Usage
-----
    from ravnest.mesh.node_server import NodeServer
    from ravnest.compute.ollama_backend import OllamaBackend
    from ravnest.agents.litellm_agent import LiteLLMAgent
    from ravnest.data_sources.text_source import TextSource

    server = NodeServer(host="0.0.0.0", port=8765)

    # Register any combination of backends
    server.add_compute(OllamaBackend(model="llama3.2"))
    server.add_agent(LiteLLMAgent(model="gpt-4o-mini"))
    server.add_data_source(TextSource(paths=["/data/docs"]))

    # Auto-registers with the registry if address provided
    server.run(registry_address="localhost:50099")
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import time
from typing import Any, Dict, List, Optional

from .base import NodeMessage, NodeResponse

logger = logging.getLogger(__name__)

_AIOHTTP_ERR = "aiohttp is not installed. Run: pip install aiohttp"


class NodeServer:
    """
    HTTP server that wraps Ravnest backends and exposes them as a mesh node.

    Args:
        host:    Bind address (default "0.0.0.0").
        port:    TCP port (default 8765).
        node_id: Override the auto-generated node_id.
        cors:    Enable CORS headers (default True).
    """

    def __init__(
        self,
        host:    str           = "0.0.0.0",
        port:    int           = 8765,
        node_id: Optional[str] = None,
        cors:    bool          = True,
    ):
        self._host    = host
        self._port    = port
        self._node_id = node_id or f"mesh_{socket.gethostname()}_{port}"
        self._cors    = cors

        self._compute_backends:      List[Any] = []
        self._agent_backends:        List[Any] = []
        self._data_source_backends:  List[Any] = []

        self._registry_address: Optional[str] = None

    # ── backend registration ──────────────────────────────────────────────

    def add_compute(self, backend) -> "NodeServer":
        """Register a ComputeBackend with this server."""
        self._compute_backends.append(backend)
        logger.info("Registered compute backend: %s", type(backend).__name__)
        return self

    def add_agent(self, backend) -> "NodeServer":
        """Register an AgentBackend with this server."""
        self._agent_backends.append(backend)
        logger.info("Registered agent backend: %s", type(backend).__name__)
        return self

    def add_data_source(self, backend) -> "NodeServer":
        """Register a DataSourceBackend with this server."""
        self._data_source_backends.append(backend)
        logger.info("Registered data source: %s", type(backend).__name__)
        return self

    # ── run ──────────────────────────────────────────────────────────────

    def run(self, registry_address: Optional[str] = None) -> None:
        """
        Start the HTTP server (blocking).

        Args:
            registry_address: If set, auto-register all backends with the
                              Ravnest node registry on startup.
        """
        self._registry_address = registry_address
        asyncio.run(self._serve())

    async def run_async(self, registry_address: Optional[str] = None) -> None:
        """Async version of run() — use when already inside an event loop."""
        self._registry_address = registry_address
        await self._serve()

    # ── aiohttp server ────────────────────────────────────────────────────

    async def _serve(self) -> None:
        try:
            from aiohttp import web
        except ImportError:
            raise ImportError(_AIOHTTP_ERR)

        if self._registry_address:
            self._register_all()

        app = web.Application()
        app.router.add_post("/message",      self._handle_message)
        app.router.add_get("/health",        self._handle_health)
        app.router.add_get("/capabilities",  self._handle_capabilities)
        app.router.add_get("/",              self._handle_root)

        if self._cors:
            try:
                import aiohttp_cors
                cors = aiohttp_cors.setup(app, defaults={
                    "*": aiohttp_cors.ResourceOptions(
                        allow_credentials=True, expose_headers="*",
                        allow_headers="*", allow_methods="*",
                    )
                })
                for route in list(app.router.routes()):
                    cors.add(route)
            except ImportError:
                pass  # CORS not available, skip

        runner = web.AppRunner(app)
        await runner.setup()
        site   = web.TCPSite(runner, self._host, self._port)
        await site.start()
        logger.info("NodeServer listening on http://%s:%d", self._host, self._port)
        print(f"[NodeServer] Listening on http://{self._host}:{self._port}  "
              f"node_id={self._node_id}")
        try:
            await asyncio.Event().wait()   # run forever
        finally:
            await runner.cleanup()

    # ── HTTP handlers ─────────────────────────────────────────────────────

    async def _handle_root(self, request) -> "web.Response":
        from aiohttp import web
        return web.json_response({
            "service":  "ravnest-mesh-node",
            "node_id":  self._node_id,
            "backends": {
                "compute":     len(self._compute_backends),
                "agent":       len(self._agent_backends),
                "data_source": len(self._data_source_backends),
            },
        })

    async def _handle_health(self, request) -> "web.Response":
        from aiohttp import web
        results = {}
        for b in self._compute_backends:
            hs = await b.ahealth()
            cap = b.capabilities()
            results[cap.node_id] = {"type": "compute", "healthy": hs.healthy,
                                     "message": hs.message}
        for b in self._agent_backends:
            hs = await b.ahealth()
            cap = b.capabilities()
            results[cap.node_id] = {"type": "agent", "healthy": hs.healthy,
                                     "message": hs.message}
        for b in self._data_source_backends:
            hs = await b.ahealth()
            cap = b.capabilities()
            results[cap.node_id] = {"type": "data_source", "healthy": hs.healthy,
                                     "message": hs.message}
        all_ok = all(v["healthy"] for v in results.values()) if results else True
        return web.json_response({"ok": all_ok, "backends": results})

    async def _handle_capabilities(self, request) -> "web.Response":
        from aiohttp import web
        caps = []
        for b in self._compute_backends:
            c = b.capabilities()
            caps.append({"type": "compute", "node_id": c.node_id,
                          "models": c.models, "backend": c.backend})
        for b in self._agent_backends:
            c = b.capabilities()
            caps.append({"type": "agent", "node_id": c.node_id,
                          "agent_type": c.agent_type, "models": c.models,
                          "tools": c.tools})
        for b in self._data_source_backends:
            c = b.capabilities()
            caps.append({"type": "data_source", "node_id": c.node_id,
                          "source_type": c.source_type,
                          "modalities": c.modalities,
                          "item_count": c.item_count})
        return web.json_response({"node_id": self._node_id, "capabilities": caps})

    async def _handle_message(self, request) -> "web.Response":
        from aiohttp import web
        t0 = time.perf_counter()
        try:
            data = await request.json()
            msg  = NodeMessage.from_dict(data)
        except Exception as exc:
            return web.json_response(
                NodeResponse.error_response(f"Invalid request: {exc}").to_dict(),
                status=400,
            )

        resp = await self._dispatch(msg)
        resp.latency_ms = (time.perf_counter() - t0) * 1000
        status = 200 if resp.ok else 500
        return web.json_response(resp.to_dict(), status=status)

    # ── dispatcher ────────────────────────────────────────────────────────

    async def _dispatch(self, msg: NodeMessage) -> NodeResponse:
        """Route a NodeMessage to the right backend and action."""
        try:
            if msg.node_type == "compute":
                return await self._dispatch_compute(msg)
            elif msg.node_type == "agent":
                return await self._dispatch_agent(msg)
            elif msg.node_type == "data_source":
                return await self._dispatch_data_source(msg)
            else:
                return NodeResponse.error_response(
                    f"Unknown node_type: {msg.node_type}",
                    message_id=msg.message_id, trace_id=msg.trace_id,
                )
        except Exception as exc:
            logger.exception("Dispatch error for %s/%s", msg.node_type, msg.action)
            return NodeResponse.error_response(
                str(exc), message_id=msg.message_id, trace_id=msg.trace_id,
            )

    async def _dispatch_compute(self, msg: NodeMessage) -> NodeResponse:
        from ravnest.compute.base import GenerateRequest, EmbedRequest, Message

        backend = self._pick_backend(
            self._compute_backends, msg.node_id,
            lambda b, m: m in (b.capabilities().models or []),
            msg.model,
        )
        if backend is None:
            return NodeResponse.error_response(
                "No compute backend available", message_id=msg.message_id)

        action = msg.action or "generate"

        if action == "generate":
            p = msg.payload
            messages = [Message(role=m["role"], content=m["content"])
                        for m in p.get("messages", [])]
            req  = GenerateRequest(
                prompt      = p.get("prompt", ""),
                messages    = messages or None,
                model       = msg.model or p.get("model"),
                max_tokens  = p.get("max_tokens", 256),
                temperature = p.get("temperature", 1.0),
                top_p       = p.get("top_p", 1.0),
                top_k       = p.get("top_k", 50),
                stop        = p.get("stop"),
            )
            resp = await backend.agenerate(req)
            return NodeResponse(
                ok         = True,
                result     = {"text": resp.text, "model": resp.model,
                              "finish_reason": resp.finish_reason,
                              "usage": resp.usage, "latency_ms": resp.latency_ms},
                message_id = msg.message_id,
                trace_id   = msg.trace_id,
            )

        if action == "embed":
            p   = msg.payload
            req = EmbedRequest(texts=p.get("texts", []),
                               model=p.get("model"))
            resp = await backend.aembed(req)
            return NodeResponse(
                ok         = True,
                result     = {"embeddings": resp.embeddings, "model": resp.model,
                              "usage": resp.usage},
                message_id = msg.message_id,
                trace_id   = msg.trace_id,
            )

        if action == "health":
            hs = await backend.ahealth()
            return NodeResponse(
                ok     = True,
                result = {"healthy": hs.healthy, "backend": hs.backend,
                          "model": hs.model, "message": hs.message},
                message_id = msg.message_id, trace_id = msg.trace_id,
            )

        return NodeResponse.error_response(
            f"Unknown compute action: {action}", message_id=msg.message_id)

    async def _dispatch_agent(self, msg: NodeMessage) -> NodeResponse:
        from ravnest.agents.base import AgentRequest, Message

        backend = self._pick_backend(
            self._agent_backends, msg.node_id,
            lambda b, t: (t is None or b.capabilities().agent_type == t),
            msg.source_type,
        )
        if backend is None:
            return NodeResponse.error_response(
                "No agent backend available", message_id=msg.message_id)

        action = msg.action or "run"

        if action in ("run", "stream"):
            p = msg.payload
            messages = [Message(role=m["role"], content=m["content"])
                        for m in p.get("messages", [])]
            req  = AgentRequest(
                messages    = messages,
                model       = msg.model or p.get("model"),
                max_steps   = p.get("max_steps", 10),
                max_tokens  = p.get("max_tokens", 1024),
                temperature = p.get("temperature", 0.7),
            )
            resp = await backend.arun(req)
            return NodeResponse(
                ok         = True,
                result     = {"text": resp.text, "agent": resp.agent,
                              "model": resp.model, "steps": resp.steps,
                              "finish_reason": resp.finish_reason,
                              "usage": resp.usage, "latency_ms": resp.latency_ms},
                message_id = msg.message_id,
                trace_id   = msg.trace_id,
            )

        if action == "health":
            hs = await backend.ahealth()
            return NodeResponse(
                ok     = True,
                result = {"healthy": hs.healthy, "agent": hs.agent,
                          "model": hs.model, "message": hs.message},
                message_id = msg.message_id, trace_id = msg.trace_id,
            )

        return NodeResponse.error_response(
            f"Unknown agent action: {action}", message_id=msg.message_id)

    async def _dispatch_data_source(self, msg: NodeMessage) -> NodeResponse:
        from ravnest.data_sources.base import DataRequest

        backend = self._pick_backend(
            self._data_source_backends, msg.node_id,
            lambda b, t: (t is None or b.capabilities().source_type == t),
            msg.source_type,
        )
        if backend is None:
            return NodeResponse.error_response(
                "No data source backend available", message_id=msg.message_id)

        action = msg.action or "query"

        if action in ("query", "stream"):
            p   = msg.payload
            req = DataRequest(
                query       = p.get("query", ""),
                vector      = p.get("vector"),
                modality    = p.get("modality", "text"),
                top_k       = p.get("top_k", 5),
                filters     = p.get("filters", {}),
                include_vectors = p.get("include_vectors", False),
            )
            resp = await backend.aquery(req)
            return NodeResponse(
                ok         = True,
                result     = {
                    "chunks": [
                        {"content": c.content, "score": c.score,
                         "source": c.source, "modality": c.modality,
                         "metadata": c.metadata, "chunk_id": c.chunk_id}
                        for c in resp.chunks
                    ],
                    "total_found": resp.total_found,
                    "latency_ms":  resp.latency_ms,
                },
                message_id = msg.message_id,
                trace_id   = msg.trace_id,
            )

        if action == "health":
            hs = await backend.ahealth()
            return NodeResponse(
                ok     = True,
                result = {"healthy": hs.healthy, "source": hs.source,
                          "item_count": hs.item_count, "message": hs.message},
                message_id = msg.message_id, trace_id = msg.trace_id,
            )

        return NodeResponse.error_response(
            f"Unknown data_source action: {action}", message_id=msg.message_id)

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _pick_backend(backends, node_id, type_filter, type_hint):
        """
        Select a backend from the list.

        Priority:
          1. Exact node_id match (if node_id specified).
          2. First backend passing type_filter(backend, type_hint).
          3. First backend in list (fallback).
        """
        if not backends:
            return None
        if node_id:
            for b in backends:
                if b.capabilities().node_id == node_id:
                    return b
        if type_hint:
            for b in backends:
                try:
                    if type_filter(b, type_hint):
                        return b
                except Exception:
                    pass
        return backends[0]

    def _register_all(self) -> None:
        """Register all hosted backends with the Ravnest node registry."""
        try:
            from ravnest.registry import RegistryClient
            from ravnest.registry.capability import NodeCapability, NodeType

            address = f"{socket.gethostname()}:{self._port}"
            client  = RegistryClient(self._registry_address)

            for b in self._compute_backends:
                cap = b.capabilities()
                nc  = NodeCapability(
                    node_id   = cap.node_id,
                    node_type = NodeType.STANDALONE_COMPUTE,
                    subtype   = cap.backend,
                    address   = address,
                    models    = cap.models,
                    metadata  = {"mesh_address": f"http://{address}",
                                 **cap.extra},
                )
                client.register(nc)
                logger.info("Registered compute node %s with registry", cap.node_id)

            for b in self._agent_backends:
                cap = b.capabilities()
                nc  = NodeCapability(
                    node_id   = cap.node_id,
                    node_type = NodeType.AGENT,
                    subtype   = cap.agent_type,
                    address   = address,
                    models    = cap.models,
                    metadata  = {"mesh_address": f"http://{address}",
                                 "tools": cap.tools},
                )
                client.register(nc)
                logger.info("Registered agent node %s with registry", cap.node_id)

            for b in self._data_source_backends:
                cap = b.capabilities()
                nc  = NodeCapability(
                    node_id   = cap.node_id,
                    node_type = NodeType.DATA_SOURCE,
                    subtype   = cap.source_type,
                    address   = address,
                    metadata  = {"mesh_address": f"http://{address}",
                                 "modalities": cap.modalities,
                                 "item_count": cap.item_count},
                )
                client.register(nc)
                logger.info("Registered data source %s with registry", cap.node_id)

        except Exception as exc:
            logger.warning("Failed to register with registry: %s", exc)
