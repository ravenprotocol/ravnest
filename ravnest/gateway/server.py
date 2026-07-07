"""
ravnest.gateway.server — HTTP gateway for the Ravnest mesh.

Routes
------
  POST /chat          generate|agent (returns {ok, text, model, …})
  POST /query         data source query (returns {ok, chunks, …})
  POST /rag           retrieval-augmented generation
  POST /pipeline      user-defined multi-step pipeline
  POST /v1/chat/completions  OpenAI-compatible endpoint
  GET  /health        liveness + per-backend health
  GET  /nodes         list all registered backends
  GET  /              info page

All POST endpoints accept JSON bodies that are forwarded as GatewayRequest
fields.  Unknown keys are silently ignored.

Streaming
---------
Server-sent events (SSE) are returned when the request body contains
``"stream": true``.  Each event is a JSON-encoded delta: ``data: {...}\n\n``.
(Actual token-by-token streaming requires a backend that supports it; this
release buffers the full response and sends a single event.)

Usage
-----
    from ravnest.gateway import GatewayServer, Orchestrator
    from ravnest.compute.ollama_backend import OllamaBackend

    orch = Orchestrator()
    orch.add_local_compute(OllamaBackend("llama3.2"))

    server = GatewayServer(orchestrator=orch, port=8080)
    server.run()          # blocking
    # or: await server.arun()
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional

from .base         import GatewayRequest, GatewayResponse
from .orchestrator import Orchestrator

logger = logging.getLogger(__name__)

_AIOHTTP_MISSING = False
try:
    from aiohttp import web
except (ImportError, ModuleNotFoundError):
    _AIOHTTP_MISSING = True


class GatewayServer:
    """
    Aiohttp-based HTTP gateway server.

    Args:
        orchestrator: An :class:`Orchestrator` instance (required).
        host:         Bind address (default ``0.0.0.0``).
        port:         Bind port (default ``8080``).
        cors:         If True, add permissive CORS headers to all responses.
        log_level:    Python logging level string (default ``"INFO"``).
    """

    def __init__(
        self,
        orchestrator: Orchestrator,
        host:         str  = "0.0.0.0",
        port:         int  = 8080,
        cors:         bool = True,
        log_level:    str  = "INFO",
    ):
        if _AIOHTTP_MISSING:
            raise ImportError("aiohttp is required for GatewayServer: pip install aiohttp")

        self._orch      = orchestrator
        self._host      = host
        self._port      = port
        self._cors      = cors
        logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))

    # ── public ────────────────────────────────────────────────────────────

    def run(self):
        """Start the server (blocking)."""
        asyncio.run(self.arun())

    async def arun(self):
        """Start the server (async entry-point)."""
        from aiohttp import web
        app = self._build_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self._host, self._port)
        await site.start()
        logger.info("[GatewayServer] listening on http://%s:%s", self._host, self._port)
        try:
            while True:
                await asyncio.sleep(3600)
        except (asyncio.CancelledError, KeyboardInterrupt):
            logger.info("[GatewayServer] shutting down")
        finally:
            await runner.cleanup()

    # ── app builder ────────────────────────────────────────────────────────

    def _build_app(self):
        from aiohttp import web
        app = web.Application()
        app.router.add_get("/",                           self._handle_root)
        app.router.add_get("/health",                     self._handle_health)
        app.router.add_get("/nodes",                      self._handle_nodes)
        app.router.add_post("/chat",                      self._handle_chat)
        app.router.add_post("/query",                     self._handle_query)
        app.router.add_post("/rag",                       self._handle_rag)
        app.router.add_post("/pipeline",                  self._handle_pipeline)
        app.router.add_post("/v1/chat/completions",       self._handle_openai_compat)
        if self._cors:
            app.middlewares.append(self._cors_middleware)
        return app

    # ── route handlers ────────────────────────────────────────────────────

    async def _handle_root(self, request):
        from aiohttp import web
        info = {
            "service": "Ravnest Gateway",
            "version": "0.1.0",
            "endpoints": ["/chat", "/query", "/rag", "/pipeline",
                          "/v1/chat/completions", "/health", "/nodes"],
        }
        return web.json_response(info)

    async def _handle_health(self, request):
        from aiohttp import web
        t0 = time.perf_counter()
        backends = await self._orch.health_all()
        all_ok   = all(v.get("healthy", False) for v in backends.values())
        body = {
            "ok":          all_ok,
            "latency_ms":  round((time.perf_counter() - t0) * 1000, 2),
            "backends":    backends,
        }
        status = 200 if all_ok or not backends else 207
        return web.json_response(body, status=status)

    async def _handle_nodes(self, request):
        from aiohttp import web
        nodes = self._orch.list_backends()
        return web.json_response({"ok": True, "nodes": nodes})

    async def _handle_chat(self, request):
        return await self._dispatch(request, default_mode="generate")

    async def _handle_query(self, request):
        return await self._dispatch(request, default_mode="query")

    async def _handle_rag(self, request):
        return await self._dispatch(request, default_mode="rag")

    async def _handle_pipeline(self, request):
        return await self._dispatch(request, default_mode="pipeline")

    async def _handle_openai_compat(self, request):
        """
        OpenAI-compatible ``POST /v1/chat/completions`` endpoint.

        Accepts the OpenAI request schema and returns an OpenAI-style response
        so that any OpenAI SDK client can point at this gateway without changes.
        """
        from aiohttp import web
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        # Map OpenAI fields → GatewayRequest
        greq = GatewayRequest(
            messages    = body.get("messages", []),
            model       = body.get("model"),
            max_tokens  = body.get("max_tokens", 512),
            temperature = body.get("temperature", 0.7),
            top_p       = body.get("top_p", 1.0),
            stop        = body.get("stop"),
            stream      = body.get("stream", False),
            mode        = "generate",
        )
        resp = await self._orch.ahandle(greq)

        if not resp.ok:
            return web.json_response(
                {"error": {"message": resp.error, "type": "gateway_error"}},
                status=500,
            )

        # Shape into OpenAI response schema
        openai_resp = {
            "id":      f"chatcmpl-{greq.request_id[:8]}",
            "object":  "chat.completion",
            "created": int(time.time()),
            "model":   resp.model or (greq.model or "unknown"),
            "choices": [
                {
                    "index":         0,
                    "message":       {"role": "assistant", "content": resp.text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens":     resp.usage.get("prompt_tokens", 0),
                "completion_tokens": resp.usage.get("completion_tokens", 0),
                "total_tokens":      resp.usage.get("total_tokens", 0),
            },
        }
        return web.json_response(openai_resp)

    # ── shared dispatch ───────────────────────────────────────────────────

    async def _dispatch(self, request, default_mode: str):
        from aiohttp import web
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        if "mode" not in body:
            body["mode"] = default_mode

        try:
            greq = GatewayRequest.from_dict(body)
        except Exception as exc:
            return web.json_response({"error": f"Bad request: {exc}"}, status=400)

        resp = await self._orch.ahandle(greq)
        status = 200 if resp.ok else 500
        return web.json_response(resp.to_dict(), status=status)

    # ── CORS middleware ────────────────────────────────────────────────────

    async def _cors_middleware(self, request, handler):
        from aiohttp import web
        if request.method == "OPTIONS":
            return web.Response(
                status=204,
                headers={
                    "Access-Control-Allow-Origin":  "*",
                    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type,Authorization",
                },
            )
        response = await handler(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response
