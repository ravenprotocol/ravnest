"""
Orchestrator — coordinates requests across all Ravnest node types.

The Orchestrator holds references to all three routers (compute, agent,
data_source) and a Pipeline builder.  It resolves a GatewayRequest to the
right execution path and returns a GatewayResponse.

Execution paths
---------------
  mode="generate"  → ComputeRouter.generate()
  mode="agent"     → AgentRouter.run()
  mode="query"     → DataRouter.query()
  mode="rag"       → DataRouter.query() → ComputeRouter.generate()
  mode="pipeline"  → Pipeline.run(steps from request)
  mode="auto"      → heuristics to pick one of the above

Auto-mode heuristics
--------------------
  1. If ``steps`` are provided → pipeline mode.
  2. If ``agent_type`` is set  → agent mode.
  3. If ``source_type`` is set and no model → query mode.
  4. If ``source_type`` is set and model     → rag mode.
  5. Default                                 → generate mode.

Local backend injection (no registry required)
----------------------------------------------
    orch = Orchestrator()
    orch.add_local_compute(OllamaBackend("llama3.2"))
    orch.add_local_data_source(TextSource(paths=["/data/docs"]))
    resp = orch.handle(GatewayRequest(prompt="…", mode="rag"))
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from .base import GatewayRequest, GatewayResponse

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Central coordinator for the Ravnest HTTP gateway.

    Can be constructed with registry addresses (for full multi-node routing)
    or with local backends (for single-process / testing usage).

    Args:
        compute_registry:     host:port of the Ravnest registry to discover
                              compute nodes from (optional).
        agent_registry:       Same for agent nodes.
        data_source_registry: Same for data source nodes.
        max_retries:          Retry budget for each router.
        rag_context_header:   System-message prefix for RAG context injection.
    """

    def __init__(
        self,
        compute_registry:     Optional[str] = None,
        agent_registry:       Optional[str] = None,
        data_source_registry: Optional[str] = None,
        max_retries:          int           = 3,
        rag_context_header:   str           = "Use the following context to answer:",
    ):
        self._compute_registry     = compute_registry
        self._agent_registry       = agent_registry
        self._data_source_registry = data_source_registry
        self._max_retries          = max_retries
        self._rag_header           = rag_context_header

        # Lazily initialised routers
        self._compute_router:     Optional[Any] = None
        self._agent_router:       Optional[Any] = None
        self._data_router:        Optional[Any] = None

        # Local backends for non-registry usage
        self._local_compute:      List[Any] = []
        self._local_agents:       List[Any] = []
        self._local_data_sources: List[Any] = []

    # ── local backend injection ───────────────────────────────────────────

    def add_local_compute(self, backend) -> "Orchestrator":
        self._local_compute.append(backend)
        return self

    def add_local_agent(self, backend) -> "Orchestrator":
        self._local_agents.append(backend)
        return self

    def add_local_data_source(self, backend) -> "Orchestrator":
        self._local_data_sources.append(backend)
        return self

    # ── public API ────────────────────────────────────────────────────────

    def handle(self, request: GatewayRequest) -> GatewayResponse:
        """Handle a GatewayRequest synchronously."""
        return _run(self.ahandle(request))

    async def ahandle(self, request: GatewayRequest) -> GatewayResponse:
        """Handle a GatewayRequest asynchronously."""
        t0   = time.perf_counter()
        mode = self._resolve_mode(request)

        logger.info("[Orchestrator] request_id=%s mode=%s", request.request_id, mode)

        try:
            if mode == "generate":
                resp = await self._handle_generate(request)
            elif mode == "agent":
                resp = await self._handle_agent(request)
            elif mode == "query":
                resp = await self._handle_query(request)
            elif mode == "rag":
                resp = await self._handle_rag(request)
            elif mode == "pipeline":
                resp = await self._handle_pipeline(request)
            else:
                resp = GatewayResponse.error_response(
                    f"Unknown mode: {mode}",
                    request_id=request.request_id,
                    trace_id=request.trace_id,
                )
        except Exception as exc:
            logger.exception("[Orchestrator] Unhandled error")
            resp = GatewayResponse.error_response(
                str(exc),
                request_id=request.request_id,
                trace_id=request.trace_id,
            )

        resp.latency_ms = (time.perf_counter() - t0) * 1000
        resp.request_id = request.request_id
        resp.trace_id   = request.trace_id
        resp.mode       = mode
        return resp

    def list_backends(self) -> Dict[str, List[dict]]:
        """Return a summary of all registered backends."""
        result: Dict[str, List[dict]] = {
            "compute": [], "agent": [], "data_source": [],
        }
        for b in self._local_compute:
            cap = b.capabilities()
            result["compute"].append(
                {"node_id": cap.node_id, "models": cap.models,
                 "backend": cap.backend, "source": "local"}
            )
        for b in self._local_agents:
            cap = b.capabilities()
            result["agent"].append(
                {"node_id": cap.node_id, "agent_type": cap.agent_type,
                 "models": cap.models, "source": "local"}
            )
        for b in self._local_data_sources:
            cap = b.capabilities()
            result["data_source"].append(
                {"node_id": cap.node_id, "source_type": cap.source_type,
                 "modalities": cap.modalities, "item_count": cap.item_count,
                 "source": "local"}
            )
        # Add registry-backed entries if routers are initialised
        if self._compute_router:
            for entry in self._compute_router.list_backends():
                result["compute"].append({**entry, "source": "registry"})
        if self._agent_router:
            for entry in self._agent_router.list_backends():
                result["agent"].append({**entry, "source": "registry"})
        if self._data_router:
            for entry in self._data_router.list_backends():
                result["data_source"].append({**entry, "source": "registry"})
        return result

    async def health_all(self) -> Dict[str, Any]:
        """Run health checks on all local backends concurrently."""
        results: Dict[str, Dict] = {}
        coros = []
        labels = []

        for b in self._local_compute:
            cap = b.capabilities()
            coros.append(b.ahealth())
            labels.append(("compute", cap.node_id))
        for b in self._local_agents:
            cap = b.capabilities()
            coros.append(b.ahealth())
            labels.append(("agent", cap.node_id))
        for b in self._local_data_sources:
            cap = b.capabilities()
            coros.append(b.ahealth())
            labels.append(("data_source", cap.node_id))

        if coros:
            health_results = await asyncio.gather(*coros, return_exceptions=True)
            for (ntype, nid), hr in zip(labels, health_results):
                if isinstance(hr, Exception):
                    results[nid] = {"type": ntype, "healthy": False,
                                    "error": str(hr)}
                else:
                    results[nid] = {"type": ntype, "healthy": hr.healthy,
                                    "message": getattr(hr, "message", "")}

        return results

    # ── mode handlers ─────────────────────────────────────────────────────

    async def _handle_generate(self, req: GatewayRequest) -> GatewayResponse:
        """Route to a compute node and return generated text."""
        from ravnest.compute.base import GenerateRequest, Message

        messages = [Message(role=m["role"], content=m["content"])
                    for m in req.messages] if req.messages else None

        greq = GenerateRequest(
            prompt      = req.prompt if not messages else "",
            messages    = messages,
            model       = req.model,
            max_tokens  = req.max_tokens,
            temperature = req.temperature,
            top_p       = req.top_p,
            stop        = req.stop,
        )

        backend = self._pick_compute(req.model, req.node_id)
        if backend is None:
            return GatewayResponse.error_response("No compute backend available")

        resp  = await backend.agenerate(greq)
        cap   = backend.capabilities()
        return GatewayResponse(
            ok      = True,
            text    = resp.text,
            model   = resp.model,
            node_id = cap.node_id,
            usage   = resp.usage or {},
        )

    async def _handle_agent(self, req: GatewayRequest) -> GatewayResponse:
        """Route to an agent node."""
        from ravnest.agents.base import AgentRequest, Message

        messages = [Message(role=m["role"], content=m["content"])
                    for m in req.messages]
        if req.prompt and not any(m.get("role") == "user" for m in req.messages):
            messages.append(Message(role="user", content=req.prompt))

        areq = AgentRequest(
            messages    = messages,
            model       = req.model,
            max_steps   = req.max_steps,
            max_tokens  = req.max_tokens,
            temperature = req.temperature,
            tools       = req.tools,
        )

        backend = self._pick_agent(req.agent_type, req.node_id)
        if backend is None:
            return GatewayResponse.error_response("No agent backend available")

        resp  = await backend.arun(areq)
        cap   = backend.capabilities()
        return GatewayResponse(
            ok         = True,
            text       = resp.text,
            model      = resp.model,
            node_id    = cap.node_id,
            agent_type = cap.agent_type,
            usage      = resp.usage or {},
            metadata   = {"steps": resp.steps,
                          "finish_reason": resp.finish_reason},
        )

    async def _handle_query(self, req: GatewayRequest) -> GatewayResponse:
        """Route to a data source node and return chunks."""
        from ravnest.data_sources.base import DataRequest

        dreq = DataRequest(
            query       = req.query_text(),
            modality    = "text",
            top_k       = req.top_k,
            filters     = req.filters,
            extra       = ({"source_type": req.source_type}
                           if req.source_type else {}),
        )

        backend = self._pick_data_source(req.source_type, req.node_id)
        if backend is None:
            return GatewayResponse.error_response("No data source backend available")

        resp  = await backend.aquery(dreq)
        cap   = backend.capabilities()
        return GatewayResponse(
            ok      = True,
            chunks  = [{"content": c.content, "score": c.score,
                        "source": c.source, "modality": c.modality,
                        "metadata": c.metadata}
                       for c in resp.chunks],
            node_id = cap.node_id,
            metadata = {"total_found": resp.total_found},
        )

    async def _handle_rag(self, req: GatewayRequest) -> GatewayResponse:
        """Retrieve context from a data source, then generate with a compute node."""
        # Step 1: retrieve
        query_resp = await self._handle_query(req)
        if not query_resp.ok:
            return query_resp

        # Step 2: build augmented prompt
        context_parts = [c["content"] for c in query_resp.chunks
                         if c.get("content")]
        context       = "\n\n".join(context_parts)
        augmented     = (f"{self._rag_header}\n\n{context}\n\n"
                         f"Question: {req.query_text()}")

        # Step 3: generate
        from ravnest.compute.base import GenerateRequest
        greq = GenerateRequest(
            prompt      = augmented,
            model       = req.model,
            max_tokens  = req.max_tokens,
            temperature = req.temperature,
            top_p       = req.top_p,
        )
        compute = self._pick_compute(req.model, req.node_id)
        if compute is None:
            return GatewayResponse.error_response(
                "No compute backend available for RAG generation"
            )

        gen_resp = await compute.agenerate(greq)
        cap      = compute.capabilities()
        return GatewayResponse(
            ok      = True,
            text    = gen_resp.text,
            chunks  = query_resp.chunks,
            model   = gen_resp.model,
            node_id = cap.node_id,
            usage   = gen_resp.usage or {},
            steps   = [
                {"label": "retrieval", "chunks": len(query_resp.chunks)},
                {"label": "generation", "model": gen_resp.model},
            ],
        )

    async def _handle_pipeline(self, req: GatewayRequest) -> GatewayResponse:
        """Execute a user-defined pipeline from the request's ``steps`` field."""
        from ravnest.mesh.pipeline import Pipeline
        from ravnest.mesh.base    import PipelineStep

        if not req.steps:
            return GatewayResponse.error_response(
                "mode='pipeline' requires at least one step in 'steps'"
            )

        pipeline = Pipeline(trace_id=req.trace_id)
        for s in req.steps:
            pipeline.step(PipelineStep(
                node_type   = s.get("node_type", "compute"),
                action      = s.get("action"),
                node_id     = s.get("node_id"),
                model       = s.get("model"),
                source_type = s.get("source_type"),
                label       = s.get("label", ""),
                extra       = s.get("extra", {}),
            ))

        # Attach all local backends
        for b in self._local_compute:
            pipeline.add_local_compute(b)
        for b in self._local_agents:
            pipeline.add_local_agent(b)
        for b in self._local_data_sources:
            pipeline.add_local_data_source(b)

        result = await pipeline.arun(req.query_text() or req.messages)

        steps_summary = [
            {"label": lbl, "ok": r.ok,
             "latency_ms": r.latency_ms, "error": r.error}
            for lbl, r in result.steps
        ]
        return GatewayResponse(
            ok      = result.ok,
            text    = result.text(),
            chunks  = result.chunks(),
            steps   = steps_summary,
            error   = result.final.error if not result.ok else "",
        )

    # ── backend selection helpers ─────────────────────────────────────────

    def _pick_compute(self, model: Optional[str],
                      node_id: Optional[str]) -> Optional[Any]:
        """Select a local compute backend, respecting node_id and model hints."""
        # Exact node_id match
        if node_id:
            for b in self._local_compute:
                if b.capabilities().node_id == node_id:
                    return b
        # Model match
        if model:
            for b in self._local_compute:
                if model in (b.capabilities().models or []):
                    return b
        # First available
        if self._local_compute:
            return self._local_compute[0]
        # Registry router fallback (returns a router, not a backend — wrap it)
        if self._compute_router:
            return _RouterBackendAdapter(self._compute_router, "compute")
        return None

    def _pick_agent(self, agent_type: Optional[str],
                    node_id: Optional[str]) -> Optional[Any]:
        if node_id:
            for b in self._local_agents:
                if b.capabilities().node_id == node_id:
                    return b
        if agent_type:
            for b in self._local_agents:
                if b.capabilities().agent_type == agent_type:
                    return b
        if self._local_agents:
            return self._local_agents[0]
        return None

    def _pick_data_source(self, source_type: Optional[str],
                          node_id: Optional[str]) -> Optional[Any]:
        if node_id:
            for b in self._local_data_sources:
                if b.capabilities().node_id == node_id:
                    return b
        if source_type:
            for b in self._local_data_sources:
                if b.capabilities().source_type == source_type:
                    return b
        if self._local_data_sources:
            return self._local_data_sources[0]
        return None

    # ── mode resolution ───────────────────────────────────────────────────

    @staticmethod
    def _resolve_mode(req: GatewayRequest) -> str:
        """Infer the execution mode when mode='auto'."""
        if req.mode != "auto":
            return req.mode
        if req.steps:
            return "pipeline"
        if req.agent_type:
            return "agent"
        if req.source_type and req.model:
            return "rag"
        if req.source_type:
            return "query"
        return "generate"


# ─────────────────────────────────────────────────────────────────────────────
# Thin adapter so a Router can be used where a backend is expected
# ─────────────────────────────────────────────────────────────────────────────

class _RouterBackendAdapter:
    """Wraps a ComputeRouter so it looks like a ComputeBackend."""

    def __init__(self, router, router_type: str):
        self._router = router
        self._type   = router_type

    def capabilities(self):
        class _FakeCap:
            node_id = "router"
            models  = []
            backend = "router"
        return _FakeCap()

    async def agenerate(self, req):
        return await self._router.agenerate(req)

    async def aembed(self, req):
        return await self._router.aembed(req)

    async def ahealth(self):
        return type("H", (), {"healthy": True, "backend": "router",
                              "model": "", "message": ""})()


def _run(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)
