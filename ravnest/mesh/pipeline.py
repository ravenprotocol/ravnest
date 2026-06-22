"""
Pipeline — chain Ravnest nodes into a multi-step workflow.

A Pipeline lets you connect compute, agent, and data-source nodes into a
directed sequence.  Each step receives the previous step's output and can
transform it before forwarding to the next node.

Execution modes
---------------
- ``Pipeline.run(input)``        — synchronous, runs all steps in order
- ``await Pipeline.arun(input)`` — async version

Default inter-step transforms
------------------------------
  data_source → compute   :  top-k chunk texts joined as context prepended to the prompt
  data_source → agent     :  same join as context in the system message
  compute     → agent     :  previous response text becomes the user message
  agent       → compute   :  agent text becomes the next prompt
  agent       → agent     :  agent text becomes the next user message
  *           → *         :  raw result dict passed through as payload

Usage
-----
    from ravnest.mesh.pipeline import Pipeline, PipelineStep

    pipeline = (
        Pipeline()
        .step(PipelineStep(
            node_type   = "data_source",
            source_type = "text",
            label       = "retrieval",
        ))
        .step(PipelineStep(
            node_type = "compute",
            model     = "llama3.2",
            label     = "generation",
            extra     = {"max_tokens": 256},
        ))
    )

    # Attach local backends (no server needed)
    from ravnest.data_sources.text_source import TextSource
    from ravnest.compute.ollama_backend   import OllamaBackend

    text_src = TextSource(paths=["/data/docs"])
    llm      = OllamaBackend(model="llama3.2")

    pipeline.add_local_compute(llm)
    pipeline.add_local_data_source(text_src)

    result = pipeline.run("What is pipeline parallelism?")
    print(result.text())
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Union

from .base import NodeMessage, NodeResponse, PipelineResult, PipelineStep

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Sequential pipeline of Ravnest node steps.

    The pipeline routes each step to either:
      1. A local backend (registered via ``add_local_*``), or
      2. A remote ``NodeClient`` (registered via ``add_client``).

    If both are available for a given step, local takes priority.
    """

    def __init__(self, trace_id: Optional[str] = None):
        import uuid
        self._steps:          List[PipelineStep]   = []
        self._trace_id:       str                  = trace_id or str(uuid.uuid4())
        self._compute_locals: List[Any]            = []
        self._agent_locals:   List[Any]            = []
        self._data_locals:    List[Any]            = []
        self._clients:        Dict[str, Any]       = {}  # node_id -> NodeClient

    # ── builder API ───────────────────────────────────────────────────────

    def step(self, step: PipelineStep) -> "Pipeline":
        """Append a step to the pipeline."""
        if not step.label:
            step.label = f"{step.node_type}_{len(self._steps)}"
        self._steps.append(step)
        return self

    def add_local_compute(self, backend) -> "Pipeline":
        self._compute_locals.append(backend)
        return self

    def add_local_agent(self, backend) -> "Pipeline":
        self._agent_locals.append(backend)
        return self

    def add_local_data_source(self, backend) -> "Pipeline":
        self._data_locals.append(backend)
        return self

    def add_client(self, node_id: str, client) -> "Pipeline":
        """Register a NodeClient for a specific node_id."""
        self._clients[node_id] = client
        return self

    # ── execution ─────────────────────────────────────────────────────────

    def run(self, input: Union[str, Dict[str, Any]]) -> PipelineResult:
        """Execute the pipeline synchronously."""
        return _run(self.arun(input))

    async def arun(self, input: Union[str, Dict[str, Any]]) -> PipelineResult:
        """Execute the pipeline asynchronously."""
        t0       = time.perf_counter()
        result   = PipelineResult(trace_id=self._trace_id)
        prev_result: Optional[NodeResponse] = None
        initial_input = input

        for i, step in enumerate(self._steps):
            msg = self._build_message(step, prev_result, initial_input, i)
            msg.trace_id = self._trace_id

            logger.info("[Pipeline] Step %s (%s/%s) …", step.label,
                        step.node_type, step.action or step.default_action())

            resp = await self._execute_step(step, msg)
            result.steps.append((step.label, resp))

            if not resp.ok:
                result.ok = False
                result.final = resp
                result.latency_ms = (time.perf_counter() - t0) * 1000
                logger.warning("[Pipeline] Step %s failed: %s",
                               step.label, resp.error)
                return result

            logger.info("[Pipeline] Step %s OK  (%.0f ms)", step.label,
                        resp.latency_ms)
            prev_result = resp

        result.final      = prev_result
        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    # ── message construction ──────────────────────────────────────────────

    def _build_message(
        self,
        step:          PipelineStep,
        prev:          Optional[NodeResponse],
        initial_input: Union[str, Dict],
        step_idx:      int,
    ) -> NodeMessage:
        """Build a NodeMessage for a step, applying inter-step transforms."""

        # If the step has a custom transform, use it
        if step.transform is not None:
            prev_dict = prev.result if prev else {}
            msg = step.transform(prev_dict)
            if isinstance(msg, NodeMessage):
                return msg
            # If transform returned a dict, wrap it
            return NodeMessage(
                node_type   = step.node_type,
                action      = step.action or step.default_action(),
                payload     = msg,
                node_id     = step.node_id,
                model       = step.model,
                source_type = step.source_type,
            )

        action = step.action or step.default_action()

        # ── First step: use initial_input ─────────────────────────────────
        if step_idx == 0 or prev is None:
            payload = self._initial_payload(step, initial_input)
            return NodeMessage(
                node_type   = step.node_type,
                action      = action,
                payload     = {**payload, **step.extra},
                node_id     = step.node_id,
                model       = step.model,
                source_type = step.source_type,
            )

        # ── Subsequent steps: transform previous output ───────────────────
        payload = self._transform_payload(step, prev)
        return NodeMessage(
            node_type   = step.node_type,
            action      = action,
            payload     = {**payload, **step.extra},
            node_id     = step.node_id,
            model       = step.model,
            source_type = step.source_type,
        )

    @staticmethod
    def _initial_payload(step: PipelineStep,
                         input: Union[str, Dict]) -> Dict:
        if isinstance(input, dict):
            return input
        text = str(input)
        if step.node_type == "data_source":
            return {"query": text, "top_k": step.extra.get("top_k", 5)}
        if step.node_type == "compute":
            return {"prompt": text, "max_tokens": step.extra.get("max_tokens", 256)}
        if step.node_type == "agent":
            return {"messages": [{"role": "user", "content": text}]}
        return {"input": text}

    @staticmethod
    def _transform_payload(step: PipelineStep,
                           prev: NodeResponse) -> Dict:
        """Default inter-step payload construction."""
        r         = prev.result
        prev_type = r.get("_node_type", "")  # not always set

        # Infer previous node type from result structure
        has_text   = "text"   in r
        has_chunks = "chunks" in r and isinstance(r["chunks"], list)

        target = step.node_type

        if has_chunks and target == "compute":
            context = "\n\n".join(
                c["content"] for c in r["chunks"] if c.get("content")
            )
            return {
                "prompt":     context,
                "max_tokens": step.extra.get("max_tokens", 512),
            }

        if has_chunks and target == "agent":
            context = "\n\n".join(
                c["content"] for c in r["chunks"] if c.get("content")
            )
            return {
                "messages": [
                    {"role": "system", "content": f"Context:\n{context}"},
                ],
            }

        if has_text and target == "compute":
            return {
                "prompt":     r["text"],
                "max_tokens": step.extra.get("max_tokens", 512),
            }

        if has_text and target == "agent":
            return {
                "messages": [{"role": "user", "content": r["text"]}],
            }

        if has_text and target == "data_source":
            return {"query": r["text"], "top_k": step.extra.get("top_k", 5)}

        # Generic fallback
        return {"payload": r}

    # ── step execution ────────────────────────────────────────────────────

    async def _execute_step(self, step: PipelineStep,
                             msg: NodeMessage) -> NodeResponse:
        """Execute a single step using local backend or NodeClient."""
        # Try local backend first
        local_resp = await self._try_local(step, msg)
        if local_resp is not None:
            return local_resp

        # Try NodeClient
        client = self._get_client(step)
        if client is not None:
            return await client.asend(msg)

        return NodeResponse.error_response(
            f"No backend or client available for step '{step.label}' "
            f"(node_type={step.node_type})",
            trace_id=msg.trace_id,
        )

    async def _try_local(self, step: PipelineStep,
                          msg: NodeMessage) -> Optional[NodeResponse]:
        """Attempt to execute a step using a local in-process backend."""
        backends = {
            "compute":     self._compute_locals,
            "agent":       self._agent_locals,
            "data_source": self._data_locals,
        }.get(step.node_type, [])

        if not backends:
            return None

        # Pick the right backend
        backend = backends[0]
        for b in backends:
            cap = b.capabilities()
            if step.node_id and cap.node_id == step.node_id:
                backend = b
                break

        try:
            return await _invoke_local(backend, step.node_type, msg)
        except Exception as exc:
            logger.warning("Local backend error at step %s: %s", step.label, exc)
            return NodeResponse.error_response(str(exc), trace_id=msg.trace_id)

    def _get_client(self, step: PipelineStep) -> Optional[Any]:
        if step.node_id and step.node_id in self._clients:
            return self._clients[step.node_id]
        if self._clients:
            return next(iter(self._clients.values()))
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Local backend invocation helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _invoke_local(backend, node_type: str,
                         msg: NodeMessage) -> NodeResponse:
    """Call a local backend and wrap the result in a NodeResponse."""
    import time
    t0 = time.perf_counter()

    if node_type == "compute":
        from ravnest.compute.base import GenerateRequest, EmbedRequest, Message
        p = msg.payload
        messages = [Message(role=m["role"], content=m["content"])
                    for m in p.get("messages", [])]
        req  = GenerateRequest(
            prompt      = p.get("prompt", ""),
            messages    = messages or None,
            model       = msg.model or p.get("model"),
            max_tokens  = p.get("max_tokens", 256),
            temperature = p.get("temperature", 1.0),
        )
        resp = await backend.agenerate(req)
        return NodeResponse(
            ok         = True,
            result     = {"text": resp.text, "model": resp.model,
                          "finish_reason": resp.finish_reason,
                          "usage": resp.usage},
            trace_id   = msg.trace_id,
            latency_ms = (time.perf_counter() - t0) * 1000,
        )

    if node_type == "agent":
        from ravnest.agents.base import AgentRequest, Message
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
                          "usage": resp.usage},
            trace_id   = msg.trace_id,
            latency_ms = (time.perf_counter() - t0) * 1000,
        )

    if node_type == "data_source":
        from ravnest.data_sources.base import DataRequest
        p   = msg.payload
        req = DataRequest(
            query    = p.get("query", ""),
            vector   = p.get("vector"),
            modality = p.get("modality", "text"),
            top_k    = p.get("top_k", 5),
            filters  = p.get("filters", {}),
        )
        resp = await backend.aquery(req)
        return NodeResponse(
            ok         = True,
            result     = {
                "chunks": [
                    {"content": c.content, "score": c.score,
                     "source": c.source, "modality": c.modality,
                     "metadata": c.metadata}
                    for c in resp.chunks
                ],
                "total_found": resp.total_found,
            },
            trace_id   = msg.trace_id,
            latency_ms = (time.perf_counter() - t0) * 1000,
        )

    return NodeResponse.error_response(
        f"Unknown node_type: {node_type}", trace_id=msg.trace_id
    )


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
