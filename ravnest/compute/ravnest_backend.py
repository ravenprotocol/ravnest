"""
RavnestBackend — wraps the local InferenceEngine pipeline.

This backend is designed to run on every node in the pipeline cluster.
Only the ROOT node's ``generate()`` returns text; STEM/LEAF nodes participate
in the distributed forward pass and return ``None``.

Usage (on every node, launched with torchrun):

    node   = Node(model=model, ..., registry_address="reg:50099")
    engine = InferenceEngine(node, tokenizer)
    backend = RavnestBackend(engine)
    backend.register_with_registry("reg:50099")

    # ROOT node drives generation; STEM/LEAF block inside check_forward_buffer
    outputs = backend.generate(GenerateRequest(
        messages=[Message("user", "What is the capital of France?")],
        max_tokens=64,
    ))
    if outputs:
        print(outputs.text)
"""

from __future__ import annotations

import socket
import time
from typing import AsyncIterator, List, Optional

from .base import (
    ComputeBackend, ComputeCapability, EmbedRequest, EmbedResponse,
    GenerateRequest, GenerateResponse, HealthStatus,
)


class RavnestBackend(ComputeBackend):
    """
    ComputeBackend wrapper around a running Ravnest InferenceEngine.

    Args:
        inference_engine: A fully initialised ``InferenceEngine`` instance.
        node_id:          Optional override for the registry node_id.
                          Defaults to ``ravnest_rank{rank}_{hostname}``.
    """

    def __init__(self, inference_engine, node_id: Optional[str] = None):
        self._engine   = inference_engine
        self._node     = inference_engine.node
        self._comm     = inference_engine.comm_session
        self._node_id  = node_id or (
            f"ravnest_rank{self._comm.rank}_{socket.gethostname()}"
        )
        self._address  = getattr(self._node, "local_address", None) or \
                         f"{socket.gethostname()}:0"

        from ..strings import NodeTypes
        self._is_root = (self._node.node_type == NodeTypes.ROOT)

    # ── native sync implementation ────────────────────────────────────────

    def generate(self, request: GenerateRequest) -> Optional[GenerateResponse]:
        """
        Run a generation pass through the distributed pipeline.

        On ROOT: tokenises, runs generate, returns ``GenerateResponse``.
        On STEM/LEAF: participates in the pipeline and returns ``None``.
        """
        from ..strings import NodeTypes

        prompts       = self._request_to_prompts(request)
        max_seq_lens  = [request.max_tokens] * len(prompts)

        t0 = time.perf_counter()
        outputs = self._engine.generate(
            prompt_list   = prompts,
            max_seq_lengths = max_seq_lens,
            top_k         = request.top_k,
            temperature   = request.temperature,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        if not self._is_root or outputs is None:
            return None

        text = outputs[0] if len(prompts) == 1 else outputs
        return GenerateResponse(
            text          = text if isinstance(text, str) else "\n---\n".join(text),
            model         = type(self._node.model).__name__,
            backend       = "ravnest",
            request_id    = request.request_id,
            finish_reason = "length" if request.max_tokens else "stop",
            latency_ms    = latency_ms,
        )

    def generate_stream(self, request: GenerateRequest):
        """
        Ravnest InferenceEngine doesn't support per-token streaming;
        yield the full response as a single chunk.
        """
        resp = self.generate(request)
        if resp is not None:
            yield resp.text

    # ── async shims ───────────────────────────────────────────────────────

    async def agenerate(self, request: GenerateRequest) -> Optional[GenerateResponse]:
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None, self.generate, request
        )

    async def agenerate_stream(self, request: GenerateRequest) -> AsyncIterator[str]:
        resp = await self.agenerate(request)
        if resp is not None:
            yield resp.text

    async def aembed(self, request: EmbedRequest) -> EmbedResponse:
        raise NotImplementedError("RavnestBackend does not support embeddings")

    async def ahealth(self) -> HealthStatus:
        import psutil
        load = {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_percent": psutil.virtual_memory().percent,
        }
        return HealthStatus(
            healthy = True,
            backend = "ravnest",
            model   = type(self._node.model).__name__,
            load    = load,
        )

    # ── capabilities ─────────────────────────────────────────────────────

    def capabilities(self) -> ComputeCapability:
        return ComputeCapability(
            backend             = "ravnest",
            models              = [type(self._node.model).__name__],
            max_context_length  = getattr(self._node.model.config, "max_position_embeddings", 4096),
            supports_streaming  = False,
            supports_embeddings = False,
            node_id             = self._node_id,
            extra               = {
                "rank":       self._comm.rank,
                "world_size": self._comm.world_size,
                "node_type":  self._node.node_type,
                "backend":    self._node.backend,
            },
        )

    # ── registry override ─────────────────────────────────────────────────

    def _build_node_capability(self):
        from ..registry import (
            NodeCapability, NodeType, ComputeSubtype, ResourceSpec,
        )
        cap = self.capabilities()
        return NodeCapability(
            node_id   = self._node_id,
            node_type = NodeType.PIPELINE_COMPUTE,
            subtype   = ComputeSubtype.RAVNEST,
            address   = self._address,
            resources = ResourceSpec.from_system(),
            models    = cap.models,
            metadata  = {
                "mode":              "inference",
                "max_context_length": cap.max_context_length,
                **cap.extra,
            },
        )

    # ── helpers ───────────────────────────────────────────────────────────

    def _request_to_prompts(self, request: GenerateRequest) -> List[str]:
        if request.prompt:
            return [request.prompt]
        if request.messages:
            from .base import messages_to_prompt
            return [messages_to_prompt(request.messages)]
        raise ValueError("GenerateRequest needs prompt or messages")
