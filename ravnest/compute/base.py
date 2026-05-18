"""
Shared data types and abstract base class for all compute backends.

Every backend (ravnest, vllm, sglang, ollama, openai-compat) implements
ComputeBackend so the router can treat them uniformly.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, Iterator, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Message:
    """A single chat message in OpenAI format."""
    role:    str   # "system" | "user" | "assistant"
    content: str


@dataclass
class GenerateRequest:
    """
    Unified generation request accepted by every backend.

    Either ``prompt`` (raw text) or ``messages`` (chat history) must be set.
    Backends that don't support chat will flatten ``messages`` to a prompt.
    """
    prompt:      Optional[str]         = None
    messages:    Optional[List[Message]] = None
    model:       Optional[str]         = None    # override backend default
    max_tokens:  int                   = 256
    temperature: float                 = 1.0
    top_p:       float                 = 1.0
    top_k:       int                   = 50
    stop:        Optional[List[str]]   = None
    stream:      bool                  = False
    request_id:  str                   = field(default_factory=lambda: str(uuid.uuid4()))
    extra:       Dict                  = field(default_factory=dict)  # backend-specific knobs

    def flat_prompt(self) -> str:
        """Return a single text string regardless of whether messages or prompt was set."""
        if self.prompt:
            return self.prompt
        if self.messages:
            return "\n".join(f"{m.role}: {m.content}" for m in self.messages)
        raise ValueError("GenerateRequest must have either prompt or messages")


@dataclass
class GenerateResponse:
    text:          str
    model:         str
    backend:       str
    request_id:    str         = ""
    finish_reason: str         = "stop"    # "stop" | "length" | "error"
    usage:         Dict        = field(default_factory=dict)  # prompt/completion/total tokens
    latency_ms:    float       = 0.0


@dataclass
class EmbedRequest:
    texts:      List[str]
    model:      Optional[str] = None
    request_id: str           = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class EmbedResponse:
    embeddings: List[List[float]]
    model:      str
    backend:    str
    request_id: str  = ""
    usage:      Dict = field(default_factory=dict)


@dataclass
class HealthStatus:
    healthy:  bool
    backend:  str
    model:    Optional[str] = None
    message:  str           = ""
    load:     Dict          = field(default_factory=dict)


@dataclass
class ComputeCapability:
    backend:              str           # ComputeSubtype value
    models:               List[str]     = field(default_factory=list)
    max_context_length:   int           = 4096
    supports_streaming:   bool          = False
    supports_embeddings:  bool          = False
    node_id:              Optional[str] = None
    extra:                Dict          = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class ComputeBackend(ABC):
    """
    Abstract interface every compute backend must implement.

    Sync wrappers (``generate``, ``embed``) are provided by default and simply
    run the async counterparts inside a new event loop.  Backends whose native
    API is synchronous (e.g. RavnestBackend) should override the sync methods
    directly and let the async versions delegate to them.
    """

    # ── abstract ──────────────────────────────────────────────────────────

    @abstractmethod
    async def agenerate(self, request: GenerateRequest) -> GenerateResponse:
        """Generate text (async, native implementation)."""
        ...

    @abstractmethod
    async def agenerate_stream(self, request: GenerateRequest) -> AsyncIterator[str]:
        """Stream generated text token by token (async generator)."""
        ...

    @abstractmethod
    async def aembed(self, request: EmbedRequest) -> EmbedResponse:
        """Produce embeddings (async, native implementation)."""
        ...

    @abstractmethod
    async def ahealth(self) -> HealthStatus:
        """Return liveness + load info."""
        ...

    @abstractmethod
    def capabilities(self) -> ComputeCapability:
        """Return static capability descriptor (always synchronous)."""
        ...

    # ── sync convenience wrappers ─────────────────────────────────────────

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Synchronous generate — runs the async version in a fresh event loop."""
        return _run(self.agenerate(request))

    def generate_stream(self, request: GenerateRequest) -> Iterator[str]:
        """Synchronous streaming — drains the async generator."""
        loop = asyncio.new_event_loop()
        try:
            agen = self.agenerate_stream(request)
            while True:
                try:
                    yield loop.run_until_complete(agen.__anext__())
                except StopAsyncIteration:
                    break
        finally:
            loop.close()

    def embed(self, request: EmbedRequest) -> EmbedResponse:
        return _run(self.aembed(request))

    def health(self) -> HealthStatus:
        return _run(self.ahealth())

    # ── registry helpers ──────────────────────────────────────────────────

    def register_with_registry(
        self,
        registry_address: str,
        heartbeat_interval: float = 10.0,
    ) -> None:
        """
        Register this backend with the Ravnest node registry and start
        sending heartbeats.  Idempotent — safe to call more than once.
        """
        from ..registry import (
            RegistryClient, HeartbeatSender,
            NodeCapability, NodeType, ComputeSubtype, ResourceSpec,
        )
        cap = self._build_node_capability()
        self._registry_client  = RegistryClient(registry_address)
        self._registry_client.register(cap)
        self._heartbeat_sender = HeartbeatSender(
            self._registry_client, cap.node_id, interval=heartbeat_interval
        )
        self._heartbeat_sender.start()

    def deregister_from_registry(self) -> None:
        """Gracefully remove this backend from the registry."""
        sender = getattr(self, "_heartbeat_sender", None)
        if sender:
            sender.stop()
        client = getattr(self, "_registry_client", None)
        if client:
            try:
                cap = self._build_node_capability()
                client.deregister(cap.node_id)
                client.close()
            except Exception:
                pass

    def _build_node_capability(self) -> "NodeCapability":  # type: ignore[return]
        """
        Subclasses should override this to provide a fully populated
        NodeCapability.  The default implementation builds a minimal one from
        ``capabilities()``.
        """
        from ..registry import (
            NodeCapability, NodeType, ComputeSubtype, ResourceSpec,
        )
        import socket
        cap  = self.capabilities()
        return NodeCapability(
            node_id   = cap.node_id or f"{cap.backend}_{socket.gethostname()}",
            node_type = NodeType.STANDALONE_COMPUTE,
            subtype   = cap.backend,
            address   = getattr(self, "_address", f"{socket.gethostname()}:0"),
            resources = ResourceSpec.from_system(),
            models    = cap.models,
            metadata  = {
                "max_context_length":  cap.max_context_length,
                "supports_streaming":  cap.supports_streaming,
                "supports_embeddings": cap.supports_embeddings,
                **cap.extra,
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run(coro):
    """Run a coroutine in a new event loop (safe to call from sync context)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Already inside an event loop (e.g. Jupyter) — use a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def messages_to_prompt(messages: List[Message]) -> str:
    """Flatten a chat history to a single text prompt."""
    return "\n".join(f"{m.role}: {m.content}" for m in messages)
