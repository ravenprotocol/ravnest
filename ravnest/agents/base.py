"""
ravnest.agents.base — Abstract AgentBackend interface and shared data types.

Every agent in Ravnest (LiteLLM, research, SQL, custom) implements this
interface.  The router and orchestrator only ever talk to ``AgentBackend``
objects, so swapping implementations requires zero changes upstream.

Data flow
---------
caller → AgentRequest → AgentBackend.arun() → AgentResponse
                      ↘ AgentBackend.astream() → AsyncIterator[str]

Tool use (optional)
-------------------
If the agent needs to call tools during a run it populates
``AgentResponse.tool_calls``.  The caller may inspect these and pass
``ToolResult`` objects back via ``AgentRequest.tool_results`` on a
follow-up call.

Usage
-----
    from ravnest.agents.base import AgentRequest, AgentResponse, Message

    req  = AgentRequest(
        messages = [Message("user", "What is the capital of France?")],
        max_steps = 1,
    )
    resp = await my_agent.arun(req)
    print(resp.text)
"""

from __future__ import annotations

import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Shared data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Message:
    """A single turn in a conversation."""
    role:    str   # "system" | "user" | "assistant" | "tool"
    content: str


@dataclass
class ToolCall:
    """
    A tool the agent wants to invoke.

    ``id``   — unique call identifier (echoed back in ToolResult).
    ``name`` — tool / function name.
    ``args`` — parsed argument dict (JSON object decoded).
    ``raw``  — original unparsed string from the model (for debugging).
    """
    id:   str
    name: str
    args: Dict[str, Any] = field(default_factory=dict)
    raw:  str            = ""


@dataclass
class ToolResult:
    """
    The result of executing a ToolCall.

    Pass a list of these back in ``AgentRequest.tool_results`` to continue
    a multi-step agent run.
    """
    tool_call_id: str
    content:      str
    is_error:     bool = False


@dataclass
class AgentRequest:
    """
    Input to an AgentBackend.

    Attributes
    ----------
    messages:      Conversation so far (system + turns).
    tool_results:  Results from a previous step's tool calls (optional).
    model:         Override the agent's default model.
    max_steps:     Maximum ReAct / tool-call iterations (agent-specific).
    max_tokens:    Token budget for each LLM call inside the agent.
    temperature:   Sampling temperature.
    tools:         Extra tool specs to inject (agent-specific format).
    stream:        Whether the caller wants a streaming response.
    request_id:    Caller-supplied idempotency key; auto-generated if None.
    extra:         Arbitrary backend-specific overrides.
    """
    messages:     List[Message]         = field(default_factory=list)
    tool_results: List[ToolResult]      = field(default_factory=list)
    model:        Optional[str]         = None
    max_steps:    int                   = 10
    max_tokens:   int                   = 1024
    temperature:  float                 = 0.7
    tools:        List[Dict[str, Any]]  = field(default_factory=list)
    stream:       bool                  = False
    request_id:   str                   = field(
        default_factory=lambda: str(uuid.uuid4())
    )
    extra:        Dict[str, Any]        = field(default_factory=dict)

    def last_user_message(self) -> Optional[str]:
        """Return the content of the most recent user turn, if any."""
        for m in reversed(self.messages):
            if m.role == "user":
                return m.content
        return None


@dataclass
class AgentResponse:
    """
    Output from an AgentBackend.

    Attributes
    ----------
    text:          Final text answer (may be empty if tool_calls is set).
    agent:         Backend identifier string, e.g. "litellm", "research".
    model:         Model that produced the response.
    request_id:    Echoed from the request.
    finish_reason: "stop" | "tool_calls" | "max_steps" | "error".
    tool_calls:    Tool calls emitted in the last step (caller handles them).
    steps:         Number of ReAct iterations executed.
    usage:         Token counts: prompt_tokens, completion_tokens, total_tokens.
    latency_ms:    Wall-clock time for the full run.
    metadata:      Any extra agent-specific info.
    """
    text:          str
    agent:         str
    model:         str                  = ""
    request_id:    str                  = ""
    finish_reason: str                  = "stop"
    tool_calls:    List[ToolCall]       = field(default_factory=list)
    steps:         int                  = 1
    usage:         Dict[str, int]       = field(default_factory=dict)
    latency_ms:    float                = 0.0
    metadata:      Dict[str, Any]       = field(default_factory=dict)


@dataclass
class AgentHealthStatus:
    """Health / readiness report for an agent node."""
    healthy:   bool
    agent:     str
    model:     str     = ""
    message:   str     = ""
    load:      Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentCapability:
    """
    Describes what an agent can do — used by the registry and router.

    ``agent_type`` is a free-form string, e.g. "litellm", "research", "sql".
    ``tools``       is the list of tool names the agent exposes.
    ``node_id``     matches the registry NodeCapability.node_id.
    """
    agent_type:         str
    models:             List[str]        = field(default_factory=list)
    tools:              List[str]        = field(default_factory=list)
    max_context_length: int              = 4096
    supports_streaming: bool             = False
    node_id:            str              = ""
    extra:              Dict[str, Any]   = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base class
# ─────────────────────────────────────────────────────────────────────────────

def _run(coro):
    """Run a coroutine from sync context, handling already-running loops."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


class AgentBackend(ABC):
    """
    Abstract base for all Ravnest agent backends.

    Sub-classes **must** implement the four async methods.
    Sync wrappers (``run``, ``stream``, ``health``) are provided for
    convenience in scripts and tests.
    """

    # ── async interface (implement these) ────────────────────────────────

    @abstractmethod
    async def arun(self, request: AgentRequest) -> AgentResponse:
        """Execute the agent and return a complete response."""
        ...

    @abstractmethod
    async def astream(self, request: AgentRequest) -> AsyncIterator[str]:
        """Execute the agent and yield text tokens as they are produced."""
        ...

    @abstractmethod
    async def ahealth(self) -> AgentHealthStatus:
        """Return the health / readiness of this agent."""
        ...

    @abstractmethod
    def capabilities(self) -> AgentCapability:
        """Describe what this agent can do."""
        ...

    # ── sync wrappers ────────────────────────────────────────────────────

    def run(self, request: AgentRequest) -> AgentResponse:
        """Synchronous wrapper around ``arun``."""
        return _run(self.arun(request))

    def stream(self, request: AgentRequest):
        """
        Synchronous generator wrapper around ``astream``.

        Usage::

            for token in agent.stream(req):
                print(token, end="", flush=True)
        """
        async def _collect():
            tokens = []
            async for tok in self.astream(request):
                tokens.append(tok)
            return tokens

        return iter(_run(_collect()))

    def health(self) -> AgentHealthStatus:
        """Synchronous wrapper around ``ahealth``."""
        return _run(self.ahealth())

    # ── registry helpers ─────────────────────────────────────────────────

    def register_with_registry(self, registry_address: str) -> None:
        """Register this agent with the Ravnest node registry."""
        from ravnest.registry import RegistryClient, HeartbeatSender
        from ravnest.registry.capability import NodeCapability, NodeType

        cap  = self._build_node_capability()
        node = NodeCapability(
            node_id   = cap.node_id,
            node_type = NodeType.AGENT,
            subtype   = cap.agent_type,
            address   = cap.extra.get("address", ""),
            models    = cap.models,
            metadata  = {
                "tools":              cap.tools,
                "supports_streaming": cap.supports_streaming,
                **cap.extra,
            },
        )
        client = RegistryClient(registry_address)
        client.register(node)
        self._heartbeat_sender = HeartbeatSender(client, cap.node_id)
        self._heartbeat_sender.start()

    def deregister_from_registry(self) -> None:
        if hasattr(self, "_heartbeat_sender"):
            self._heartbeat_sender.stop()
        if hasattr(self, "_registry_client"):
            cap = self._build_node_capability()
            self._registry_client.deregister(cap.node_id)

    def _build_node_capability(self) -> AgentCapability:
        """Return an AgentCapability for registry registration."""
        return self.capabilities()
