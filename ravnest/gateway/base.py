"""
ravnest.gateway.base — Request/response types for the HTTP gateway.

Every external client request arrives as a ``GatewayRequest`` and leaves as a
``GatewayResponse``.  The gateway maps these onto the right mesh node type(s)
and returns a unified result regardless of whether the work was done by a
compute node, an agent, a data source, or a multi-step pipeline.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Gateway request
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GatewayRequest:
    """
    Unified inbound request for the Ravnest HTTP gateway.

    Routing hints
    -------------
    ``mode``       controls how the gateway handles the request:
      "auto"       — gateway infers the best mode from the content.
      "generate"   — send directly to a compute node (LLM completion).
      "agent"      — send to an agent node.
      "rag"        — retrieve from data sources, then generate.
      "query"      — retrieve from data sources only.
      "pipeline"   — run a user-specified pipeline (``steps`` field).

    ``agent_type`` filters agent selection ("litellm", "research", "sql", …).
    ``source_type`` filters data source selection ("text", "vector_db", …).
    ``model``      pins the compute/agent model.
    ``node_id``    pins routing to a specific node.

    Pipeline mode
    -------------
    When ``mode="pipeline"``, ``steps`` must be a list of dicts, each with:
      {"node_type": "data_source"|"compute"|"agent", "label": "...", ...}
    These are converted to PipelineStep objects and executed in order.
    """
    # ── content ───────────────────────────────────────────────────────────
    prompt:      str                    = ""
    messages:    List[Dict[str, Any]]   = field(default_factory=list)

    # ── routing ───────────────────────────────────────────────────────────
    mode:        str                    = "auto"   # generate|agent|rag|query|pipeline
    model:       Optional[str]          = None
    node_id:     Optional[str]          = None
    agent_type:  Optional[str]          = None
    source_type: Optional[str]          = None

    # ── generation params ─────────────────────────────────────────────────
    max_tokens:  int                    = 512
    temperature: float                  = 0.7
    top_p:       float                  = 1.0
    stop:        Optional[List[str]]    = None
    stream:      bool                   = False

    # ── RAG params ────────────────────────────────────────────────────────
    top_k:       int                    = 5
    filters:     Dict[str, Any]         = field(default_factory=dict)

    # ── pipeline mode ─────────────────────────────────────────────────────
    steps:       List[Dict[str, Any]]   = field(default_factory=list)

    # ── agent params ─────────────────────────────────────────────────────
    max_steps:   int                    = 10
    tools:       List[Dict[str, Any]]   = field(default_factory=list)

    # ── meta ──────────────────────────────────────────────────────────────
    request_id:  str                    = field(
        default_factory=lambda: str(uuid.uuid4())
    )
    trace_id:    str                    = field(
        default_factory=lambda: str(uuid.uuid4())
    )
    metadata:    Dict[str, Any]         = field(default_factory=dict)

    def query_text(self) -> str:
        """Return the primary query string (prompt or last user message)."""
        if self.prompt:
            return self.prompt
        for m in reversed(self.messages):
            if m.get("role") == "user":
                return m.get("content", "")
        return ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GatewayRequest":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt":      self.prompt,
            "messages":    self.messages,
            "mode":        self.mode,
            "model":       self.model,
            "node_id":     self.node_id,
            "agent_type":  self.agent_type,
            "source_type": self.source_type,
            "max_tokens":  self.max_tokens,
            "temperature": self.temperature,
            "top_p":       self.top_p,
            "stop":        self.stop,
            "stream":      self.stream,
            "top_k":       self.top_k,
            "filters":     self.filters,
            "steps":       self.steps,
            "max_steps":   self.max_steps,
            "tools":       self.tools,
            "request_id":  self.request_id,
            "trace_id":    self.trace_id,
            "metadata":    self.metadata,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Gateway response
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GatewayResponse:
    """
    Unified outbound response from the Ravnest HTTP gateway.

    Attributes
    ----------
    ok:          True if the request was handled successfully.
    text:        Generated text (compute / agent / RAG final answer).
    chunks:      Retrieved data chunks (query / RAG retrieval step).
    mode:        Which mode was actually used to handle the request.
    model:       Which model produced the response.
    node_id:     Which node handled the request.
    agent_type:  Agent type (agent mode only).
    steps:       Ordered list of step summaries for pipeline / RAG mode.
    usage:       Token counts (prompt_tokens, completion_tokens, total_tokens).
    latency_ms:  Total wall-clock time for the request.
    request_id:  Echoed from the GatewayRequest.
    trace_id:    Trace id propagated through the mesh.
    error:       Error message (when ok=False).
    metadata:    Any extra backend-specific info.
    """
    ok:         bool                    = True
    text:       str                     = ""
    chunks:     List[Dict[str, Any]]    = field(default_factory=list)
    mode:       str                     = ""
    model:      str                     = ""
    node_id:    str                     = ""
    agent_type: str                     = ""
    steps:      List[Dict[str, Any]]    = field(default_factory=list)
    usage:      Dict[str, int]          = field(default_factory=dict)
    latency_ms: float                   = 0.0
    request_id: str                     = ""
    trace_id:   str                     = ""
    error:      str                     = ""
    metadata:   Dict[str, Any]          = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok":         self.ok,
            "text":       self.text,
            "chunks":     self.chunks,
            "mode":       self.mode,
            "model":      self.model,
            "node_id":    self.node_id,
            "agent_type": self.agent_type,
            "steps":      self.steps,
            "usage":      self.usage,
            "latency_ms": self.latency_ms,
            "request_id": self.request_id,
            "trace_id":   self.trace_id,
            "error":      self.error,
            "metadata":   self.metadata,
        }

    @classmethod
    def error_response(cls, error: str, request_id: str = "",
                       trace_id: str = "") -> "GatewayResponse":
        return cls(ok=False, error=error,
                   request_id=request_id, trace_id=trace_id)
