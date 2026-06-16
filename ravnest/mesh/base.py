"""
ravnest.mesh.base — Shared types for inter-node communication.

Every message that crosses a node boundary (compute ↔ agent ↔ data_source)
is serialised as a ``NodeMessage``.  Pipelines are expressed as ordered lists
of ``PipelineStep`` objects, each targeting a specific node type and optional
node_id.

Wire format
-----------
All messages are JSON-serialisable dicts.  ``NodeMessage.to_dict()`` /
``NodeMessage.from_dict()`` handle the round-trip.

Node types recognised by the mesh
----------------------------------
  "compute"     → routed via ComputeRouter   → GenerateRequest/Response
  "agent"       → routed via AgentRouter     → AgentRequest/Response
  "data_source" → routed via DataRouter      → DataRequest/Response
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Node message envelope
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NodeMessage:
    """
    Universal message envelope for inter-node communication.

    Attributes
    ----------
    node_type:   Target node type — "compute", "agent", "data_source".
    action:      Node-specific action:
                   compute    → "generate" | "embed" | "health"
                   agent      → "run" | "stream" | "health"
                   data_source→ "query" | "stream" | "health"
    payload:     Action-specific payload dict (serialisable).
    node_id:     Optional target node_id (None = let router pick).
    model:       Optional model hint forwarded to compute/agent.
    source_type: Optional data-source type hint for data_source nodes.
    message_id:  Auto-generated idempotency key.
    trace_id:    Propagated across a multi-hop pipeline for observability.
    metadata:    Arbitrary caller-supplied key-value pairs.
    """
    node_type:   str
    action:      str
    payload:     Dict[str, Any]      = field(default_factory=dict)
    node_id:     Optional[str]       = None
    model:       Optional[str]       = None
    source_type: Optional[str]       = None
    message_id:  str                 = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id:    str                 = field(default_factory=lambda: str(uuid.uuid4()))
    metadata:    Dict[str, Any]      = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type":   self.node_type,
            "action":      self.action,
            "payload":     self.payload,
            "node_id":     self.node_id,
            "model":       self.model,
            "source_type": self.source_type,
            "message_id":  self.message_id,
            "trace_id":    self.trace_id,
            "metadata":    self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NodeMessage":
        return cls(
            node_type   = d["node_type"],
            action      = d["action"],
            payload     = d.get("payload", {}),
            node_id     = d.get("node_id"),
            model       = d.get("model"),
            source_type = d.get("source_type"),
            message_id  = d.get("message_id", str(uuid.uuid4())),
            trace_id    = d.get("trace_id",   str(uuid.uuid4())),
            metadata    = d.get("metadata",   {}),
        )

    # ── convenience constructors ──────────────────────────────────────────

    @classmethod
    def generate(cls, prompt: str = "", messages: List[Dict] = None,
                 model: str = None, max_tokens: int = 256,
                 **kw) -> "NodeMessage":
        """Create a compute/generate message."""
        return cls(
            node_type = "compute",
            action    = "generate",
            payload   = {"prompt": prompt, "messages": messages or [],
                         "max_tokens": max_tokens, **kw},
            model     = model,
        )

    @classmethod
    def agent_run(cls, messages: List[Dict], model: str = None,
                  max_steps: int = 10, agent_type: str = None,
                  **kw) -> "NodeMessage":
        """Create an agent/run message."""
        return cls(
            node_type   = "agent",
            action      = "run",
            payload     = {"messages": messages, "max_steps": max_steps, **kw},
            model       = model,
            source_type = agent_type,
        )

    @classmethod
    def data_query(cls, query: str, top_k: int = 5,
                   source_type: str = None, **kw) -> "NodeMessage":
        """Create a data_source/query message."""
        return cls(
            node_type   = "data_source",
            action      = "query",
            payload     = {"query": query, "top_k": top_k, **kw},
            source_type = source_type,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Node response
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NodeResponse:
    """
    Universal response from a mesh node.

    ``result`` holds the action-specific response object serialised to a dict:
      - compute/generate  → GenerateResponse fields
      - agent/run         → AgentResponse fields
      - data_source/query → DataResponse fields (chunks as list of dicts)
    """
    ok:         bool
    result:     Dict[str, Any]   = field(default_factory=dict)
    error:      str              = ""
    message_id: str              = ""
    trace_id:   str              = ""
    latency_ms: float            = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok":         self.ok,
            "result":     self.result,
            "error":      self.error,
            "message_id": self.message_id,
            "trace_id":   self.trace_id,
            "latency_ms": self.latency_ms,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NodeResponse":
        return cls(
            ok         = d.get("ok", False),
            result     = d.get("result", {}),
            error      = d.get("error", ""),
            message_id = d.get("message_id", ""),
            trace_id   = d.get("trace_id",   ""),
            latency_ms = d.get("latency_ms", 0.0),
        )

    @classmethod
    def error_response(cls, msg: str, message_id: str = "",
                       trace_id: str = "") -> "NodeResponse":
        return cls(ok=False, error=msg, message_id=message_id,
                   trace_id=trace_id)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineStep:
    """
    A single step in a ``Pipeline``.

    Attributes
    ----------
    node_type:   "compute" | "agent" | "data_source"
    action:      Action to invoke (default inferred from node_type).
    node_id:     Pin to a specific node (None = router picks).
    model:       Model hint.
    source_type: Type hint for agent/data_source.
    transform:   Optional callable ``(prev_result: dict) → NodeMessage``
                 for custom input shaping between steps.
                 If None, a default transform is applied.
    extra:       Extra payload fields passed verbatim.
    label:       Human-readable step name for logging.
    """
    node_type:   str
    action:      Optional[str]                           = None
    node_id:     Optional[str]                           = None
    model:       Optional[str]                           = None
    source_type: Optional[str]                           = None
    transform:   Optional[Any]                           = None  # Callable
    extra:       Dict[str, Any]                          = field(default_factory=dict)
    label:       str                                     = ""

    def default_action(self) -> str:
        return {
            "compute":     "generate",
            "agent":       "run",
            "data_source": "query",
        }.get(self.node_type, "run")


@dataclass
class PipelineResult:
    """
    The outcome of running a Pipeline.

    Attributes
    ----------
    steps:       Ordered list of (step_label, NodeResponse) for each step.
    final:       The NodeResponse from the last step.
    trace_id:    Shared trace_id across all steps.
    latency_ms:  Total wall-clock time.
    ok:          True if every step succeeded.
    """
    steps:      List[tuple]     = field(default_factory=list)  # (label, NodeResponse)
    final:      Optional[NodeResponse] = None
    trace_id:   str             = ""
    latency_ms: float           = 0.0
    ok:         bool            = True

    def step_result(self, label: str) -> Optional[NodeResponse]:
        """Return the NodeResponse for a step with the given label."""
        for lbl, resp in self.steps:
            if lbl == label:
                return resp
        return None

    def text(self) -> str:
        """Convenience: return the text from the final step's result."""
        if self.final and self.final.ok:
            r = self.final.result
            return r.get("text", r.get("content", ""))
        return ""

    def chunks(self) -> List[Dict]:
        """Convenience: return chunks from the final step (data_source output)."""
        if self.final and self.final.ok:
            return self.final.result.get("chunks", [])
        return []
