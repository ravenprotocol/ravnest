"""
ravnest.compute — Compute backend abstraction layer.

Provides a unified async interface over multiple LLM serving runtimes so that
the rest of Ravnest (router, agents, orchestrator) can treat every inference
engine identically.

Quick-start
-----------
>>> from ravnest.compute import (
...     GenerateRequest, GenerateResponse,
...     Message, ComputeBackend,
...     OllamaBackend, OpenAICompatBackend,
...     ComputeRouter, LoadBasedStrategy,
... )

Backends available:
    - RavnestBackend      — wraps InferenceEngine (Ravnest pipeline)
    - VLLMBackend         — local vLLM AsyncLLMEngine
    - SGLangBackend       — SGLang HTTP server
    - OllamaBackend       — Ollama REST API
    - OpenAICompatBackend — any OpenAI-compatible endpoint (OpenAI, Together, Groq, …)

Router:
    - ComputeRouter       — discovers nodes from registry, routes with pluggable strategy
    - LoadBasedStrategy   — route to least-loaded GPU/CPU node
    - RoundRobinStrategy  — even distribution across all nodes
    - ModelMatchStrategy  — prefer exact model match, tie-break via inner strategy
"""

from .base import (
    ComputeBackend,
    ComputeCapability,
    EmbedRequest,
    EmbedResponse,
    GenerateRequest,
    GenerateResponse,
    HealthStatus,
    Message,
)

from .router import (
    ComputeRouter,
    LoadBasedStrategy,
    ModelMatchStrategy,
    RoutingStrategy,
    RoundRobinStrategy,
)

# Concrete backends — imported lazily in their own modules to avoid hard
# dependency on vllm / httpx / openai at import time.  Expose the classes here
# so callers can do `from ravnest.compute import OllamaBackend` without having
# to know which sub-module it lives in.
from .ollama_backend   import OllamaBackend
from .openai_compat    import OpenAICompatBackend
from .ravnest_backend  import RavnestBackend
from .sglang_backend   import SGLangBackend
from .vllm_backend     import VLLMBackend

__all__ = [
    # ── data classes ──────────────────────────────────────────────────────
    "Message",
    "GenerateRequest",
    "GenerateResponse",
    "EmbedRequest",
    "EmbedResponse",
    "HealthStatus",
    "ComputeCapability",
    # ── abstract base ─────────────────────────────────────────────────────
    "ComputeBackend",
    # ── concrete backends ─────────────────────────────────────────────────
    "RavnestBackend",
    "VLLMBackend",
    "SGLangBackend",
    "OllamaBackend",
    "OpenAICompatBackend",
    # ── router + strategies ───────────────────────────────────────────────
    "ComputeRouter",
    "RoutingStrategy",
    "LoadBasedStrategy",
    "RoundRobinStrategy",
    "ModelMatchStrategy",
]
