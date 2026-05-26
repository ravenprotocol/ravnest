"""
ComputeRouter — load-aware request routing across all registered backends.

The router queries the Ravnest node registry to discover available compute
nodes and routes GenerateRequests to the best available one based on:

  1. Model match   — node must serve the requested model (if specified).
  2. Strategy      — load-based (least-loaded GPU) by default; pluggable.
  3. Fallback      — if the best node is unreachable, try the next in line.

Usage:
    from ravnest.registry import RegistryClient
    from ravnest.compute.router import ComputeRouter, LoadBasedStrategy

    router = ComputeRouter(
        registry_address = "registry_host:50099",
        strategy         = LoadBasedStrategy(),
    )

    # Route a request — picks the best available backend automatically.
    resp = router.generate(GenerateRequest(
        messages = [Message("user", "Explain transformers in one sentence.")],
        model    = "llama-3.1-8b",   # optional — omit to let router choose
        max_tokens = 128,
    ))
    print(resp.text)
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from .base import (
    ComputeBackend, ComputeCapability, EmbedRequest, EmbedResponse,
    GenerateRequest, GenerateResponse, HealthStatus,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Routing strategies
# ─────────────────────────────────────────────────────────────────────────────

class RoutingStrategy(ABC):
    """Pick the best node from a list of candidates."""

    @abstractmethod
    def pick(self, candidates: List) -> Optional[object]:
        """Return the best ``NodeCapability`` from *candidates*, or ``None``."""
        ...


class LoadBasedStrategy(RoutingStrategy):
    """
    Route to the node with the lowest combined GPU + VRAM utilisation.
    Falls back to RAM utilisation for CPU-only nodes.
    """

    def pick(self, candidates: List) -> Optional[object]:
        if not candidates:
            return None

        def score(cap) -> float:
            load = cap.current_load or {}
            gpu  = load.get("gpu_percent",      0.0)
            vram = load.get("gpu_vram_percent",  0.0)
            ram  = load.get("ram_percent",        0.0)
            # GPU nodes: GPU% * 0.6 + VRAM% * 0.4
            # CPU nodes: RAM%
            if vram > 0 or gpu > 0:
                return gpu * 0.6 + vram * 0.4
            return ram

        return min(candidates, key=score)


class RoundRobinStrategy(RoutingStrategy):
    """Distribute requests evenly across all candidates."""

    def __init__(self):
        self._idx  = 0
        self._lock = threading.Lock()

    def pick(self, candidates: List) -> Optional[object]:
        if not candidates:
            return None
        with self._lock:
            chosen   = candidates[self._idx % len(candidates)]
            self._idx += 1
        return chosen


class ModelMatchStrategy(RoutingStrategy):
    """
    Prefer nodes that serve the exact requested model, then fall back to
    load-based ordering within that filtered set.

    Wrap another strategy to handle final tie-breaking:
        ModelMatchStrategy(inner=LoadBasedStrategy())
    """

    def __init__(self, model: str, inner: Optional[RoutingStrategy] = None):
        self._model = model
        self._inner = inner or LoadBasedStrategy()

    def pick(self, candidates: List) -> Optional[object]:
        exact = [c for c in candidates if self._model in c.models]
        pool  = exact if exact else candidates
        return self._inner.pick(pool)


# ─────────────────────────────────────────────────────────────────────────────
# Backend factory
# ─────────────────────────────────────────────────────────────────────────────

def _backend_for_node(cap) -> Optional[ComputeBackend]:
    """
    Instantiate the right ComputeBackend given a NodeCapability.

    For STANDALONE_COMPUTE nodes whose address is an HTTP URL we create an
    HTTP-client backend.  We cannot instantiate RavnestBackend here (it
    requires a live Node process) so pipeline nodes are skipped.
    """
    subtype = cap.subtype
    meta    = cap.metadata or {}
    address = cap.address  # host:port *or* http://... for HTTP backends

    try:
        if subtype == "vllm":
            # vLLM exposes an OpenAI-compat API on its HTTP server
            base_url = meta.get("base_url") or f"http://{address}/v1"
            model    = (cap.models or ["unknown"])[0]
            from .openai_compat import OpenAICompatBackend
            return OpenAICompatBackend(model=model, base_url=base_url, api_key="EMPTY",
                                       node_id=cap.node_id)

        if subtype == "sglang":
            base_url = meta.get("base_url") or f"http://{address}"
            model    = (cap.models or ["unknown"])[0]
            from .sglang_backend import SGLangBackend
            return SGLangBackend(model=model, base_url=base_url, node_id=cap.node_id)

        if subtype == "ollama":
            base_url = meta.get("base_url") or f"http://{address}"
            model    = (cap.models or ["llama3"])[0]
            from .ollama_backend import OllamaBackend
            return OllamaBackend(model=model, base_url=base_url, node_id=cap.node_id)

        if subtype == "openai_compat":
            base_url = meta.get("base_url") or f"http://{address}/v1"
            model    = (cap.models or ["unknown"])[0]
            api_key  = meta.get("api_key", "EMPTY")
            from .openai_compat import OpenAICompatBackend
            return OpenAICompatBackend(model=model, base_url=base_url, api_key=api_key,
                                       node_id=cap.node_id)

        if subtype == "ravnest":
            # Cannot instantiate remotely — skip.  The local RavnestBackend
            # can be injected directly via ComputeRouter.add_local_backend().
            logger.debug("Skipping remote ravnest node %s (local-only)", cap.node_id)
            return None

    except Exception as exc:
        logger.warning("Failed to create backend for node %s: %s", cap.node_id, exc)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────

class ComputeRouter:
    """
    Routes GenerateRequests to the best available ComputeBackend.

    Backends are discovered from the Ravnest node registry and cached locally.
    The cache is refreshed every ``refresh_interval`` seconds (default 30 s)
    or immediately after a routing failure.

    Args:
        registry_address: host:port of the registry server.
        strategy:         Routing strategy (default: LoadBasedStrategy).
        node_types:       Registry node_types to query (default: standalone_compute).
        refresh_interval: Seconds between automatic backend rediscovery.
        max_retries:      How many backends to try before giving up.
    """

    def __init__(
        self,
        registry_address: str,
        strategy:         Optional[RoutingStrategy] = None,
        node_types:       List[str]                 = None,
        refresh_interval: float                     = 30.0,
        max_retries:      int                       = 3,
    ):
        from ..registry import RegistryClient
        self._registry          = RegistryClient(registry_address, cache_ttl=refresh_interval)
        self._strategy          = strategy or LoadBasedStrategy()
        self._node_types        = node_types or ["standalone_compute"]
        self._refresh_interval  = refresh_interval
        self._max_retries       = max_retries

        # node_id -> (ComputeBackend, NodeCapability)
        self._backends: Dict[str, Tuple[ComputeBackend, object]] = {}
        self._lock      = threading.RLock()
        self._last_refresh = 0.0

        # Optional local backends injected directly (e.g. RavnestBackend)
        self._local_backends: List[Tuple[ComputeBackend, object]] = []

        self._refresh_backends()
        self._start_refresh_thread()

    # ── public API ────────────────────────────────────────────────────────

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Route a generation request to the best available backend."""
        return self._route_with_retry(request, method="generate")

    async def agenerate(self, request: GenerateRequest) -> GenerateResponse:
        """Async version — awaits the winning backend's agenerate."""
        candidates = self._rank_candidates(request)
        last_exc: Optional[Exception] = None
        for cap, backend in candidates[:self._max_retries]:
            try:
                return await backend.agenerate(request)
            except Exception as exc:
                logger.warning("Backend %s failed: %s — trying next", cap.node_id, exc)
                last_exc = exc
                self._refresh_backends(force=True)
        raise RuntimeError(f"All backends failed. Last error: {last_exc}")

    async def agenerate_stream(self, request: GenerateRequest):
        """Async streaming — delegates to the winning backend."""
        candidates = self._rank_candidates(request)
        for cap, backend in candidates[:self._max_retries]:
            try:
                async for token in backend.agenerate_stream(request):
                    yield token
                return
            except Exception as exc:
                logger.warning("Backend %s stream failed: %s — trying next", cap.node_id, exc)
                self._refresh_backends(force=True)
        raise RuntimeError("All backends failed during streaming")

    def embed(self, request: EmbedRequest) -> EmbedResponse:
        return self._route_embed_with_retry(request)

    def add_local_backend(self, backend: ComputeBackend, cap=None) -> None:
        """
        Inject a local backend (e.g. RavnestBackend) that can't be discovered
        from the registry.  It participates in routing alongside registry nodes.
        """
        if cap is None:
            cap = backend._build_node_capability()
        with self._lock:
            self._local_backends.append((backend, cap))
        logger.info("Added local backend %s (%s)", cap.node_id, cap.subtype)

    def remove_local_backend(self, node_id: str) -> None:
        with self._lock:
            self._local_backends = [
                (b, c) for b, c in self._local_backends if c.node_id != node_id
            ]

    def list_backends(self) -> List[dict]:
        """Return a summary of all known backends and their current load."""
        with self._lock:
            registry = [
                {
                    "node_id":   cap.node_id,
                    "subtype":   cap.subtype,
                    "models":    cap.models,
                    "address":   cap.address,
                    "load":      cap.current_load,
                    "source":    "registry",
                }
                for _, cap in self._backends.values()
            ]
            local = [
                {
                    "node_id":  cap.node_id,
                    "subtype":  cap.subtype,
                    "models":   cap.models,
                    "address":  getattr(backend, "_address", "local"),
                    "load":     {},
                    "source":   "local",
                }
                for backend, cap in self._local_backends
            ]
        return registry + local

    # ── internal ──────────────────────────────────────────────────────────

    def _route_with_retry(self, request: GenerateRequest, method: str) -> GenerateResponse:
        candidates = self._rank_candidates(request)
        if not candidates:
            raise RuntimeError("No compute backends available in the registry")

        last_exc: Optional[Exception] = None
        for cap, backend in candidates[:self._max_retries]:
            try:
                return getattr(backend, method)(request)
            except Exception as exc:
                logger.warning(
                    "Backend %s (%s) failed (%s): %s — retrying",
                    cap.node_id, cap.subtype, method, exc
                )
                last_exc = exc
                self._refresh_backends(force=True)

        raise RuntimeError(
            f"All {len(candidates)} backend(s) failed. Last error: {last_exc}"
        )

    def _route_embed_with_retry(self, request: EmbedRequest) -> EmbedResponse:
        candidates = self._embedding_candidates()
        if not candidates:
            raise RuntimeError("No embedding-capable backends available")
        last_exc = None
        for cap, backend in candidates[:self._max_retries]:
            try:
                return backend.embed(request)
            except Exception as exc:
                last_exc = exc
        raise RuntimeError(f"All embedding backends failed. Last error: {last_exc}")

    def _rank_candidates(self, request: GenerateRequest) -> List[Tuple[object, ComputeBackend]]:
        """Return (NodeCapability, backend) pairs ordered by strategy score."""
        with self._lock:
            all_caps = [cap for _, cap in self._backends.values()]
            # include local backends
            for backend, cap in self._local_backends:
                all_caps.append(cap)

        # Filter by requested model if specified
        if request.model:
            filtered = [c for c in all_caps if request.model in (c.models or [])]
            if not filtered:
                filtered = all_caps   # fall back to all if no exact match
        else:
            filtered = all_caps

        if not filtered:
            return []

        # Apply strategy to pick ordering
        ranked_caps = []
        remaining   = list(filtered)
        while remaining:
            best = self._strategy.pick(remaining)
            if best is None:
                break
            ranked_caps.append(best)
            remaining.remove(best)

        # Attach backends to caps
        result = []
        for cap in ranked_caps:
            backend = self._get_backend_for_cap(cap)
            if backend is not None:
                result.append((cap, backend))
        return result

    def _embedding_candidates(self) -> List[Tuple[object, ComputeBackend]]:
        with self._lock:
            caps = [cap for _, cap in self._backends.values()
                    if cap.metadata.get("supports_embeddings")]
            local = [(b, c) for b, c in self._local_backends
                     if c.metadata.get("supports_embeddings")]
        result = [(cap, self._get_backend_for_cap(cap)) for cap in caps]
        result = [(c, b) for c, b in result if b is not None]
        result += local
        return result

    def _get_backend_for_cap(self, cap) -> Optional[ComputeBackend]:
        """Return cached backend or create a new one from cap."""
        # Check local backends first
        for backend, local_cap in self._local_backends:
            if local_cap.node_id == cap.node_id:
                return backend

        with self._lock:
            entry = self._backends.get(cap.node_id)
        if entry:
            return entry[0]
        # Cap came from local list but isn't in registry cache — create it
        backend = _backend_for_node(cap)
        if backend:
            with self._lock:
                self._backends[cap.node_id] = (backend, cap)
        return backend

    def _refresh_backends(self, force: bool = False) -> None:
        """Query the registry and rebuild the backend cache."""
        now = time.monotonic()
        if not force and (now - self._last_refresh) < self._refresh_interval:
            return

        logger.debug("Refreshing backend list from registry")
        all_caps = []
        for nt in self._node_types:
            try:
                caps = self._registry.discover(node_type=nt, force=True)
                all_caps.extend(caps)
            except Exception as exc:
                logger.warning("Registry discover failed for type=%s: %s", nt, exc)

        new_backends: Dict[str, Tuple[ComputeBackend, object]] = {}
        for cap in all_caps:
            # Reuse existing backend instance if the node is already known
            with self._lock:
                existing = self._backends.get(cap.node_id)
            if existing:
                new_backends[cap.node_id] = (existing[0], cap)  # update cap (fresh load)
            else:
                backend = _backend_for_node(cap)
                if backend:
                    new_backends[cap.node_id] = (backend, cap)
                    logger.info("Discovered new backend: %s (%s)", cap.node_id, cap.subtype)

        with self._lock:
            removed = set(self._backends) - set(new_backends)
            for nid in removed:
                logger.info("Backend removed from registry: %s", nid)
            self._backends    = new_backends
            self._last_refresh = now

    def _start_refresh_thread(self) -> None:
        def _loop():
            while True:
                time.sleep(self._refresh_interval)
                try:
                    self._refresh_backends(force=True)
                except Exception as exc:
                    logger.warning("Background backend refresh failed: %s", exc)

        t = threading.Thread(target=_loop, daemon=True, name="router-refresh")
        t.start()
