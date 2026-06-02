"""
AgentRouter — load-aware request routing across registered agent nodes.

The router queries the Ravnest node registry to discover available agent
nodes and routes AgentRequests to the best one based on:

  1. Agent-type match  — prefer nodes with the exact requested agent_type.
  2. Model match       — prefer nodes serving the requested model (if set).
  3. Strategy          — load-based (least-loaded) by default; pluggable.
  4. Fallback          — on failure, try the next node; refresh registry.

Usage
-----
    from ravnest.agents.router import AgentRouter

    router = AgentRouter(registry_address="registry_host:50099")

    resp = router.run(AgentRequest(
        messages   = [Message("user", "Summarise the Q3 earnings report.")],
        extra      = {"agent_type": "research"},   # optional type hint
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
    AgentBackend, AgentCapability, AgentHealthStatus, AgentRequest,
    AgentResponse,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Routing strategies
# ─────────────────────────────────────────────────────────────────────────────

class AgentRoutingStrategy(ABC):
    @abstractmethod
    def pick(self, candidates: List) -> Optional[object]:
        ...


class LoadBasedAgentStrategy(AgentRoutingStrategy):
    """Route to the agent node with the lowest CPU / RAM load."""

    def pick(self, candidates: List) -> Optional[object]:
        if not candidates:
            return None

        def score(cap) -> float:
            load = cap.current_load or {}
            return load.get("ram_percent", load.get("cpu_percent", 0.0))

        return min(candidates, key=score)


class RoundRobinAgentStrategy(AgentRoutingStrategy):
    def __init__(self):
        self._idx  = 0
        self._lock = threading.Lock()

    def pick(self, candidates: List) -> Optional[object]:
        if not candidates:
            return None
        with self._lock:
            chosen    = candidates[self._idx % len(candidates)]
            self._idx += 1
        return chosen


class AgentTypeStrategy(AgentRoutingStrategy):
    """
    Prefer agent nodes of the given type; fall back to all nodes.
    Wraps an inner strategy for tie-breaking within the filtered set.
    """

    def __init__(self, agent_type: str,
                 inner: Optional[AgentRoutingStrategy] = None):
        self._type  = agent_type
        self._inner = inner or LoadBasedAgentStrategy()

    def pick(self, candidates: List) -> Optional[object]:
        exact = [c for c in candidates
                 if (c.subtype or c.metadata.get("agent_type")) == self._type]
        pool  = exact if exact else candidates
        return self._inner.pick(pool)


# ─────────────────────────────────────────────────────────────────────────────
# Backend factory
# ─────────────────────────────────────────────────────────────────────────────

def _backend_for_node(cap) -> Optional[AgentBackend]:
    """
    Instantiate the right AgentBackend given a registry NodeCapability.

    Supports: litellm, research, sql subtypes discovered from the registry.
    """
    subtype = cap.subtype or cap.metadata.get("agent_type", "")
    meta    = cap.metadata or {}
    model   = (cap.models or ["unknown"])[0]

    try:
        if subtype == "litellm":
            from .litellm_agent import LiteLLMAgent
            return LiteLLMAgent(
                model    = model,
                api_base = meta.get("api_base"),
                api_key  = meta.get("api_key"),
                node_id  = cap.node_id,
            )

        if subtype == "research":
            from .research_agent import ResearchAgent
            return ResearchAgent(
                model    = model,
                api_base = meta.get("api_base"),
                api_key  = meta.get("api_key"),
                node_id  = cap.node_id,
            )

        if subtype == "sql":
            db_url = meta.get("db_url")
            if not db_url:
                logger.warning("SQL agent node %s has no db_url in metadata", cap.node_id)
                return None
            from .sql_agent import SQLAgent
            return SQLAgent(
                model   = model,
                db_url  = db_url,
                node_id = cap.node_id,
            )

    except Exception as exc:
        logger.warning("Failed to create agent backend for node %s: %s", cap.node_id, exc)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────

class AgentRouter:
    """
    Routes AgentRequests to the best available AgentBackend.

    Backends are discovered from the Ravnest node registry (node_type=agent)
    and cached locally.  The cache is refreshed every ``refresh_interval``
    seconds (default 30 s) or immediately after a routing failure.

    Args:
        registry_address: host:port of the registry server.
        strategy:         Routing strategy (default: LoadBasedAgentStrategy).
        refresh_interval: Seconds between automatic backend rediscovery.
        max_retries:      How many agent nodes to try before giving up.
    """

    def __init__(
        self,
        registry_address: str,
        strategy:         Optional[AgentRoutingStrategy] = None,
        refresh_interval: float                          = 30.0,
        max_retries:      int                            = 3,
    ):
        from ravnest.registry import RegistryClient
        self._registry         = RegistryClient(registry_address,
                                                cache_ttl=refresh_interval)
        self._strategy         = strategy or LoadBasedAgentStrategy()
        self._refresh_interval = refresh_interval
        self._max_retries      = max_retries

        # node_id -> (AgentBackend, NodeCapability)
        self._backends: Dict[str, Tuple[AgentBackend, object]] = {}
        self._lock      = threading.RLock()
        self._last_refresh = 0.0

        # Local backends injected via add_local_backend()
        self._local_backends: List[Tuple[AgentBackend, object]] = []

        self._refresh_backends()
        self._start_refresh_thread()

    # ── public API ────────────────────────────────────────────────────────

    def run(self, request: AgentRequest) -> AgentResponse:
        """Route a request to the best available agent (synchronous)."""
        return self._route_with_retry(request)

    async def arun(self, request: AgentRequest) -> AgentResponse:
        """Async version — awaits the selected backend's arun."""
        candidates = self._rank_candidates(request)
        if not candidates:
            raise RuntimeError("No agent backends available in the registry")
        last_exc: Optional[Exception] = None
        for cap, backend in candidates[:self._max_retries]:
            try:
                return await backend.arun(request)
            except Exception as exc:
                logger.warning("Agent %s failed: %s — trying next", cap.node_id, exc)
                last_exc = exc
                self._refresh_backends(force=True)
        raise RuntimeError(f"All agent backends failed. Last error: {last_exc}")

    async def astream(self, request: AgentRequest):
        """Async streaming — delegates to the selected backend."""
        candidates = self._rank_candidates(request)
        if not candidates:
            raise RuntimeError("No agent backends available")
        for cap, backend in candidates[:self._max_retries]:
            try:
                async for token in backend.astream(request):
                    yield token
                return
            except Exception as exc:
                logger.warning("Agent %s stream failed: %s — trying next",
                               cap.node_id, exc)
                self._refresh_backends(force=True)
        raise RuntimeError("All agent backends failed during streaming")

    def add_local_backend(self, backend: AgentBackend, cap=None) -> None:
        """
        Inject a local agent backend (e.g. one that was started in-process).
        It participates in routing alongside registry-discovered nodes.
        """
        if cap is None:
            cap = backend._build_node_capability()
        with self._lock:
            self._local_backends.append((backend, cap))
        logger.info("Added local agent backend %s (%s)",
                    cap.node_id, cap.agent_type)

    def remove_local_backend(self, node_id: str) -> None:
        with self._lock:
            self._local_backends = [
                (b, c) for b, c in self._local_backends if c.node_id != node_id
            ]

    def list_backends(self) -> List[dict]:
        """Return a summary of all known agent backends."""
        with self._lock:
            registry = [
                {
                    "node_id":    cap.node_id,
                    "agent_type": cap.subtype,
                    "models":     cap.models,
                    "address":    cap.address,
                    "load":       cap.current_load,
                    "source":     "registry",
                }
                for _, cap in self._backends.values()
            ]
            local = [
                {
                    "node_id":    cap.node_id,
                    "agent_type": cap.agent_type,
                    "models":     cap.models,
                    "address":    cap.extra.get("address", "local"),
                    "load":       {},
                    "source":     "local",
                }
                for _, cap in self._local_backends
            ]
        return registry + local

    # ── internal ──────────────────────────────────────────────────────────

    def _route_with_retry(self, request: AgentRequest) -> AgentResponse:
        candidates = self._rank_candidates(request)
        if not candidates:
            raise RuntimeError("No agent backends available in the registry")

        last_exc: Optional[Exception] = None
        for cap, backend in candidates[:self._max_retries]:
            try:
                return backend.run(request)
            except Exception as exc:
                logger.warning("Agent %s failed: %s — retrying", cap.node_id, exc)
                last_exc = exc
                self._refresh_backends(force=True)

        raise RuntimeError(
            f"All {len(candidates)} agent backend(s) failed. Last error: {last_exc}"
        )

    def _rank_candidates(self, request: AgentRequest):
        """Return (NodeCapability-like, backend) pairs ordered by strategy."""
        with self._lock:
            # For registry caps, they are NodeCapability objects
            all_caps = [cap for _, cap in self._backends.values()]
            # For local backends, caps are AgentCapability objects — we need
            # to wrap them so they have the same interface as NodeCapability
            for backend, cap in self._local_backends:
                all_caps.append(_AgentCapWrapper(cap))

        # Optional agent_type filter from request.extra
        agent_type = request.extra.get("agent_type")
        if agent_type:
            filtered = [c for c in all_caps
                        if (getattr(c, "subtype", None) or
                            getattr(c, "agent_type", None)) == agent_type]
            if not filtered:
                filtered = all_caps
        else:
            filtered = all_caps

        if not filtered:
            return []

        # Rank via strategy
        ranked = []
        remaining = list(filtered)
        while remaining:
            best = self._strategy.pick(remaining)
            if best is None:
                break
            ranked.append(best)
            remaining.remove(best)

        # Attach backends
        result = []
        for cap in ranked:
            backend = self._get_backend_for_cap(cap)
            if backend is not None:
                result.append((cap, backend))
        return result

    def _get_backend_for_cap(self, cap) -> Optional[AgentBackend]:
        node_id = getattr(cap, "node_id", None)
        # Check local first
        for backend, local_cap in self._local_backends:
            if local_cap.node_id == node_id:
                return backend
        with self._lock:
            entry = self._backends.get(node_id)
        if entry:
            return entry[0]
        return None

    def _refresh_backends(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_refresh) < self._refresh_interval:
            return

        logger.debug("Refreshing agent backend list from registry")
        all_caps = []
        try:
            caps = self._registry.discover(node_type="agent", force=True)
            all_caps.extend(caps)
        except Exception as exc:
            logger.warning("Registry discover failed for agent nodes: %s", exc)

        new_backends: Dict[str, Tuple[AgentBackend, object]] = {}
        for cap in all_caps:
            with self._lock:
                existing = self._backends.get(cap.node_id)
            if existing:
                new_backends[cap.node_id] = (existing[0], cap)
            else:
                backend = _backend_for_node(cap)
                if backend:
                    new_backends[cap.node_id] = (backend, cap)
                    logger.info("Discovered agent backend: %s (%s)",
                                cap.node_id, cap.subtype)

        with self._lock:
            removed = set(self._backends) - set(new_backends)
            for nid in removed:
                logger.info("Agent backend removed from registry: %s", nid)
            self._backends    = new_backends
            self._last_refresh = now

    def _start_refresh_thread(self) -> None:
        def _loop():
            while True:
                time.sleep(self._refresh_interval)
                try:
                    self._refresh_backends(force=True)
                except Exception as exc:
                    logger.warning("Background agent refresh failed: %s", exc)

        t = threading.Thread(target=_loop, daemon=True, name="agent-router-refresh")
        t.start()


# ─────────────────────────────────────────────────────────────────────────────
# Thin wrapper so AgentCapability objects behave like NodeCapability in ranking
# ─────────────────────────────────────────────────────────────────────────────

class _AgentCapWrapper:
    """Wraps an AgentCapability (or any cap-like object) to expose the
    NodeCapability interface expected by routing strategies."""

    def __init__(self, cap):
        self._cap          = cap
        self.node_id       = cap.node_id
        # Support both AgentCapability (.agent_type) and NodeCapability-like
        # objects (.subtype) transparently.
        self.agent_type    = getattr(cap, "agent_type",
                                     getattr(cap, "subtype", "unknown"))
        self.subtype       = self.agent_type
        self.models        = getattr(cap, "models", [])
        # Preserve load info if present (e.g. from registry or test helpers)
        self.current_load  = getattr(cap, "current_load", {})
        self.address       = (cap.extra.get("address", "local")
                               if hasattr(cap, "extra") else
                               getattr(cap, "address", "local"))
        self.metadata      = getattr(cap, "metadata",
                                     {"agent_type": self.agent_type})
