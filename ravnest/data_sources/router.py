"""
DataRouter — load-aware routing across registered data source nodes.

Discovers ``data_source`` nodes from the Ravnest registry and routes
DataRequests to the best available backend based on:

  1. Source-type match — prefer nodes that serve the requested modality / type.
  2. Strategy         — load-based (least-loaded) by default; pluggable.
  3. Fallback         — on failure, try the next node and refresh the registry.

Usage
-----
    from ravnest.data_sources.router import DataRouter
    from ravnest.data_sources.base import DataRequest

    router = DataRouter(registry_address="registry_host:50099")

    resp = router.query(DataRequest(
        query    = "distributed inference",
        modality = "text",
        top_k    = 5,
        extra    = {"source_type": "text"},   # optional type hint
    ))
    for chunk in resp.chunks:
        print(chunk.score, chunk.content[:60])
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from .base import (
    DataChunk, DataRequest, DataResponse, DataSourceBackend,
    DataSourceCapability, DataSourceHealthStatus,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Routing strategies
# ─────────────────────────────────────────────────────────────────────────────

class DataRoutingStrategy(ABC):
    @abstractmethod
    def pick(self, candidates: List) -> Optional[object]:
        ...


class LoadBasedDataStrategy(DataRoutingStrategy):
    """Route to the data source node with the lowest RAM load."""

    def pick(self, candidates: List) -> Optional[object]:
        if not candidates:
            return None
        def score(cap) -> float:
            load = getattr(cap, "current_load", {}) or {}
            return load.get("ram_percent", load.get("cpu_percent", 0.0))
        return min(candidates, key=score)


class RoundRobinDataStrategy(DataRoutingStrategy):
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


class SourceTypeStrategy(DataRoutingStrategy):
    """
    Prefer data source nodes of a given type (text, image, vector_db, graph_db).
    Falls back to all nodes if no exact match.  Inner strategy breaks ties.
    """

    def __init__(self, source_type: str,
                 inner: Optional[DataRoutingStrategy] = None):
        self._type  = source_type
        self._inner = inner or LoadBasedDataStrategy()

    def pick(self, candidates: List) -> Optional[object]:
        exact = [c for c in candidates
                 if (getattr(c, "subtype", None) or
                     getattr(c, "source_type", None)) == self._type]
        pool  = exact if exact else candidates
        return self._inner.pick(pool)


# ─────────────────────────────────────────────────────────────────────────────
# Backend factory
# ─────────────────────────────────────────────────────────────────────────────

def _backend_for_node(cap) -> Optional[DataSourceBackend]:
    """Instantiate the right DataSourceBackend for a registry NodeCapability."""
    subtype = cap.subtype or cap.metadata.get("source_type", "")
    meta    = cap.metadata or {}

    try:
        if subtype == "text":
            paths = meta.get("paths", ["."])
            from .text_source import TextSource
            return TextSource(paths=paths, node_id=cap.node_id)

        if subtype == "image":
            paths = meta.get("paths", [])
            from .image_source import ImageSource
            return ImageSource(paths=paths, node_id=cap.node_id)

        if subtype == "vector_db":
            collection = meta.get("collection", "default")
            backend    = meta.get("backend", "auto")
            from .vector_db import VectorDBSource
            return VectorDBSource(backend=backend, collection=collection,
                                  node_id=cap.node_id)

        if subtype == "graph_db":
            db_backend = meta.get("backend", "auto")
            from .graph_db import GraphDBSource
            return GraphDBSource(backend=db_backend, node_id=cap.node_id)

    except Exception as exc:
        logger.warning("Failed to create data backend for node %s: %s",
                       cap.node_id, exc)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────

class DataRouter:
    """
    Routes DataRequests to the best available DataSourceBackend.

    Backends are discovered from the Ravnest node registry (node_type=data_source)
    and cached locally.  The cache is refreshed every ``refresh_interval``
    seconds (default 30 s) or immediately after a routing failure.

    Args:
        registry_address: host:port of the registry server.
        strategy:         Routing strategy (default: LoadBasedDataStrategy).
        refresh_interval: Seconds between automatic backend rediscovery.
        max_retries:      How many data source nodes to try before giving up.
    """

    def __init__(
        self,
        registry_address: str,
        strategy:         Optional[DataRoutingStrategy] = None,
        refresh_interval: float                         = 30.0,
        max_retries:      int                           = 3,
    ):
        from ravnest.registry import RegistryClient
        self._registry         = RegistryClient(registry_address,
                                                cache_ttl=refresh_interval)
        self._strategy         = strategy or LoadBasedDataStrategy()
        self._refresh_interval = refresh_interval
        self._max_retries      = max_retries

        # node_id -> (DataSourceBackend, NodeCapability)
        self._backends: Dict[str, Tuple[DataSourceBackend, object]] = {}
        self._lock      = threading.RLock()
        self._last_refresh = 0.0

        # Local backends injected manually
        self._local_backends: List[Tuple[DataSourceBackend, object]] = []

        self._refresh_backends()
        self._start_refresh_thread()

    # ── public API ────────────────────────────────────────────────────────

    def query(self, request: DataRequest) -> DataResponse:
        """Route a query to the best available data source (synchronous)."""
        return self._route_with_retry(request)

    async def aquery(self, request: DataRequest) -> DataResponse:
        """Async version — awaits the selected backend's aquery."""
        candidates = self._rank_candidates(request)
        if not candidates:
            raise RuntimeError("No data source backends available")
        last_exc: Optional[Exception] = None
        for cap, backend in candidates[:self._max_retries]:
            try:
                return await backend.aquery(request)
            except Exception as exc:
                logger.warning("Data source %s failed: %s — trying next",
                               cap.node_id, exc)
                last_exc = exc
                self._refresh_backends(force=True)
        raise RuntimeError(f"All data sources failed. Last: {last_exc}")

    async def astream(self, request: DataRequest):
        """Async streaming — delegates to the selected backend."""
        candidates = self._rank_candidates(request)
        if not candidates:
            raise RuntimeError("No data source backends available")
        for cap, backend in candidates[:self._max_retries]:
            try:
                async for chunk in backend.astream(request):
                    yield chunk
                return
            except Exception as exc:
                logger.warning("Data source %s stream failed: %s", cap.node_id, exc)
                self._refresh_backends(force=True)
        raise RuntimeError("All data source backends failed during streaming")

    def add_local_backend(self, backend: DataSourceBackend, cap=None) -> None:
        """
        Inject a local data source backend (e.g. an in-process TextSource).
        It participates in routing alongside registry-discovered nodes.
        """
        if cap is None:
            cap = backend.capabilities()
        with self._lock:
            self._local_backends.append((backend, cap))
        logger.info("Added local data source %s (%s)",
                    cap.node_id, cap.source_type)

    def remove_local_backend(self, node_id: str) -> None:
        with self._lock:
            self._local_backends = [
                (b, c) for b, c in self._local_backends if c.node_id != node_id
            ]

    def list_backends(self) -> List[dict]:
        """Return a summary of all known data source backends."""
        with self._lock:
            registry = [
                {
                    "node_id":     cap.node_id,
                    "source_type": cap.subtype,
                    "address":     cap.address,
                    "load":        cap.current_load,
                    "source":      "registry",
                }
                for _, cap in self._backends.values()
            ]
            local = [
                {
                    "node_id":     cap.node_id,
                    "source_type": cap.source_type,
                    "address":     cap.extra.get("address", "local"),
                    "load":        {},
                    "source":      "local",
                }
                for _, cap in self._local_backends
            ]
        return registry + local

    # ── internal ──────────────────────────────────────────────────────────

    def _route_with_retry(self, request: DataRequest) -> DataResponse:
        candidates = self._rank_candidates(request)
        if not candidates:
            raise RuntimeError("No data source backends available")
        last_exc = None
        for cap, backend in candidates[:self._max_retries]:
            try:
                return backend.query(request)
            except Exception as exc:
                logger.warning("Data source %s failed: %s — retrying",
                               cap.node_id, exc)
                last_exc = exc
                self._refresh_backends(force=True)
        raise RuntimeError(f"All data sources failed. Last: {last_exc}")

    def _rank_candidates(self, request: DataRequest):
        with self._lock:
            all_caps = [cap for _, cap in self._backends.values()]
            for backend, cap in self._local_backends:
                all_caps.append(_DataCapWrapper(cap))

        # Optional source_type filter
        source_type = request.extra.get("source_type")
        if source_type:
            filtered = [c for c in all_caps
                        if (getattr(c, "subtype", None) or
                            getattr(c, "source_type", None)) == source_type]
            if not filtered:
                filtered = all_caps
        else:
            filtered = all_caps

        if not filtered:
            return []

        ranked = []
        remaining = list(filtered)
        while remaining:
            best = self._strategy.pick(remaining)
            if best is None:
                break
            ranked.append(best)
            remaining.remove(best)

        result = []
        for cap in ranked:
            backend = self._get_backend_for_cap(cap)
            if backend is not None:
                result.append((cap, backend))
        return result

    def _get_backend_for_cap(self, cap) -> Optional[DataSourceBackend]:
        node_id = getattr(cap, "node_id", None)
        for backend, local_cap in self._local_backends:
            lc_id = getattr(local_cap, "node_id",
                            getattr(local_cap, "_node_id", None))
            if lc_id == node_id:
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
        logger.debug("Refreshing data source backend list from registry")
        all_caps = []
        try:
            caps = self._registry.discover(node_type="data_source", force=True)
            all_caps.extend(caps)
        except Exception as exc:
            logger.warning("Registry discover failed for data_source: %s", exc)

        new_backends: Dict[str, Tuple] = {}
        for cap in all_caps:
            with self._lock:
                existing = self._backends.get(cap.node_id)
            if existing:
                new_backends[cap.node_id] = (existing[0], cap)
            else:
                backend = _backend_for_node(cap)
                if backend:
                    new_backends[cap.node_id] = (backend, cap)
                    logger.info("Discovered data source: %s (%s)",
                                cap.node_id, cap.subtype)

        with self._lock:
            removed = set(self._backends) - set(new_backends)
            for nid in removed:
                logger.info("Data source removed from registry: %s", nid)
            self._backends    = new_backends
            self._last_refresh = now

    def _start_refresh_thread(self) -> None:
        def _loop():
            while True:
                time.sleep(self._refresh_interval)
                try:
                    self._refresh_backends(force=True)
                except Exception as exc:
                    logger.warning("Background data source refresh failed: %s", exc)
        t = threading.Thread(target=_loop, daemon=True,
                             name="data-router-refresh")
        t.start()


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper so DataSourceCapability behaves like NodeCapability in ranking
# ─────────────────────────────────────────────────────────────────────────────

class _DataCapWrapper:
    def __init__(self, cap):
        self._cap        = cap
        self.node_id     = cap.node_id
        self.source_type = getattr(cap, "source_type",
                                   getattr(cap, "subtype", "unknown"))
        self.subtype     = self.source_type
        self.current_load = getattr(cap, "current_load", {})
        self.address     = (cap.extra.get("address", "local")
                            if hasattr(cap, "extra") else
                            getattr(cap, "address", "local"))
        self.metadata    = getattr(cap, "metadata",
                                   {"source_type": self.source_type})
