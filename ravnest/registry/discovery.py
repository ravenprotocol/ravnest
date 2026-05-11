"""
Registry client — used by every node type (compute, agent, data_source) to
register itself, send heartbeats, and discover other nodes.

Discover results are cached locally with a configurable TTL so that:
  - Repeated discover() calls don't hammer the registry on the hot path.
  - The cache keeps returning the last known list if the registry is briefly
    unreachable, instead of raising.
"""

import json
import logging
import time
import threading
from typing import Dict, List, Optional, Tuple

import grpc

from .capability import NodeCapability, NodeType, ResourceSpec

logger = logging.getLogger(__name__)

_SENTINEL = object()   # used as a cache-miss marker


def _discover_key(node_type: str, subtype: str, models: List[str]) -> tuple:
    return (node_type, subtype, tuple(sorted(models or [])))


class RegistryClient:
    """
    gRPC client for the NodeRegistryService.

    Args:
        registry_address: host:port of the registry server, e.g. "10.0.0.1:50099".
        timeout:          Per-RPC timeout in seconds (default 5 s).
        cache_ttl:        Seconds before a cached discover/get_node result expires.
                          Set to 0 to disable caching (default 30 s).
    """

    def __init__(
        self,
        registry_address: str,
        timeout:   float = 5.0,
        cache_ttl: float = 30.0,
    ):
        self._address   = registry_address
        self._timeout   = timeout
        self._cache_ttl = cache_ttl
        self._channel   = grpc.insecure_channel(registry_address)
        # Stubs imported lazily so the module loads before pb2 files exist.
        self._stub = None
        self._lock = threading.Lock()

        # discover cache: key -> (result_list, expiry_timestamp)
        self._discover_cache: Dict[tuple, Tuple[List[NodeCapability], float]] = {}
        # get_node cache:  node_id -> (NodeCapability | None, expiry_timestamp)
        self._node_cache: Dict[str, Tuple[Optional[NodeCapability], float]] = {}

    # ------------------------------------------------------------------ #
    # internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _get_stub(self):
        if self._stub is None:
            from ravnest.protos.registry_pb2_grpc import NodeRegistryServiceStub
            self._stub = NodeRegistryServiceStub(self._channel)
        return self._stub

    def _pb2(self):
        from ravnest.protos import registry_pb2
        return registry_pb2

    def _cache_expiry(self) -> float:
        return time.monotonic() + self._cache_ttl

    def _invalidate_discover_cache(self):
        """Wipe all discover results (called after any topology-changing write)."""
        with self._lock:
            self._discover_cache.clear()

    def _invalidate_node_cache(self, node_id: str):
        with self._lock:
            self._node_cache.pop(node_id, None)

    # ------------------------------------------------------------------ #
    # write ops                                                            #
    # ------------------------------------------------------------------ #

    def register(self, cap: NodeCapability) -> bool:
        """Announce this node to the registry."""
        pb2  = self._pb2()
        stub = self._get_stub()
        nt   = cap.node_type.value if hasattr(cap.node_type, "value") else str(cap.node_type)
        info = pb2.NodeInfo(
            node_id        = cap.node_id,
            node_type      = nt,
            subtype        = cap.subtype,
            address        = cap.address,
            ram_mb         = cap.resources.ram_mb,
            vram_mb        = cap.resources.vram_mb,
            bandwidth_mbps = cap.resources.bandwidth_mbps,
            disk_gb        = cap.resources.disk_gb,
            cpu_cores      = cap.resources.cpu_cores,
            models         = cap.models,
            metadata       = json.dumps(cap.metadata),
            registered_at  = int(cap.registered_at),
        )
        resp = stub.Register(pb2.RegisterRequest(node_info=info), timeout=self._timeout)
        logger.info("Registered %s with registry at %s: %s", cap.node_id, self._address, resp.message)
        if resp.success:
            # A new node changes discover results — drop the cache.
            self._invalidate_discover_cache()
            self._invalidate_node_cache(cap.node_id)
        return resp.success

    def heartbeat(self, node_id: str, load: dict) -> bool:
        """Report liveness and current load.  Does NOT invalidate the cache —
        load metrics are refreshed on the next cache miss naturally."""
        pb2  = self._pb2()
        stub = self._get_stub()
        load_msg = pb2.LoadInfo(
            cpu_percent      = load.get("cpu_percent", 0.0),
            ram_percent      = load.get("ram_percent", 0.0),
            gpu_percent      = load.get("gpu_percent", 0.0),
            gpu_vram_percent = load.get("gpu_vram_percent", 0.0),
        )
        resp = stub.Heartbeat(
            pb2.HeartbeatRequest(node_id=node_id, load=load_msg),
            timeout=self._timeout,
        )
        return resp.acknowledged

    def deregister(self, node_id: str) -> bool:
        """Gracefully remove this node from the registry."""
        pb2  = self._pb2()
        stub = self._get_stub()
        resp = stub.Deregister(pb2.DeregisterRequest(node_id=node_id), timeout=self._timeout)
        if resp.success:
            self._invalidate_discover_cache()
            self._invalidate_node_cache(node_id)
        return resp.success

    # ------------------------------------------------------------------ #
    # read ops (with TTL cache)                                            #
    # ------------------------------------------------------------------ #

    def discover(
        self,
        node_type: str = "",
        subtype:   str = "",
        models:    List[str] = None,
        force:     bool = False,
    ) -> List[NodeCapability]:
        """
        Query the registry for nodes matching the given filters.

        Results are cached locally for ``cache_ttl`` seconds.  Pass
        ``force=True`` to bypass the cache and always hit the registry.

        Args:
            node_type: e.g. "standalone_compute", "agent" — empty = any type.
            subtype:   e.g. "vllm", "research" — empty = any subtype.
            models:    Node must serve at least one of these models.
            force:     Skip the local cache and fetch fresh results.
        """
        key = _discover_key(node_type, subtype, models)

        if not force and self._cache_ttl > 0:
            with self._lock:
                cached = self._discover_cache.get(key)
            if cached is not None:
                result, expiry = cached
                if time.monotonic() < expiry:
                    logger.debug("discover cache hit for key=%s", key)
                    return result

        # Cache miss or forced refresh — go to registry.
        try:
            pb2  = self._pb2()
            stub = self._get_stub()
            req  = pb2.DiscoverRequest(
                node_type = node_type,
                subtype   = subtype,
                models    = models or [],
            )
            resp   = stub.Discover(req, timeout=self._timeout)
            result = [self._entry_to_cap(e) for e in resp.nodes]
        except Exception as exc:
            logger.warning("Registry discover() RPC failed: %s — returning stale cache if available", exc)
            with self._lock:
                stale = self._discover_cache.get(key)
            if stale is not None:
                return stale[0]
            return []

        if self._cache_ttl > 0:
            with self._lock:
                self._discover_cache[key] = (result, self._cache_expiry())

        return result

    def get_node(self, node_id: str, force: bool = False) -> Optional[NodeCapability]:
        """
        Fetch a single node by ID.

        Cached for ``cache_ttl`` seconds.  Pass ``force=True`` to skip cache.
        """
        if not force and self._cache_ttl > 0:
            with self._lock:
                cached = self._node_cache.get(node_id)
            if cached is not None:
                result, expiry = cached
                if time.monotonic() < expiry:
                    logger.debug("get_node cache hit for %s", node_id)
                    return result

        try:
            pb2  = self._pb2()
            stub = self._get_stub()
            resp = stub.GetNode(pb2.GetNodeRequest(node_id=node_id), timeout=self._timeout)
            result = self._entry_to_cap(resp.node) if resp.found else None
        except Exception as exc:
            logger.warning("Registry get_node() RPC failed for %s: %s — returning stale cache", node_id, exc)
            with self._lock:
                stale = self._node_cache.get(node_id)
            return stale[0] if stale is not None else None

        if self._cache_ttl > 0:
            with self._lock:
                self._node_cache[node_id] = (result, self._cache_expiry())

        return result

    def invalidate_cache(self):
        """Manually flush the entire local cache."""
        with self._lock:
            self._discover_cache.clear()
            self._node_cache.clear()
        logger.debug("RegistryClient cache invalidated")

    # ------------------------------------------------------------------ #
    # conversion                                                           #
    # ------------------------------------------------------------------ #

    def _entry_to_cap(self, entry) -> NodeCapability:
        info = entry.node_info
        return NodeCapability(
            node_id       = info.node_id,
            node_type     = NodeType(info.node_type),
            subtype       = info.subtype,
            address       = info.address,
            resources     = ResourceSpec(
                ram_mb         = info.ram_mb,
                vram_mb        = info.vram_mb,
                bandwidth_mbps = info.bandwidth_mbps,
                disk_gb        = info.disk_gb,
                cpu_cores      = info.cpu_cores,
            ),
            models        = list(info.models),
            metadata      = json.loads(info.metadata) if info.metadata else {},
            registered_at = float(info.registered_at),
            last_heartbeat= float(entry.last_heartbeat),
            current_load  = {
                "cpu_percent":      entry.current_load.cpu_percent,
                "ram_percent":      entry.current_load.ram_percent,
                "gpu_percent":      entry.current_load.gpu_percent,
                "gpu_vram_percent": entry.current_load.gpu_vram_percent,
            },
        )

    # ------------------------------------------------------------------ #
    # lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def close(self):
        self._channel.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
