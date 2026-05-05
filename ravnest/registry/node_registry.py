"""
Centralized node registry — in-memory store + gRPC service layer.

Standalone usage:
    python -m ravnest.registry.node_registry          # listens on 0.0.0.0:50099

Embedded usage:
    from ravnest.registry.node_registry import NodeRegistry
    registry = NodeRegistry()
"""

import json
import logging
import threading
import time
from concurrent import futures
from typing import Dict, List, Optional

import grpc

from .capability import NodeCapability, NodeType, ResourceSpec

logger = logging.getLogger(__name__)

HEARTBEAT_TTL_SECONDS = 30   # evict a node after this many seconds without a heartbeat
REAPER_INTERVAL       = 10   # seconds between reaper sweeps
REGISTRY_DEFAULT_PORT = 50099


# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------

class NodeRegistry:
    """Thread-safe, in-memory store of registered NodeCapability objects."""

    def __init__(self, ttl_seconds: float = HEARTBEAT_TTL_SECONDS):
        self._nodes: Dict[str, NodeCapability] = {}
        self._lock  = threading.RLock()
        self._ttl   = ttl_seconds
        self._start_reaper()

    # -- write ops ----------------------------------------------------------

    def register(self, cap: NodeCapability) -> bool:
        with self._lock:
            cap.registered_at  = time.time()
            cap.last_heartbeat = time.time()
            self._nodes[cap.node_id] = cap
        logger.info("Registered %s (%s/%s) @ %s", cap.node_id, cap.node_type, cap.subtype, cap.address)
        return True

    def heartbeat(self, node_id: str, load: dict) -> bool:
        with self._lock:
            cap = self._nodes.get(node_id)
            if cap is None:
                return False
            cap.last_heartbeat = time.time()
            cap.current_load   = load
        return True

    def deregister(self, node_id: str) -> bool:
        with self._lock:
            if node_id in self._nodes:
                del self._nodes[node_id]
                logger.info("Deregistered %s", node_id)
                return True
        return False

    # -- read ops -----------------------------------------------------------

    def discover(
        self,
        node_type: str = "",
        subtype:   str = "",
        models:    List[str] = None,
    ) -> List[NodeCapability]:
        with self._lock:
            results = list(self._nodes.values())
        if node_type:
            results = [n for n in results if n.node_type == node_type]
        if subtype:
            results = [n for n in results if n.subtype == subtype]
        if models:
            results = [n for n in results if any(m in n.models for m in models)]
        return results

    def get_node(self, node_id: str) -> Optional[NodeCapability]:
        with self._lock:
            return self._nodes.get(node_id)

    def all_nodes(self) -> List[NodeCapability]:
        with self._lock:
            return list(self._nodes.values())

    # -- TTL reaper ---------------------------------------------------------

    def _start_reaper(self):
        t = threading.Thread(target=self._reap_loop, daemon=True, name="registry-reaper")
        t.start()

    def _reap_loop(self):
        while True:
            time.sleep(REAPER_INTERVAL)
            cutoff = time.time() - self._ttl
            with self._lock:
                expired = [nid for nid, cap in self._nodes.items()
                           if cap.last_heartbeat < cutoff]
            for nid in expired:
                with self._lock:
                    self._nodes.pop(nid, None)
                logger.warning("Evicted %s (no heartbeat for >%ss)", nid, self._ttl)


# ---------------------------------------------------------------------------
# Proto helpers
# ---------------------------------------------------------------------------

def _cap_to_node_info(cap: NodeCapability, pb2):
    nt = cap.node_type.value if hasattr(cap.node_type, "value") else str(cap.node_type)
    return pb2.NodeInfo(
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


def _node_info_to_cap(info) -> NodeCapability:
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
    )


def _make_entry(cap: NodeCapability, pb2):
    load = pb2.LoadInfo(
        cpu_percent      = cap.current_load.get("cpu_percent", 0.0),
        ram_percent      = cap.current_load.get("ram_percent", 0.0),
        gpu_percent      = cap.current_load.get("gpu_percent", 0.0),
        gpu_vram_percent = cap.current_load.get("gpu_vram_percent", 0.0),
    )
    return pb2.NodeEntry(
        node_info      = _cap_to_node_info(cap, pb2),
        current_load   = load,
        last_heartbeat = int(cap.last_heartbeat),
    )


# ---------------------------------------------------------------------------
# gRPC servicer
# ---------------------------------------------------------------------------

class RegistryServicer:
    """
    Implements NodeRegistryService RPCs.
    Inherits from the generated base class at runtime so this file doesn't
    depend on pb2 being present at import time (pb2 is compiled during install).
    """

    def __init__(self, registry: NodeRegistry):
        self._registry = registry

    def _pb2(self):
        from ravnest.protos import registry_pb2
        return registry_pb2

    def Register(self, request, context):
        pb2 = self._pb2()
        cap = _node_info_to_cap(request.node_info)
        ok  = self._registry.register(cap)
        return pb2.RegisterResponse(success=ok, message="ok" if ok else "failed")

    def Heartbeat(self, request, context):
        pb2  = self._pb2()
        load = {
            "cpu_percent":      request.load.cpu_percent,
            "ram_percent":      request.load.ram_percent,
            "gpu_percent":      request.load.gpu_percent,
            "gpu_vram_percent": request.load.gpu_vram_percent,
        }
        ok = self._registry.heartbeat(request.node_id, load)
        return pb2.HeartbeatResponse(acknowledged=ok)

    def Deregister(self, request, context):
        pb2 = self._pb2()
        ok  = self._registry.deregister(request.node_id)
        return pb2.DeregisterResponse(success=ok)

    def Discover(self, request, context):
        pb2   = self._pb2()
        nodes = self._registry.discover(
            node_type = request.node_type,
            subtype   = request.subtype,
            models    = list(request.models) if request.models else None,
        )
        return pb2.DiscoverResponse(nodes=[_make_entry(n, pb2) for n in nodes])

    def GetNode(self, request, context):
        pb2 = self._pb2()
        cap = self._registry.get_node(request.node_id)
        if cap is None:
            return pb2.GetNodeResponse(found=False)
        return pb2.GetNodeResponse(node=_make_entry(cap, pb2), found=True)


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

def build_grpc_server(registry: NodeRegistry, max_workers: int = 10) -> grpc.Server:
    from ravnest.protos import registry_pb2_grpc
    server   = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    servicer = RegistryServicer(registry)
    registry_pb2_grpc.add_NodeRegistryServiceServicer_to_server(servicer, server)
    return server


def serve(
    address:     str   = f"0.0.0.0:{REGISTRY_DEFAULT_PORT}",
    ttl_seconds: float = HEARTBEAT_TTL_SECONDS,
    max_workers: int   = 10,
    block:       bool  = True,
) -> grpc.Server:
    """
    Start the registry gRPC server.

    Args:
        address:     Listen address, e.g. "0.0.0.0:50099".
        ttl_seconds: Evict a node after this many seconds without a heartbeat.
        max_workers: gRPC thread-pool size.
        block:       Block until terminated (True) or return immediately (False).
    """
    registry = NodeRegistry(ttl_seconds=ttl_seconds)
    server   = build_grpc_server(registry, max_workers=max_workers)
    server.add_insecure_port(address)
    server.start()
    print(f"[Registry] Listening on {address}")
    logger.info("Registry server listening on %s", address)
    if block:
        server.wait_for_termination()
    return server


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ravnest node registry server")
    parser.add_argument("--address", default=f"0.0.0.0:{REGISTRY_DEFAULT_PORT}",
                        help="gRPC listen address (default: 0.0.0.0:50099)")
    parser.add_argument("--ttl", type=float, default=HEARTBEAT_TTL_SECONDS,
                        help="Heartbeat TTL in seconds before evicting a node (default: 30)")
    args = parser.parse_args()
    serve(address=args.address, ttl_seconds=args.ttl)
