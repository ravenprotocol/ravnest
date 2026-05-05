"""
Registry client — used by every node type (compute, agent, data_source) to
register itself, send heartbeats, and discover other nodes.
"""

import json
import logging
from typing import List, Optional

import grpc

from .capability import NodeCapability, NodeType, ResourceSpec

logger = logging.getLogger(__name__)


class RegistryClient:
    """
    gRPC client for the NodeRegistryService.

    Args:
        registry_address: host:port of the registry server (e.g. "10.0.0.1:50099").
        timeout:          Per-RPC timeout in seconds.
    """

    def __init__(self, registry_address: str, timeout: float = 5.0):
        self._address = registry_address
        self._timeout = timeout
        self._channel = grpc.insecure_channel(registry_address)
        # Stub is imported lazily so this module can be imported before pb2 exists.
        self._stub    = None

    def _get_stub(self):
        if self._stub is None:
            from ravnest.protos.registry_pb2_grpc import NodeRegistryServiceStub
            self._stub = NodeRegistryServiceStub(self._channel)
        return self._stub

    def _pb2(self):
        from ravnest.protos import registry_pb2
        return registry_pb2

    # -- write ops ----------------------------------------------------------

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
        return resp.success

    def heartbeat(self, node_id: str, load: dict) -> bool:
        """Report liveness and current load."""
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
        return resp.success

    # -- read ops -----------------------------------------------------------

    def discover(
        self,
        node_type: str = "",
        subtype:   str = "",
        models:    List[str] = None,
    ) -> List[NodeCapability]:
        """
        Query the registry for nodes matching the given filters.

        Args:
            node_type: e.g. "standalone_compute", "agent" — empty matches all.
            subtype:   e.g. "vllm", "research" — empty matches all.
            models:    List of model names — node must serve at least one.
        """
        pb2  = self._pb2()
        stub = self._get_stub()
        req  = pb2.DiscoverRequest(
            node_type = node_type,
            subtype   = subtype,
            models    = models or [],
        )
        resp = stub.Discover(req, timeout=self._timeout)
        return [self._entry_to_cap(e) for e in resp.nodes]

    def get_node(self, node_id: str) -> Optional[NodeCapability]:
        """Fetch a single node by ID."""
        pb2  = self._pb2()
        stub = self._get_stub()
        resp = stub.GetNode(pb2.GetNodeRequest(node_id=node_id), timeout=self._timeout)
        if resp.found:
            return self._entry_to_cap(resp.node)
        return None

    # -- conversion ---------------------------------------------------------

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

    def close(self):
        self._channel.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
