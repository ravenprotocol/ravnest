"""
NodeClient — HTTP client for calling remote mesh nodes.

A ``NodeClient`` sends ``NodeMessage`` objects to a remote ``NodeServer``
and returns ``NodeResponse`` objects.  Nodes can be addressed directly by
URL or discovered via the Ravnest registry.

Install:  pip install httpx  (or aiohttp)

Usage
-----
    from ravnest.mesh.node_client import NodeClient
    from ravnest.mesh.base import NodeMessage

    # Direct address
    client = NodeClient("http://10.0.0.5:8765")

    # Send a generate request to a remote compute node
    resp = client.send(NodeMessage.generate(
        prompt     = "What is Ravnest?",
        max_tokens = 128,
    ))
    print(resp.result["text"])

    # Async
    resp = await client.asend(NodeMessage.data_query(
        query       = "distributed training",
        source_type = "text",
        top_k       = 5,
    ))
    for chunk in resp.result["chunks"]:
        print(chunk["score"], chunk["content"][:60])

    # Registry-backed: pick the best node automatically
    client = NodeClient.from_registry(
        registry_address = "localhost:50099",
        node_type        = "agent",
    )
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from .base import NodeMessage, NodeResponse

logger = logging.getLogger(__name__)

_HTTPX_ERR = "httpx is required for NodeClient. Run: pip install httpx"


class NodeClient:
    """
    HTTP client for sending NodeMessages to a remote NodeServer.

    Args:
        base_url: Base URL of the remote NodeServer, e.g. "http://host:8765".
        timeout:  Per-request HTTP timeout in seconds.
        retries:  Number of retry attempts on network failure.
    """

    def __init__(
        self,
        base_url: str,
        timeout:  float = 30.0,
        retries:  int   = 2,
    ):
        try:
            import httpx  # noqa: F401
        except ImportError:
            raise ImportError(_HTTPX_ERR)

        self._base_url = base_url.rstrip("/")
        self._timeout  = timeout
        self._retries  = retries

    # ── sync wrapper ──────────────────────────────────────────────────────

    def send(self, message: NodeMessage) -> NodeResponse:
        """Send a NodeMessage synchronously and return a NodeResponse."""
        return _run(self.asend(message))

    def health(self) -> Dict[str, Any]:
        """Fetch health status from the remote server."""
        return _run(self.ahealth())

    def capabilities(self) -> Dict[str, Any]:
        """Fetch capabilities from the remote server."""
        return _run(self.acapabilities())

    # ── async interface ───────────────────────────────────────────────────

    async def asend(self, message: NodeMessage) -> NodeResponse:
        """Async: send a NodeMessage and return a NodeResponse."""
        import httpx

        payload = message.to_dict()
        last_exc: Optional[Exception] = None

        for attempt in range(self._retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    t0   = time.perf_counter()
                    resp = await client.post(
                        f"{self._base_url}/message",
                        json=payload,
                        headers={"Content-Type": "application/json"},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    nr   = NodeResponse.from_dict(data)
                    if not nr.latency_ms:
                        nr.latency_ms = (time.perf_counter() - t0) * 1000
                    return nr
            except Exception as exc:
                last_exc = exc
                if attempt < self._retries:
                    await asyncio.sleep(0.5 * (attempt + 1))

        return NodeResponse.error_response(
            f"Request failed after {self._retries + 1} attempts: {last_exc}",
            message_id=message.message_id,
            trace_id=message.trace_id,
        )

    async def ahealth(self) -> Dict[str, Any]:
        """Async: GET /health from the remote server."""
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{self._base_url}/health")
            resp.raise_for_status()
            return resp.json()

    async def acapabilities(self) -> Dict[str, Any]:
        """Async: GET /capabilities from the remote server."""
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{self._base_url}/capabilities")
            resp.raise_for_status()
            return resp.json()

    async def astream(self, message: NodeMessage) -> AsyncIterator[str]:
        """
        Async streaming — sends the message and yields SSE/NDJSON tokens.

        For backends that support streaming, the server should emit
        newline-delimited JSON; each line is a token dict with a "token" key.
        Falls back to a single-response yield for non-streaming backends.
        """
        import httpx

        # Most NodeServer backends return full responses; we do a single call
        # and yield the text as one chunk.  Real streaming would require the
        # server to emit SSE — this is the v1 implementation.
        resp = await self.asend(message)
        if resp.ok:
            text = resp.result.get("text", "")
            if text:
                yield text
            for chunk in resp.result.get("chunks", []):
                yield chunk.get("content", "")
        else:
            raise RuntimeError(f"Remote node error: {resp.error}")

    # ── class methods ─────────────────────────────────────────────────────

    @classmethod
    def from_registry(
        cls,
        registry_address: str,
        node_type:        str,
        node_id:          Optional[str]  = None,
        subtype:          Optional[str]  = None,
        timeout:          float          = 30.0,
    ) -> "NodeClient":
        """
        Discover a node from the registry and return a client for it.

        Args:
            registry_address: Registry host:port.
            node_type:        "standalone_compute" | "agent" | "data_source".
            node_id:          Pin to a specific node_id (None = pick first).
            subtype:          Filter by subtype (e.g. "research", "text").
        """
        from ravnest.registry import RegistryClient

        rc   = RegistryClient(registry_address)
        caps = rc.discover(node_type=node_type)
        if not caps:
            raise RuntimeError(
                f"No nodes of type '{node_type}' found in registry"
            )

        # Filter by node_id / subtype
        if node_id:
            caps = [c for c in caps if c.node_id == node_id]
        if subtype:
            caps = [c for c in caps if c.subtype == subtype]
        if not caps:
            raise RuntimeError(
                f"No matching node found (type={node_type}, "
                f"node_id={node_id}, subtype={subtype})"
            )

        cap        = caps[0]
        # Prefer the mesh_address (http://...) from metadata; else derive from address
        mesh_addr  = (cap.metadata or {}).get("mesh_address")
        if not mesh_addr:
            address   = cap.address or "localhost:8765"
            mesh_addr = f"http://{address}"

        logger.info("NodeClient: using node %s at %s", cap.node_id, mesh_addr)
        return cls(base_url=mesh_addr, timeout=timeout)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _run(coro):
    """Run a coroutine from sync context, handling already-running loops."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)
