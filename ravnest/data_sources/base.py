"""
ravnest.data_sources.base — Abstract DataSourceBackend + shared data types.

Every data source in Ravnest (text files, images, vector DBs, graph DBs)
implements this interface.  The router and agents only ever talk to
``DataSourceBackend`` objects, so swapping backends requires zero changes
upstream.

Data flow
---------
caller → DataRequest → DataSourceBackend.aquery() → DataResponse
                     ↘ DataSourceBackend.astream() → AsyncIterator[DataChunk]

Modalities
----------
- "text"   — plain text passages, paragraphs, or documents
- "image"  — base64-encoded images or file paths
- "vector" — raw embedding vectors
- "graph"  — structured graph triples / paths
- "any"    — source handles multiple modalities
"""

from __future__ import annotations

import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Shared data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataChunk:
    """
    A single unit of data returned by a DataSourceBackend.

    Attributes
    ----------
    content:    The actual data — text string, base64 image, etc.
    chunk_id:   Unique identifier within the response.
    modality:   "text" | "image" | "vector" | "graph"
    score:      Relevance / similarity score (0-1, higher = more relevant).
                0.0 means unscored (e.g. sequential reads).
    source:     Origin reference — file path, URL, node name, etc.
    metadata:   Arbitrary key-value pairs from the backend (page, offset, …).
    vector:     Raw embedding (when modality == "vector").
    """
    content:  str
    chunk_id: str                = field(default_factory=lambda: str(uuid.uuid4()))
    modality: str                = "text"
    score:    float              = 0.0
    source:   str                = ""
    metadata: Dict[str, Any]     = field(default_factory=dict)
    vector:   Optional[List[float]] = None


@dataclass
class DataRequest:
    """
    Input to a DataSourceBackend.

    Attributes
    ----------
    query:       Free-text query or search string.
    vector:      Pre-computed query embedding (used instead of query if set).
    modality:    Expected output modality — "text", "image", "vector", "graph", "any".
    top_k:       Maximum number of results to return.
    filters:     Backend-specific filter dict (metadata, tags, labels, …).
    include_vectors: Include raw embedding vectors in DataChunk.vector.
    stream:      Whether the caller wants a streaming response.
    request_id:  Caller-supplied key; auto-generated if None.
    extra:       Backend-specific overrides.
    """
    query:           str                    = ""
    vector:          Optional[List[float]]  = None
    modality:        str                    = "text"
    top_k:           int                    = 5
    filters:         Dict[str, Any]         = field(default_factory=dict)
    include_vectors: bool                   = False
    stream:          bool                   = False
    request_id:      str                    = field(
        default_factory=lambda: str(uuid.uuid4())
    )
    extra:           Dict[str, Any]         = field(default_factory=dict)


@dataclass
class DataResponse:
    """
    Output from a DataSourceBackend.

    Attributes
    ----------
    chunks:      Ordered list of matching DataChunks.
    source:      Backend identifier string.
    request_id:  Echoed from the request.
    total_found: Total matching items in the backend (may exceed top_k).
    latency_ms:  Wall-clock time for the query.
    metadata:    Any extra backend-specific info.
    """
    chunks:      List[DataChunk]    = field(default_factory=list)
    source:      str                = ""
    request_id:  str                = ""
    total_found: int                = 0
    latency_ms:  float              = 0.0
    metadata:    Dict[str, Any]     = field(default_factory=dict)

    def texts(self) -> List[str]:
        """Return content strings from all text-modality chunks."""
        return [c.content for c in self.chunks if c.modality == "text"]

    def top(self) -> Optional[DataChunk]:
        """Return the highest-scoring chunk, or the first one."""
        if not self.chunks:
            return None
        return max(self.chunks, key=lambda c: c.score) if any(
            c.score > 0 for c in self.chunks
        ) else self.chunks[0]


@dataclass
class DataSourceHealthStatus:
    """Health / readiness report for a data source node."""
    healthy:    bool
    source:     str
    message:    str              = ""
    item_count: int              = 0
    load:       Dict[str, Any]   = field(default_factory=dict)


@dataclass
class DataSourceCapability:
    """
    Describes what a data source node provides — used by the registry/router.

    ``source_type`` is "text", "image", "vector_db", "graph_db", or "custom".
    ``modalities``  lists output modalities this source supports.
    ``node_id``     matches the registry NodeCapability.node_id.
    """
    source_type:    str
    modalities:     List[str]        = field(default_factory=list)
    item_count:     int              = 0
    node_id:        str              = ""
    extra:          Dict[str, Any]   = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base class
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


class DataSourceBackend(ABC):
    """
    Abstract base for all Ravnest data source backends.

    Sub-classes **must** implement the four async methods.
    Sync wrappers (``query``, ``stream``, ``health``) are provided for
    convenience in scripts and tests.
    """

    # ── async interface (implement these) ────────────────────────────────

    @abstractmethod
    async def aquery(self, request: DataRequest) -> DataResponse:
        """Execute a query and return a complete DataResponse."""
        ...

    @abstractmethod
    async def astream(self, request: DataRequest) -> AsyncIterator[DataChunk]:
        """Execute a query and yield DataChunks as they are retrieved."""
        ...

    @abstractmethod
    async def ahealth(self) -> DataSourceHealthStatus:
        """Return the health / readiness of this data source."""
        ...

    @abstractmethod
    def capabilities(self) -> DataSourceCapability:
        """Describe what this data source provides."""
        ...

    # ── sync wrappers ────────────────────────────────────────────────────

    def query(self, request: DataRequest) -> DataResponse:
        """Synchronous wrapper around ``aquery``."""
        return _run(self.aquery(request))

    def stream(self, request: DataRequest):
        """
        Synchronous generator wrapper around ``astream``.

        Usage::

            for chunk in source.stream(req):
                print(chunk.content)
        """
        async def _collect():
            chunks = []
            async for chunk in self.astream(request):
                chunks.append(chunk)
            return chunks

        return iter(_run(_collect()))

    def health(self) -> DataSourceHealthStatus:
        """Synchronous wrapper around ``ahealth``."""
        return _run(self.ahealth())

    # ── registry helpers ─────────────────────────────────────────────────

    def register_with_registry(self, registry_address: str) -> None:
        """Register this data source with the Ravnest node registry."""
        from ravnest.registry import RegistryClient, HeartbeatSender
        from ravnest.registry.capability import NodeCapability, NodeType

        cap  = self.capabilities()
        node = NodeCapability(
            node_id   = cap.node_id,
            node_type = NodeType.DATA_SOURCE,
            subtype   = cap.source_type,
            address   = cap.extra.get("address", ""),
            metadata  = {
                "modalities":  cap.modalities,
                "item_count":  cap.item_count,
                **cap.extra,
            },
        )
        client = RegistryClient(registry_address)
        client.register(node)
        self._registry_client  = client
        self._heartbeat_sender = HeartbeatSender(client, cap.node_id)
        self._heartbeat_sender.start()

    def deregister_from_registry(self) -> None:
        if hasattr(self, "_heartbeat_sender"):
            self._heartbeat_sender.stop()
        if hasattr(self, "_registry_client"):
            cap = self.capabilities()
            self._registry_client.deregister(cap.node_id)
