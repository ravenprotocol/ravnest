"""
ravnest.data_sources — Data source node framework.

Every node that participates as a "data source" in the Ravnest network
implements ``DataSourceBackend``.  The router discovers data source nodes
from the registry and dispatches ``DataRequest`` objects to the best one.

Quick-start
-----------
>>> from ravnest.data_sources import (
...     DataRequest, DataResponse, DataChunk,
...     TextSource, ImageSource, VectorDBSource, GraphDBSource,
...     DataRouter,
... )
>>>
>>> # Text search (in-process, no server needed)
>>> source = TextSource(paths=["/data/docs"])
>>> resp   = source.query(DataRequest(query="distributed training", top_k=3))
>>> for chunk in resp.chunks:
...     print(chunk.score, chunk.content[:60])
>>>
>>> # Vector similarity (NumPy fallback, always available)
>>> vdb = VectorDBSource(backend="numpy", collection="kb",
...                      embed_model=None)   # embed_model=None: raw vectors only
>>> vdb.add_vectors([[0.1, 0.2, 0.3]], texts=["example text"])
>>>
>>> # Graph (NetworkX, always available)
>>> gdb = GraphDBSource(backend="networkx")
>>> gdb.add_triples([("Alice", "KNOWS", "Bob")])

Source types
------------
- TextSource     — local files/directories, BM25-style ranking
- ImageSource    — local/remote images, optional CLIP semantic search
- VectorDBSource — Chroma / Qdrant / Pinecone / FAISS / NumPy
- GraphDBSource  — Neo4j / NetworkX

Router strategies
-----------------
- LoadBasedDataStrategy  — route to least-loaded node (default)
- RoundRobinDataStrategy — distribute evenly
- SourceTypeStrategy     — prefer exact source_type match, then load-based
"""

from .base import (
    DataChunk,
    DataRequest,
    DataResponse,
    DataSourceBackend,
    DataSourceCapability,
    DataSourceHealthStatus,
)

from .text_source  import TextSource
from .image_source import ImageSource
from .vector_db    import VectorDBSource
from .graph_db     import GraphDBSource

from .router import (
    DataRouter,
    DataRoutingStrategy,
    LoadBasedDataStrategy,
    RoundRobinDataStrategy,
    SourceTypeStrategy,
)

__all__ = [
    # ── data classes ──────────────────────────────────────────────────────
    "DataChunk",
    "DataRequest",
    "DataResponse",
    "DataSourceCapability",
    "DataSourceHealthStatus",
    # ── abstract base ─────────────────────────────────────────────────────
    "DataSourceBackend",
    # ── concrete backends ─────────────────────────────────────────────────
    "TextSource",
    "ImageSource",
    "VectorDBSource",
    "GraphDBSource",
    # ── router + strategies ───────────────────────────────────────────────
    "DataRouter",
    "DataRoutingStrategy",
    "LoadBasedDataStrategy",
    "RoundRobinDataStrategy",
    "SourceTypeStrategy",
]
