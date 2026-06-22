"""
ravnest.mesh — Unified inter-node communication layer.

Provides the message envelope, HTTP node server, client, and pipeline
builder that let compute, agent, and data-source nodes discover and call
each other across the Ravnest mesh.

Quick-start
-----------
>>> # 1 — expose a local backend as a mesh node
>>> from ravnest.mesh import NodeServer
>>> from ravnest.data_sources.text_source import TextSource
>>>
>>> server = NodeServer(port=8765)
>>> server.add_data_source(TextSource(paths=["/data/docs"]))
>>> server.run()   # blocking

>>> # 2 — call it from another process
>>> from ravnest.mesh import NodeClient, NodeMessage
>>> client = NodeClient("http://localhost:8765")
>>> resp = client.send(NodeMessage.data_query("distributed training", top_k=3))
>>> for chunk in resp.result["chunks"]:
...     print(chunk["score"], chunk["content"][:60])

>>> # 3 — chain nodes into a pipeline (no server needed for local backends)
>>> from ravnest.mesh import Pipeline, PipelineStep
>>> from ravnest.data_sources.text_source  import TextSource
>>> from ravnest.compute.ollama_backend    import OllamaBackend
>>>
>>> pipeline = (
...     Pipeline()
...     .step(PipelineStep(node_type="data_source", label="retrieve"))
...     .step(PipelineStep(node_type="compute",     label="generate",
...                        extra={"max_tokens": 256}))
... )
>>> pipeline.add_local_data_source(TextSource(paths=["/data/docs"]))
>>> pipeline.add_local_compute(OllamaBackend(model="llama3.2"))
>>>
>>> result = pipeline.run("What is pipeline parallelism?")
>>> print(result.text())
"""

from .base import (
    NodeMessage,
    NodeResponse,
    PipelineStep,
    PipelineResult,
)

from .node_server import NodeServer
from .node_client import NodeClient
from .pipeline    import Pipeline

__all__ = [
    # ── message types ─────────────────────────────────────────────────────
    "NodeMessage",
    "NodeResponse",
    "PipelineStep",
    "PipelineResult",
    # ── server & client ───────────────────────────────────────────────────
    "NodeServer",
    "NodeClient",
    # ── pipeline ──────────────────────────────────────────────────────────
    "Pipeline",
]
