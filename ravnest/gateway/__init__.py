"""
ravnest.gateway — HTTP gateway and orchestration layer.

The gateway is the single external-facing entry point for the Ravnest mesh.
External clients send a ``GatewayRequest`` (JSON over HTTP) and receive a
``GatewayResponse`` regardless of whether the request was handled by a compute
node, an agent, a data source, or a pipeline.

Quick-start
-----------
>>> from ravnest.gateway import GatewayServer, Orchestrator
>>> from ravnest.compute.ollama_backend    import OllamaBackend
>>> from ravnest.data_sources.text_source  import TextSource
>>>
>>> orch = Orchestrator()
>>> orch.add_local_compute(OllamaBackend("llama3.2"))
>>> orch.add_local_data_source(TextSource(paths=["/data/docs"]))
>>>
>>> GatewayServer(orch, port=8080).run()

From a client:
>>> import requests
>>> r = requests.post("http://localhost:8080/rag",
...                   json={"prompt": "What is pipeline parallelism?",
...                         "source_type": "text"})
>>> print(r.json()["text"])
"""

from .base         import GatewayRequest, GatewayResponse
from .orchestrator import Orchestrator
from .server       import GatewayServer

__all__ = [
    "GatewayRequest",
    "GatewayResponse",
    "Orchestrator",
    "GatewayServer",
]
