"""
node_compute.py — Llama compute node (port 8765).

Hosts Llama 3.2 via Ollama and exposes it as a Ravnest mesh NodeServer.
Every LLM call from the agent is routed through this node.

Swapping backends
-----------------
By default this uses OllamaBackend (single process, Ollama must be running).
To use the actual Ravnest distributed pipeline across multiple machines, swap
in RavnestBackend:

    from ravnest.compute.ravnest_backend import RavnestBackend
    from ravnest.inference_engine import InferenceEngine

    engine  = InferenceEngine(model_path="meta-llama/Llama-3.2-8B", ...)
    backend = RavnestBackend(engine=engine, role="ROOT")

The NodeServer API is identical — the rest of the network doesn't change.

Usage
-----
    python3 node_compute.py [--port 8765] [--model llama3.2] [--host 0.0.0.0]
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from ravnest.mesh.node_server      import NodeServer
from ravnest.compute.ollama_backend import OllamaBackend


def main():
    p = argparse.ArgumentParser(description="Ravnest compute node (Llama via Ollama)")
    p.add_argument("--port",  default=8765,       type=int)
    p.add_argument("--host",  default="0.0.0.0")
    p.add_argument("--model", default="llama3.2")
    p.add_argument("--base-url", default="http://localhost:11434")
    args = p.parse_args()

    print(f"[compute] Starting Llama node")
    print(f"[compute]   model    = {args.model}")
    print(f"[compute]   ollama   = {args.base_url}")
    print(f"[compute]   endpoint = http://{args.host}:{args.port}")
    print(f"[compute]   (swap OllamaBackend → RavnestBackend for true distribution)")

    backend = OllamaBackend(model=args.model, base_url=args.base_url)
    server  = NodeServer(host=args.host, port=args.port)
    server.add_compute(backend)
    server.run()


if __name__ == "__main__":
    main()
