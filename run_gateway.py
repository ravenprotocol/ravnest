#!/usr/bin/env python3
"""
run_gateway.py — Start a standalone Ravnest HTTP gateway.

This script starts the gateway server which external clients can use to
interact with any combination of compute, agent, and data-source backends.

Usage
-----
    python3 run_gateway.py [OPTIONS]

Options
-------
  --host HOST               Bind host (default: 0.0.0.0)
  --port PORT               Bind port (default: 8080)
  --compute-registry ADDR   Registry address for compute nodes (host:port)
  --agent-registry ADDR     Registry address for agent nodes (host:port)
  --data-registry ADDR      Registry address for data source nodes (host:port)
  --log-level LEVEL         Logging level (default: INFO)
  --no-cors                 Disable permissive CORS headers
  --help                    Show this message

Quick start (no registry, local Ollama backend)
-----------------------------------------------
    python3 run_gateway.py --port 8080

    curl -s -X POST http://localhost:8080/chat \\
         -H "Content-Type: application/json" \\
         -d '{"prompt": "Hello!", "model": "llama3.2"}'

Quick start with a registry
----------------------------
    # Terminal 1 — start registry
    python3 run_registry.py --address 0.0.0.0:50099

    # Terminal 2 — start gateway
    python3 run_gateway.py --compute-registry localhost:50099

    # Terminal 3 — register a compute node and call it
    # (see examples/gateway/demo.py for a full walkthrough)
"""

import argparse
import logging
import sys

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Ravnest HTTP Gateway",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--host",             default="0.0.0.0",  metavar="HOST")
    p.add_argument("--port",             default=8080,        type=int, metavar="PORT")
    p.add_argument("--compute-registry", default=None,        metavar="ADDR",
                   help="host:port of the Ravnest registry for compute nodes")
    p.add_argument("--agent-registry",   default=None,        metavar="ADDR",
                   help="host:port of the Ravnest registry for agent nodes")
    p.add_argument("--data-registry",    default=None,        metavar="ADDR",
                   help="host:port of the Ravnest registry for data source nodes")
    p.add_argument("--log-level",        default="INFO",      metavar="LEVEL")
    p.add_argument("--no-cors",          action="store_true",
                   help="Disable CORS headers")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        from ravnest.gateway import Orchestrator, GatewayServer
    except ImportError as exc:
        logger.error("Failed to import gateway: %s", exc)
        sys.exit(1)

    orch = Orchestrator(
        compute_registry     = args.compute_registry,
        agent_registry       = args.agent_registry,
        data_source_registry = args.data_registry,
    )

    # Attempt to attach a local Ollama backend if no compute registry given
    # and ollama is running on localhost.
    if not args.compute_registry:
        _try_attach_local_ollama(orch)

    server = GatewayServer(
        orchestrator = orch,
        host         = args.host,
        port         = args.port,
        cors         = not args.no_cors,
        log_level    = args.log_level,
    )

    print(f"Ravnest Gateway  →  http://{args.host}:{args.port}")
    print("Endpoints: /chat  /query  /rag  /pipeline  /v1/chat/completions  /health  /nodes")
    server.run()


def _try_attach_local_ollama(orch):
    """Auto-attach Ollama backend if it is reachable on localhost:11434."""
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
    except Exception:
        return  # Ollama not running — that's fine

    try:
        from ravnest.compute.ollama_backend import OllamaBackend
        # Default model; user can override via request.model
        backend = OllamaBackend(model="llama3.2", base_url="http://localhost:11434")
        orch.add_local_compute(backend)
        logger.info("[run_gateway] Auto-attached local Ollama backend (llama3.2)")
    except Exception as exc:
        logger.warning("[run_gateway] Ollama auto-attach failed: %s", exc)


if __name__ == "__main__":
    main()
