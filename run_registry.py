"""
Standalone registry server entry point.

Usage:
    python run_registry.py
    python run_registry.py --address 0.0.0.0:50099 --ttl 30

Nodes register themselves on startup by passing registry_address to Node():
    node = Node(model=model, ..., registry_address="<registry_host>:50099")

Query the network from any process:
    from ravnest.registry import RegistryClient, NodeType
    with RegistryClient("registry_host:50099") as client:
        compute_nodes = client.discover(node_type=NodeType.PIPELINE_COMPUTE)
        print(compute_nodes)
"""

import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from ravnest.registry.node_registry import serve, REGISTRY_DEFAULT_PORT

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ravnest decentralized node registry")
    parser.add_argument(
        "--address",
        default=f"0.0.0.0:{REGISTRY_DEFAULT_PORT}",
        help=f"gRPC listen address (default: 0.0.0.0:{REGISTRY_DEFAULT_PORT})",
    )
    parser.add_argument(
        "--ttl",
        type=float,
        default=30.0,
        help="Seconds before a silent node is evicted (default: 30)",
    )
    args = parser.parse_args()
    serve(address=args.address, ttl_seconds=args.ttl)
