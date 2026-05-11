"""
Registry smoke test / demo
==========================

Exercises the full gRPC round-trip:
  1. Start the registry server in a background thread
  2. Register three fake nodes (compute, agent, data_source)
  3. Discover nodes with various filters
  4. Send a heartbeat and verify load update
  5. Deregister one node and verify it disappears

Run:
    python examples/registry/demo.py

No GPU or external services required.  All traffic is on localhost.
"""

import sys
import time
import threading

REGISTRY_ADDR = "127.0.0.1:50199"   # use a non-standard port to avoid conflicts


# ------------------------------------------------------------------ #
# Step 0: start the registry server in a background thread            #
# ------------------------------------------------------------------ #

def _start_registry():
    from ravnest.registry.node_registry import serve
    serve(address=REGISTRY_ADDR, ttl_seconds=15, block=True)

registry_thread = threading.Thread(target=_start_registry, daemon=True)
registry_thread.start()
time.sleep(0.5)   # give gRPC server a moment to bind


# ------------------------------------------------------------------ #
# Step 1: build fake NodeCapability objects                           #
# ------------------------------------------------------------------ #

import time as _t
from ravnest.registry import (
    RegistryClient,
    NodeCapability, NodeType, ComputeSubtype, AgentSubtype, DataSubtype,
    ResourceSpec,
)

nodes = [
    NodeCapability(
        node_id   = "compute-0",
        node_type = NodeType.PIPELINE_COMPUTE,
        subtype   = ComputeSubtype.RAVNEST,
        address   = "10.0.0.1:8080",
        resources = ResourceSpec(ram_mb=16384, vram_mb=24576, cpu_cores=8, bandwidth_mbps=1000),
        models    = ["LlamaForCausalLM", "Qwen2ForCausalLM"],
        metadata  = {"rank": 0, "world_size": 3, "pipeline_role": "root"},
    ),
    NodeCapability(
        node_id   = "agent-research-0",
        node_type = NodeType.AGENT,
        subtype   = AgentSubtype.RESEARCH,
        address   = "10.0.0.2:9090",
        resources = ResourceSpec(ram_mb=8192, cpu_cores=4),
        models    = ["gpt-4o", "claude-3-5-sonnet"],
        metadata  = {"tools": ["web_search", "arxiv", "wikipedia"]},
    ),
    NodeCapability(
        node_id   = "vector-db-0",
        node_type = NodeType.DATA_SOURCE,
        subtype   = DataSubtype.VECTOR_DB,
        address   = "10.0.0.3:6333",
        resources = ResourceSpec(ram_mb=32768, disk_gb=500),
        metadata  = {"engine": "qdrant", "collections": ["papers", "code", "docs"]},
    ),
]


# ------------------------------------------------------------------ #
# Step 2: register all nodes                                          #
# ------------------------------------------------------------------ #

client = RegistryClient(REGISTRY_ADDR, cache_ttl=5.0)

print("\n=== Registering nodes ===")
for node in nodes:
    ok = client.register(node)
    print(f"  register({node.node_id}) -> {'OK' if ok else 'FAILED'}")


# ------------------------------------------------------------------ #
# Step 3: discover                                                     #
# ------------------------------------------------------------------ #

print("\n=== Discover all nodes ===")
all_nodes = client.discover()
for n in all_nodes:
    print(f"  {n.node_id:25s}  type={n.node_type:25s}  subtype={n.subtype}")

assert len(all_nodes) == 3, f"Expected 3 nodes, got {len(all_nodes)}"

print("\n=== Discover compute nodes only ===")
compute_nodes = client.discover(node_type=NodeType.PIPELINE_COMPUTE)
for n in compute_nodes:
    print(f"  {n.node_id}")
assert len(compute_nodes) == 1

print("\n=== Discover nodes serving LlamaForCausalLM ===")
llama_nodes = client.discover(models=["LlamaForCausalLM"])
for n in llama_nodes:
    print(f"  {n.node_id} -> models={n.models}")
assert len(llama_nodes) == 1 and llama_nodes[0].node_id == "compute-0"

print("\n=== Discover vector DB data sources ===")
vdb_nodes = client.discover(node_type=NodeType.DATA_SOURCE, subtype=DataSubtype.VECTOR_DB)
for n in vdb_nodes:
    print(f"  {n.node_id} -> metadata={n.metadata}")
assert len(vdb_nodes) == 1


# ------------------------------------------------------------------ #
# Step 4: cache hit check                                             #
# ------------------------------------------------------------------ #

print("\n=== Cache hit check (same query, should be served from cache) ===")
t0 = time.monotonic()
cached_result = client.discover(node_type=NodeType.PIPELINE_COMPUTE)
t1 = time.monotonic()
print(f"  Returned {len(cached_result)} node(s) in {(t1-t0)*1000:.2f} ms (cache TTL=5s)")
assert len(cached_result) == 1


# ------------------------------------------------------------------ #
# Step 5: heartbeat + load update                                     #
# ------------------------------------------------------------------ #

print("\n=== Heartbeat with load update ===")
ok = client.heartbeat("compute-0", {
    "cpu_percent":      42.5,
    "ram_percent":      61.0,
    "gpu_percent":      88.0,
    "gpu_vram_percent": 73.0,
})
print(f"  heartbeat(compute-0) -> {'OK' if ok else 'FAILED'}")
assert ok

# Force-fetch to bypass cache and see updated load
refreshed = client.get_node("compute-0", force=True)
assert refreshed is not None
load = refreshed.current_load
print(f"  Updated load: cpu={load['cpu_percent']}%  ram={load['ram_percent']}%  "
      f"gpu={load['gpu_percent']}%  vram={load['gpu_vram_percent']}%")
assert abs(load["gpu_percent"] - 88.0) < 0.1


# ------------------------------------------------------------------ #
# Step 6: deregister one node and verify                              #
# ------------------------------------------------------------------ #

print("\n=== Deregister agent node ===")
ok = client.deregister("agent-research-0")
print(f"  deregister(agent-research-0) -> {'OK' if ok else 'FAILED'}")
assert ok

# Cache was invalidated by deregister, so this is a fresh RPC.
remaining = client.discover()
print(f"  Remaining nodes: {[n.node_id for n in remaining]}")
assert len(remaining) == 2
assert all(n.node_id != "agent-research-0" for n in remaining)


# ------------------------------------------------------------------ #
# Step 7: TTL eviction (optional slow check — disabled by default)    #
# ------------------------------------------------------------------ #
# Uncomment to verify that nodes missing heartbeats are evicted.
#
# print("\n=== TTL eviction (waiting 20s for ttl=15s registry) ===")
# client.heartbeat("compute-0", {})   # keep compute-0 alive
# time.sleep(20)
# evicted = client.discover(force=True)
# print(f"  After eviction: {[n.node_id for n in evicted]}")
# assert "vector-db-0" not in [n.node_id for n in evicted], "vector-db-0 should have been evicted"


# ------------------------------------------------------------------ #
# Done                                                                 #
# ------------------------------------------------------------------ #

client.close()
print("\n✓ All assertions passed — registry demo complete.")
