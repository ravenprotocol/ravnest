"""
node_compute_ravnest.py — Llama compute node using Ravnest pipeline parallelism.

Splits a Llama model across N machines (or N processes on one machine) using
Ravnest's pipeline-parallel InferenceEngine.  The ROOT rank hosts the mesh
NodeServer; STEM/LEAF ranks participate in the forward pass automatically via
torch.distributed.

This script MUST be launched with torchrun — do not run it with plain python3.

How it works
------------
  - Every rank (ROOT, STEM, LEAF) loads the full model metadata via
    LazyInitContext — no GPU memory is used at this stage.
  - node_tcp.Node distributes model layers across ranks:
      ROOT  → embedding + first N layers
      STEM  → middle N layers   (only present when num_nodes >= 3)
      LEAF  → last N layers + lm_head
  - InferenceEngine wraps the node; all ranks call generate() in lock-step.
  - Only ROOT has the NodeServer HTTP endpoint that the agent calls.
  - STEM/LEAF ranks loop on generate() with an empty prompt list, blocking
    until ROOT drives them with real activations.

Network topology for the SQL agent example
------------------------------------------
                       ┌──────────────────┐
  node_agent.py  ─────►  ROOT  rank 0      │  :8765  NodeServer
                       │  STEM  rank 1      │         (no HTTP)
                       │  LEAF  rank 2      │         (no HTTP)
                       └──────────────────┘
  (ranks communicate via torch.distributed NCCL/Gloo, not HTTP)

Launching
---------
  # 3 nodes on one machine (e.g. 3 GPUs):
  torchrun --nnodes=1 --nproc_per_node=3 node_compute_ravnest.py \\
      --model meta-llama/Llama-3.2-8B-Instruct --num-nodes 3

  # 3 physical machines:
  #  Machine 0 (root/master):
  torchrun --nnodes=3 --nproc_per_node=1 --node_rank=0 \\
      --master_addr=<machine0-ip> --master_port=29500 \\
      node_compute_ravnest.py --model meta-llama/Llama-3.2-8B-Instruct --num-nodes 3

  #  Machine 1:
  torchrun --nnodes=3 --nproc_per_node=1 --node_rank=1 \\
      --master_addr=<machine0-ip> --master_port=29500 \\
      node_compute_ravnest.py --model meta-llama/Llama-3.2-8B-Instruct --num-nodes 3

  #  Machine 2:
  torchrun --nnodes=3 --nproc_per_node=1 --node_rank=2 \\
      --master_addr=<machine0-ip> --master_port=29500 \\
      node_compute_ravnest.py --model meta-llama/Llama-3.2-8B-Instruct --num-nodes 3

  # Point the agent at the ROOT node:
  python3 node_agent.py --compute-url http://<machine0-ip>:8765

Prerequisites
-------------
    pip install torch transformers aiohttp
    # GPU required (NCCL backend); for CPU-only testing use --backend gloo
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ravnest.lazy_init.lazy_context    import LazyInitContext
from ravnest.node_tcp                  import Node
from ravnest.node                      import NodeTypes
from ravnest.inference.inference_engine import InferenceEngine
from ravnest.compute.ravnest_backend   import RavnestBackend
from ravnest.mesh.node_server          import NodeServer


def main() -> None:
    p = argparse.ArgumentParser(
        description="Ravnest pipeline-parallel compute node (torchrun-launched)"
    )
    p.add_argument("--port",       default=8765,   type=int,
                   help="HTTP port for the ROOT rank's NodeServer")
    p.add_argument("--host",       default="0.0.0.0")
    p.add_argument("--model",      default="meta-llama/Llama-3.2-8B-Instruct",
                   help="HuggingFace model ID or local path")
    p.add_argument("--num-nodes",  default=3,      type=int,
                   help="Total number of pipeline stages (must match torchrun --nnodes)")
    p.add_argument("--seq-length", default=128,    type=int,
                   help="Initial sequence length for the pipeline configuration")
    p.add_argument("--batch-size", default=1,      type=int)
    p.add_argument("--backend",    default="nccl",
                   choices=["nccl", "gloo"],
                   help="torch.distributed backend (gloo for CPU / no NVLink)")
    p.add_argument("--dtype",      default="float16",
                   choices=["float16", "bfloat16", "float32"])
    args = p.parse_args()

    local_rank = int(torch.distributed.get_rank()) if torch.distributed.is_initialized() else 0
    device     = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch_dtype = {
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
        "float32":  torch.float32,
    }[args.dtype]

    print(f"[ravnest-node] rank={local_rank}  device={device}")
    print(f"[ravnest-node]   model      = {args.model}")
    print(f"[ravnest-node]   num_nodes  = {args.num_nodes}")
    print(f"[ravnest-node]   backend    = {args.backend}")
    print(f"[ravnest-node] Loading model weights lazily…")

    # Load model structure without materialising weights (memory-efficient).
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    init_ctx = LazyInitContext()
    with init_ctx:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype = torch_dtype,
            device_map  = str(device),
        )

    # Node.__init__ calls configure_model() which:
    #   1. Calls get_split_spec(model) → LlamaForCausalLMSplitSpec
    #   2. Assigns this rank's layer slice
    #   3. Initialises torch.distributed (sets up NCCL/Gloo process group)
    #   4. Sets self.node_type = ROOT / STEM / LEAF
    node = Node(
        model          = model,
        device         = device,
        dtype          = args.dtype,
        batch_size     = args.batch_size,
        seq_length     = args.seq_length,
        cluster_length = args.num_nodes,
        mode           = "inference",
        backend        = args.backend,
        reduce_factor  = 1,
    )

    inference_engine = InferenceEngine(node, tokenizer)
    backend          = RavnestBackend(inference_engine)

    if node.node_type == NodeTypes.ROOT:
        print(f"[ravnest-node] ROOT rank — starting NodeServer on :{args.port}")
        server = NodeServer(host=args.host, port=args.port)
        server.add_compute(backend)
        # run() is blocking; it drives InferenceEngine.generate() for every
        # incoming request.  STEM/LEAF ranks unblock in lock-step via NCCL.
        server.run()
    else:
        role = "STEM" if node.node_type == NodeTypes.STEM else "LEAF"
        print(f"[ravnest-node] {role} rank — waiting for ROOT to drive inference…")
        # STEM/LEAF loop: each call blocks until ROOT sends activations through
        # the pipeline, then returns None (output only on ROOT).
        while True:
            try:
                inference_engine.generate([], [])
            except Exception as exc:
                print(f"[ravnest-node] {role} generate error: {exc}")
                time.sleep(0.1)


if __name__ == "__main__":
    main()
