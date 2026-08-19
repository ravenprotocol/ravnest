"""
node_compute_vllm.py — Llama compute node using vLLM (port 8765).

Hosts any HuggingFace-compatible LLM via vLLM's AsyncLLMEngine and exposes it
as a Ravnest mesh NodeServer.  Supports tensor parallelism across multiple GPUs
on a single machine via --tensor-parallel-size.

Compared to node_compute.py (Ollama):
  - Runs the model in-process — no external Ollama daemon needed.
  - Better throughput via continuous batching.
  - Native tensor parallelism (--tensor-parallel-size > 1 for multi-GPU).
  - Requires a GPU and `pip install vllm`.

Usage
-----
    # Single GPU
    python3 node_compute_vllm.py --model meta-llama/Llama-3.2-3B-Instruct

    # Two GPUs (tensor parallel)
    python3 node_compute_vllm.py \\
        --model meta-llama/Llama-3.2-8B-Instruct \\
        --tensor-parallel-size 2

    # Quantised / reduced precision
    python3 node_compute_vllm.py \\
        --model meta-llama/Llama-3.2-8B-Instruct \\
        --dtype bfloat16 \\
        --gpu-memory-utilization 0.85

    # Point the agent at this node:
    python3 node_agent.py --compute-url http://localhost:8765

Prerequisites
-------------
    pip install vllm aiohttp
    # HuggingFace model weights must be accessible (local or via HF_TOKEN).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from ravnest.mesh.node_server         import NodeServer
from ravnest.compute.vllm_backend     import VLLMBackend


def main() -> None:
    p = argparse.ArgumentParser(description="Ravnest compute node (vLLM backend)")
    p.add_argument("--port",                   default=8765,        type=int)
    p.add_argument("--host",                   default="0.0.0.0")
    p.add_argument("--model",                  default="meta-llama/Llama-3.2-3B-Instruct",
                   help="HuggingFace model ID or local path")
    p.add_argument("--tensor-parallel-size",   default=1,           type=int,
                   help="Number of GPUs to use for tensor parallelism")
    p.add_argument("--dtype",                  default="auto",
                   choices=["auto", "float16", "bfloat16", "float32"],
                   help="Weight dtype (auto = detect from model config)")
    p.add_argument("--max-model-len",          default=None,        type=int,
                   help="Override max context length (tokens)")
    p.add_argument("--gpu-memory-utilization", default=0.90,        type=float,
                   help="Fraction of GPU memory vLLM may use (0-1)")
    args = p.parse_args()

    print("[compute-vllm] Starting vLLM compute node")
    print(f"[compute-vllm]   model                  = {args.model}")
    print(f"[compute-vllm]   tensor_parallel_size   = {args.tensor_parallel_size}")
    print(f"[compute-vllm]   dtype                  = {args.dtype}")
    print(f"[compute-vllm]   gpu_memory_utilization = {args.gpu_memory_utilization}")
    print(f"[compute-vllm]   endpoint               = http://{args.host}:{args.port}")
    print("[compute-vllm] Loading model (this may take a minute)…")

    backend = VLLMBackend(
        model                  = args.model,
        tensor_parallel_size   = args.tensor_parallel_size,
        dtype                  = args.dtype,
        max_model_len          = args.max_model_len,
        gpu_memory_utilization = args.gpu_memory_utilization,
    )

    server = NodeServer(host=args.host, port=args.port)
    server.add_compute(backend)
    server.run()


if __name__ == "__main__":
    main()
