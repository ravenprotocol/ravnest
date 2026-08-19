"""
run_network.py — Launch the full distributed SQL agent network and run a demo.

Starts three processes:
    1. node_compute.py  (Llama compute node, port 8765)  ← backend is selectable
    2. node_sql.py      (SQLite tool node,   port 8766)
    3. node_agent.py    (ReAct SQL agent,    port 8767)

Waits for all three to be healthy, then sends a demo question and prints
the answer.  Ctrl-C shuts everything down cleanly.

Backend choices (--backend)
---------------------------
  ollama   (default)  OllamaBackend — requires `ollama` daemon + `ollama pull <model>`
  vllm                VLLMBackend   — requires `pip install vllm` and a GPU
  ravnest             RavnestBackend via torchrun pipeline parallelism — requires
                      GPU(s) and `pip install torch transformers`; use
                      --num-nodes to set the pipeline depth

Prerequisites
-------------
    pip install aiohttp
    python3 setup_db.py          # create shop.db

    # Ollama backend (default):
    pip install ollama
    ollama pull llama3.2

    # vLLM backend:
    pip install vllm

    # Ravnest backend:
    pip install torch transformers

Usage
-----
    python3 run_network.py
    python3 run_network.py --backend vllm --model meta-llama/Llama-3.2-3B-Instruct
    python3 run_network.py --backend ravnest --model meta-llama/Llama-3.2-8B-Instruct --num-nodes 3
    python3 run_network.py --query "Which city has the most customers?"
    python3 run_network.py --no-demo   # just start the nodes, don't send a query
"""

from __future__ import annotations

import argparse
import pathlib
import signal
import subprocess
import sys
import time
import urllib.request

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE      = pathlib.Path(__file__).resolve().parent
DB_PATH   = HERE / "shop.db"

SQL_NODES = [
    {
        "name":   "sql-tool",
        "script": HERE / "node_sql.py",
        "port":   8766,
        "health": "http://localhost:8766/health",
        "extra":  ["--db", str(DB_PATH)],
    },
    {
        "name":   "agent",
        "script": HERE / "node_agent.py",
        "port":   8767,
        "health": "http://localhost:8767/health",
        "extra":  ["--db", str(DB_PATH)],
    },
]

# Compute node config per backend — filled in at runtime
COMPUTE_NODE_DEFAULTS: dict[str, dict] = {
    "ollama": {
        "script": HERE / "node_compute.py",
        "extra":  [],
    },
    "vllm": {
        "script": HERE / "node_compute_vllm.py",
        "extra":  [],
    },
    "ravnest": {
        "script": HERE / "node_compute_ravnest.py",
        "extra":  [],   # torchrun args added dynamically
    },
}


# ── process management ────────────────────────────────────────────────────────

def start_compute_node(backend: str, model: str, port: int, host: str,
                       num_nodes: int, ravnest_master_addr: str,
                       ravnest_master_port: int) -> subprocess.Popen:
    """Launch the compute node subprocess for the chosen backend."""
    cfg = COMPUTE_NODE_DEFAULTS[backend]

    if backend == "ravnest":
        # torchrun handles distributed init; one process per pipeline stage
        cmd = [
            "torchrun",
            f"--nnodes={num_nodes}",
            "--nproc_per_node=1",
            "--node_rank=0",
            f"--master_addr={ravnest_master_addr}",
            f"--master_port={ravnest_master_port}",
            str(cfg["script"]),
            "--port", str(port),
            "--host", host,
            "--model", model,
            "--num-nodes", str(num_nodes),
            *cfg["extra"],
        ]
        print("[run] NOTE: Ravnest backend requires additional torchrun invocations")
        print(f"[run]   for ranks 1..{num_nodes - 1} on the other machines.")
        print(f"[run]   See node_compute_ravnest.py docstring for details.")
    else:
        cmd = [
            sys.executable, str(cfg["script"]),
            "--port", str(port),
            "--host", host,
            "--model", model,
            *cfg["extra"],
        ]

    return subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        bufsize = 1,
        text    = True,
    )


def start_node(node: dict, model: str) -> subprocess.Popen:
    cmd = [
        sys.executable, str(node["script"]),
        "--port", str(node["port"]),
        *node["extra"],
    ]
    if node["name"] == "agent":
        cmd += ["--model", model]

    return subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        bufsize = 1,
        text    = True,
    )


def wait_healthy(url: str, name: str, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status in (200, 207):
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    print(f"  [run] TIMEOUT: {name} not healthy after {timeout}s")
    return False


# ── demo query ────────────────────────────────────────────────────────────────

def send_query(question: str, agent_url: str = "http://localhost:8767") -> str:
    import json
    import urllib.request

    payload = json.dumps({
        "node_type": "agent",
        "action":    "run",
        "payload": {
            "messages":    [{"role": "user", "content": question}],
            "max_steps":   6,
            "max_tokens":  512,
            "temperature": 0.1,
        },
    }).encode()

    req = urllib.request.Request(
        f"{agent_url}/message",
        data    = payload,
        headers = {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())

    if data.get("ok"):
        return data["result"].get("text", "(no answer)")
    return f"Error: {data.get('error', 'unknown')}"


# ── entrypoint ────────────────────────────────────────────────────────────────

DEMO_QUESTIONS = [
    "Who are the top 3 customers by total spend?",
    "What is the best-selling product category?",
    "Which products cost less than $50?",
]


def main():
    p = argparse.ArgumentParser(description="Launch the distributed SQL agent network")
    p.add_argument("--model",   default=None,
                   help="Model name / HF model ID (default depends on --backend)")
    p.add_argument("--backend", default="ollama",
                   choices=["ollama", "vllm", "ravnest"],
                   help="Compute backend for the LLM node")
    p.add_argument("--num-nodes", default=3, type=int,
                   help="[ravnest only] Number of pipeline stages")
    p.add_argument("--ravnest-master-addr", default="localhost",
                   help="[ravnest only] torchrun master address")
    p.add_argument("--ravnest-master-port", default=29500, type=int,
                   help="[ravnest only] torchrun master port")
    p.add_argument("--query",   default=None,
                   help="Ask a custom question after startup (overrides demo questions)")
    p.add_argument("--no-demo", action="store_true",
                   help="Start nodes only — don't send any query")
    args = p.parse_args()

    # Default model per backend
    if args.model is None:
        args.model = {
            "ollama":  "llama3.2",
            "vllm":    "meta-llama/Llama-3.2-3B-Instruct",
            "ravnest": "meta-llama/Llama-3.2-8B-Instruct",
        }[args.backend]

    # Pre-flight checks
    if not DB_PATH.exists():
        print(f"[run] Database not found: {DB_PATH}")
        print(f"[run]   Run:  python3 {HERE}/setup_db.py")
        sys.exit(1)

    print("=" * 60)
    print(" Ravnest Distributed SQL Agent — network startup")
    print("=" * 60)
    print(f"  backend : {args.backend}")
    print(f"  model   : {args.model}")
    print(f"  db      : {DB_PATH}")
    print()

    procs: list[subprocess.Popen] = []

    def _shutdown(sig=None, frame=None):
        print("\n[run] Shutting down nodes…")
        for proc in procs:
            proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        print("[run] Done.")
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Start compute node (backend-specific)
    print(f"[run] Starting compute node (backend={args.backend}, port=8765)…")
    compute_proc = start_compute_node(
        backend              = args.backend,
        model                = args.model,
        port                 = 8765,
        host                 = "0.0.0.0",
        num_nodes            = args.num_nodes,
        ravnest_master_addr  = args.ravnest_master_addr,
        ravnest_master_port  = args.ravnest_master_port,
    )
    procs.append(compute_proc)
    time.sleep(0.3)

    # Start SQL tool and agent nodes
    for node in SQL_NODES:
        print(f"[run] Starting {node['name']} node (port {node['port']})…")
        proc = start_node(node, args.model)
        procs.append(proc)
        time.sleep(0.3)

    all_nodes = [
        {"name": "compute", "health": "http://localhost:8765/health",
         "timeout": 120.0 if args.backend in ("vllm", "ravnest") else 30.0},
        {"name": "sql-tool", "health": "http://localhost:8766/health", "timeout": 30.0},
        {"name": "agent",    "health": "http://localhost:8767/health", "timeout": 30.0},
    ]

    # Wait for all three to be healthy
    print()
    print("[run] Waiting for nodes to be healthy…")
    if args.backend in ("vllm", "ravnest"):
        print("[run]   (model loading may take a minute…)")
    all_healthy = True
    for node in all_nodes:
        ok = wait_healthy(node["health"], node["name"], timeout=node["timeout"])
        status = "OK" if ok else "FAILED"
        print(f"  {node['name']:10s}  {node['health']}  [{status}]")
        if not ok:
            all_healthy = False

    if not all_healthy:
        print("[run] One or more nodes failed to start. Check logs above.")
        _shutdown()

    print()
    print("[run] Network is up.")
    print(f"  Compute node ({args.backend}) : http://localhost:8765")
    print(f"  SQL tool node                : http://localhost:8766")
    print(f"  Agent node                   : http://localhost:8767")
    print()

    if args.no_demo:
        print("[run] Running in server mode. Press Ctrl-C to stop.")
        signal.pause()
        return

    # Send demo queries
    questions = [args.query] if args.query else DEMO_QUESTIONS
    for q in questions:
        print(f"Q: {q}")
        print("   (thinking…)")
        t0  = time.time()
        try:
            ans = send_query(q)
            elapsed = time.time() - t0
            print(f"A: {ans}")
            print(f"   [{elapsed:.1f}s]")
        except Exception as exc:
            print(f"   ERROR: {exc}")
        print()

    print("[run] Demo complete. Press Ctrl-C to stop the nodes.")
    try:
        signal.pause()
    except AttributeError:
        # signal.pause() not available on Windows
        while True:
            time.sleep(1)


if __name__ == "__main__":
    main()
