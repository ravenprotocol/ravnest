"""
run_network.py — Launch the full distributed SQL agent network and run a demo.

Starts three processes:
    1. node_compute.py  (Llama via Ollama, port 8765)
    2. node_sql.py      (SQLite tool node,  port 8766)
    3. node_agent.py    (ReAct SQL agent,   port 8767)

Waits for all three to be healthy, then sends a demo question and prints
the answer.  Ctrl-C shuts everything down cleanly.

Prerequisites
-------------
    pip install aiohttp httpx ollama
    ollama pull llama3.2
    python3 setup_db.py          # create shop.db

Usage
-----
    python3 run_network.py
    python3 run_network.py --query "Which city has the most customers?"
    python3 run_network.py --model llama3.1 --query "List all product categories"
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

NODES = [
    {
        "name":    "compute",
        "script":  HERE / "node_compute.py",
        "port":    8765,
        "health":  "http://localhost:8765/health",
        "extra":   [],
    },
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


# ── process management ────────────────────────────────────────────────────────

def start_node(node: dict, model: str) -> subprocess.Popen:
    cmd = [
        sys.executable, str(node["script"]),
        "--port", str(node["port"]),
        *node["extra"],
    ]
    if node["name"] in ("compute", "agent"):
        cmd += ["--model", model]

    proc = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        bufsize = 1,
        text    = True,
    )
    return proc


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
    p.add_argument("--model",   default="llama3.2")
    p.add_argument("--query",   default=None,
                   help="Ask a custom question after startup (overrides demo questions)")
    p.add_argument("--no-demo", action="store_true",
                   help="Start nodes only — don't send any query")
    args = p.parse_args()

    # Pre-flight checks
    if not DB_PATH.exists():
        print(f"[run] Database not found: {DB_PATH}")
        print(f"[run]   Run:  python3 {HERE}/setup_db.py")
        sys.exit(1)

    print("=" * 60)
    print(" Ravnest Distributed SQL Agent — network startup")
    print("=" * 60)
    print(f"  model : {args.model}")
    print(f"  db    : {DB_PATH}")
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

    # Start nodes in order
    for node in NODES:
        print(f"[run] Starting {node['name']} node (port {node['port']})…")
        proc = start_node(node, args.model)
        procs.append(proc)
        time.sleep(0.3)   # slight stagger so logs are readable

    # Wait for all three to be healthy
    print()
    print("[run] Waiting for nodes to be healthy…")
    all_healthy = True
    for node in NODES:
        ok = wait_healthy(node["health"], node["name"], timeout=60.0)
        status = "OK" if ok else "FAILED"
        print(f"  {node['name']:10s}  {node['health']}  [{status}]")
        if not ok:
            all_healthy = False

    if not all_healthy:
        print("[run] One or more nodes failed to start. Check logs above.")
        _shutdown()

    print()
    print("[run] Network is up.")
    print(f"  Compute node : http://localhost:8765")
    print(f"  SQL tool node: http://localhost:8766")
    print(f"  Agent node   : http://localhost:8767")
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
