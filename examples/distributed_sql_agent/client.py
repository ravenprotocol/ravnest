"""
client.py — Send natural-language questions to the running agent network.

The network must already be up (run_network.py or start nodes manually).

Usage
-----
    # Single question
    python3 client.py "Which customer spent the most?"

    # Interactive REPL
    python3 client.py --repl

    # Point at a non-default agent URL
    python3 client.py --agent-url http://192.168.1.10:8767 "Total revenue?"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request


def ask(question: str, agent_url: str, timeout: float = 120.0) -> dict:
    """
    Send a question to the agent node.  Returns the raw result dict.
    """
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
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def print_response(data: dict, elapsed: float) -> None:
    if data.get("ok"):
        res   = data.get("result", {})
        text  = res.get("text", "(no answer)")
        steps = res.get("steps", "?")
        model = res.get("model", "")
        print(f"\nA: {text}")
        print(f"   [{elapsed:.1f}s  |  {steps} step(s)  |  {model}]")
    else:
        print(f"\nERROR: {data.get('error', 'unknown')}")


def repl(agent_url: str) -> None:
    print(f"Connected to agent at {agent_url}")
    print("Type your question and press Enter.  Ctrl-C or 'quit' to exit.\n")
    while True:
        try:
            q = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q or q.lower() in ("quit", "exit", "q"):
            break
        t0   = time.time()
        data = ask(q, agent_url)
        print_response(data, time.time() - t0)
        print()


def main():
    p = argparse.ArgumentParser(description="Ravnest distributed SQL agent client")
    p.add_argument("question",     nargs="?", default=None,
                   help="Natural-language question to ask")
    p.add_argument("--agent-url",  default="http://localhost:8767")
    p.add_argument("--repl",       action="store_true",
                   help="Start an interactive question REPL")
    p.add_argument("--timeout",    default=120.0, type=float)
    args = p.parse_args()

    if args.repl:
        repl(args.agent_url)
        return

    if not args.question:
        p.print_help()
        sys.exit(1)

    print(f"Q: {args.question}")
    print("   (thinking…)")
    t0   = time.time()
    try:
        data = ask(args.question, args.agent_url, timeout=args.timeout)
        print_response(data, time.time() - t0)
    except Exception as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
