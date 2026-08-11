"""
node_agent.py — Distributed SQL agent node (port 8767).

Implements a ReAct agent that:
  1. Receives a natural-language question.
  2. Sends it to the compute node (Llama) for reasoning.
  3. Parses the model response for SQL actions.
  4. Sends SQL to the SQL tool node for execution.
  5. Feeds results back to the model for the next step.
  6. Returns the final answer.

Every LLM call and every SQL execution crosses the mesh via NodeClient —
no model weights or database files live on this process.

Prompt format
-------------
The agent uses a structured action format that works reliably with
small instruction-tuned models:

    ACTION: sql
    QUERY: SELECT ...

    ANSWER: <final natural-language answer>

Usage
-----
    python3 node_agent.py \\
        [--port 8767] \\
        [--compute-url http://localhost:8765] \\
        [--sql-url http://localhost:8766] \\
        [--model llama3.2] \\
        [--max-steps 6]
"""

import argparse
import pathlib
import re
import sqlite3
import sys
import time
import uuid
from typing import AsyncIterator, Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from ravnest.mesh.node_client  import NodeClient
from ravnest.mesh.base         import NodeMessage
from ravnest.mesh.node_server  import NodeServer
from ravnest.agents.base       import (
    AgentBackend, AgentRequest, AgentResponse,
    AgentCapability, AgentHealthStatus,
)


_SYSTEM_PROMPT = """\
You are a data analyst with read-only access to an e-commerce SQLite database.

Database schema:
{schema}

To query the database, output EXACTLY:
ACTION: sql
QUERY: <your single-line SQL here>

When you have enough information to answer the user, output EXACTLY:
ANSWER: <your concise answer here>

Rules:
- Write valid SQLite SQL (single statement, no semicolons at the end).
- Use only one ACTION or one ANSWER per response — never both.
- After receiving SQL results, decide whether to run another query or answer.
- Be concise in the ANSWER.
"""


class DistributedSQLAgent(AgentBackend):
    """
    ReAct agent whose two tools (LLM + SQL) run on separate mesh nodes.

    Args:
        compute_url:  HTTP address of the compute node (Llama).
        sql_url:      HTTP address of the SQL tool node.
        db_schema:    Schema string injected into the system prompt.
        model:        Model name forwarded to the compute node.
        max_steps:    Maximum ReAct iterations per request.
        node_id:      Override auto-generated node id.
        timeout:      Per-request timeout in seconds.
    """

    def __init__(
        self,
        compute_url: str,
        sql_url:     str,
        db_schema:   str   = "",
        model:       str   = "llama3.2",
        max_steps:   int   = 6,
        node_id:     Optional[str] = None,
        timeout:     float = 60.0,
    ):
        self._compute   = NodeClient(compute_url, timeout=timeout)
        self._sql       = NodeClient(sql_url,     timeout=timeout)
        self._schema    = db_schema
        self._model     = model
        self._max_steps = max_steps
        self._node_id   = node_id or f"sql-agent-{uuid.uuid4().hex[:8]}"
        self._timeout   = timeout

    # ── AgentBackend interface ─────────────────────────────────────────────

    async def arun(self, request: AgentRequest) -> AgentResponse:
        t0      = time.perf_counter()
        model   = request.model or self._model
        steps   = request.max_steps or self._max_steps
        history = self._build_history(request)

        for step in range(steps):
            llm_text = await self._llm_call(history, model,
                                            request.max_tokens,
                                            request.temperature)
            history.append({"role": "assistant", "content": llm_text})

            if self._is_sql_action(llm_text):
                sql    = self._extract_sql(llm_text)
                result = await self._sql_call(sql) if sql else "Error: could not parse SQL"
                history.append({
                    "role":    "user",
                    "content": f"SQL Result:\n{result}",
                })
                continue

            if "ANSWER:" in llm_text:
                return AgentResponse(
                    text          = self._extract_answer(llm_text),
                    agent         = "distributed-sql",
                    model         = model,
                    steps         = step + 1,
                    finish_reason = "stop",
                    latency_ms    = (time.perf_counter() - t0) * 1000,
                )

        # Max steps reached — return last assistant message as best effort
        last_text = next(
            (m["content"] for m in reversed(history) if m["role"] == "assistant"),
            "Max steps reached without a final answer.",
        )
        return AgentResponse(
            text          = last_text,
            agent         = "distributed-sql",
            model         = model,
            steps         = steps,
            finish_reason = "max_steps",
            latency_ms    = (time.perf_counter() - t0) * 1000,
        )

    async def astream(self, request: AgentRequest) -> AsyncIterator[str]:
        resp = await self.arun(request)
        yield resp.text

    async def ahealth(self) -> AgentHealthStatus:
        try:
            msg = NodeMessage(node_type="compute", action="health", payload={})
            r   = await self._compute.asend(msg)
            healthy = r.ok
            message = r.result.get("message", "") if r.ok else r.error
        except Exception as exc:
            healthy = False
            message = str(exc)
        return AgentHealthStatus(
            healthy = healthy,
            agent   = "distributed-sql",
            model   = self._model,
            message = message,
        )

    def capabilities(self) -> AgentCapability:
        return AgentCapability(
            node_id    = self._node_id,
            agent_type = "distributed-sql",
            models     = [self._model],
            tools      = ["sql_query"],
        )

    # ── mesh calls ─────────────────────────────────────────────────────────

    async def _llm_call(self, messages: list, model: str,
                        max_tokens: int, temperature: float) -> str:
        """Send the conversation to the compute node and return the reply."""
        msg  = NodeMessage(
            node_type = "compute",
            action    = "generate",
            model     = model,
            payload   = {
                "messages":    messages,
                "max_tokens":  max_tokens,
                "temperature": temperature,
            },
        )
        resp = await self._compute.asend(msg)
        if not resp.ok:
            raise RuntimeError(f"Compute node error: {resp.error}")
        return resp.result.get("text", "")

    async def _sql_call(self, sql: str) -> str:
        """Send SQL to the SQL tool node and return a formatted result string."""
        msg  = NodeMessage(
            node_type = "data_source",
            action    = "query",
            payload   = {"query": sql, "top_k": 500},
        )
        resp = await self._sql.asend(msg)
        if not resp.ok:
            return f"SQL node error: {resp.error}"

        chunks = resp.result.get("chunks", [])
        if not chunks:
            return "Query returned 0 rows."

        # First chunk might be an error
        if chunks[0].get("metadata", {}).get("error"):
            return chunks[0]["content"]

        rows = [c["content"] for c in chunks]
        header = f"({len(rows)} row{'s' if len(rows) != 1 else ''})"
        return header + "\n" + "\n".join(rows)

    # ── helpers ────────────────────────────────────────────────────────────

    def _build_history(self, request: AgentRequest) -> list:
        history = [{"role": "system",
                    "content": _SYSTEM_PROMPT.format(schema=self._schema)}]
        for m in request.messages:
            history.append({"role": m.role, "content": m.content})
        return history

    @staticmethod
    def _is_sql_action(text: str) -> bool:
        return "ACTION: sql" in text or "ACTION:sql" in text

    @staticmethod
    def _extract_sql(text: str) -> str:
        m = re.search(r"QUERY:\s*(.+?)(?:\n(?:ACTION|ANSWER)|$)", text,
                      re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _extract_answer(text: str) -> str:
        m = re.search(r"ANSWER:\s*(.+)", text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else text.strip()


# ── schema helper ─────────────────────────────────────────────────────────────

def _fetch_schema(db_path: str) -> str:
    """Return CREATE TABLE statements from a SQLite database."""
    try:
        conn   = sqlite3.connect(db_path)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        lines  = []
        for (name,) in tables:
            sql = conn.execute(
                "SELECT sql FROM sqlite_master WHERE name=?", (name,)
            ).fetchone()[0]
            lines.append(sql)
        conn.close()
        return "\n\n".join(lines)
    except Exception as exc:
        return f"(schema unavailable: {exc})"


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Ravnest distributed SQL agent node")
    p.add_argument("--port",        default=8767,                      type=int)
    p.add_argument("--host",        default="0.0.0.0")
    p.add_argument("--compute-url", default="http://localhost:8765")
    p.add_argument("--sql-url",     default="http://localhost:8766")
    p.add_argument("--model",       default="llama3.2")
    p.add_argument("--max-steps",   default=6,                         type=int)
    p.add_argument("--db",          default="./shop.db",
                   help="Path to SQLite DB used to read schema for the prompt")
    args = p.parse_args()

    schema = _fetch_schema(args.db)

    print(f"[agent] Starting distributed SQL agent node")
    print(f"[agent]   compute → {args.compute_url}")
    print(f"[agent]   sql     → {args.sql_url}")
    print(f"[agent]   model   = {args.model}")
    print(f"[agent]   endpoint = http://{args.host}:{args.port}")

    agent  = DistributedSQLAgent(
        compute_url = args.compute_url,
        sql_url     = args.sql_url,
        db_schema   = schema,
        model       = args.model,
        max_steps   = args.max_steps,
    )
    server = NodeServer(host=args.host, port=args.port)
    server.add_agent(agent)
    server.run()


if __name__ == "__main__":
    main()
