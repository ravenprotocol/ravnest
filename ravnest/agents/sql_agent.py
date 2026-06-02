"""
SQLAgent — natural-language-to-SQL agent.

The agent:
  1. Introspects the database schema (table names + column definitions).
  2. Generates a SQL query from the user's question.
  3. Executes it via SQLAlchemy.
  4. If the query fails, the agent self-corrects up to ``max_steps`` times.
  5. Returns a plain-English answer with the query results included.

Install:  pip install sqlalchemy litellm
          # plus your DB driver, e.g. pip install psycopg2-binary / pymysql

Usage
-----
    from ravnest.agents.sql_agent import SQLAgent
    from ravnest.agents.base import AgentRequest, Message

    agent = SQLAgent(
        model      = "gpt-4o-mini",
        db_url     = "sqlite:///northwind.db",
    )
    resp = agent.run(AgentRequest(
        messages = [Message("user", "How many orders were placed in 2023?")],
    ))
    print(resp.text)

    # Restrict which tables the agent can see
    agent = SQLAgent(
        model      = "gpt-4o-mini",
        db_url     = "postgresql+psycopg2://user:pw@host/db",
        allowed_tables = ["orders", "customers"],
    )
"""

from __future__ import annotations

import json
import re
import socket
import time
from typing import AsyncIterator, Dict, List, Optional, Any

from .base import (
    AgentBackend, AgentCapability, AgentHealthStatus, AgentRequest,
    AgentResponse, Message,
)

_SA_ERR     = "sqlalchemy is not installed. Run: pip install sqlalchemy"
_LITELLM_ERR = "litellm is not installed. Run: pip install litellm"

_SYSTEM_PROMPT_TMPL = """\
You are a SQL expert assistant. Given a database schema and a natural-language
question, generate a valid SQL query to answer it.

Rules:
- Use ONLY the tables and columns listed in the schema below.
- Write SELECT queries only (no INSERT / UPDATE / DELETE / DROP).
- Return ONLY the SQL query — no explanation, no markdown fences.
- If the question cannot be answered from the schema, reply with:
  NO_SQL: <reason>

Schema:
{schema}
"""

_ANSWER_PROMPT_TMPL = """\
The user asked: {question}

The SQL query executed was:
{query}

The result was:
{result}

Provide a clear, concise natural-language answer based on the result.
"""


class SQLAgent(AgentBackend):
    """
    Natural-language-to-SQL agent backed by any LiteLLM model.

    Args:
        model:          LiteLLM model string.
        db_url:         SQLAlchemy database URL (e.g. "sqlite:///mydb.db").
        allowed_tables: Whitelist of table names visible to the agent.
                        If None, all tables are shown.
        max_rows:       Maximum rows returned per query (default 100).
        api_base:       LiteLLM api_base override.
        api_key:        LiteLLM api_key override.
        node_id:        Registry node_id override.
    """

    def __init__(
        self,
        model:          str,
        db_url:         str,
        allowed_tables: Optional[List[str]] = None,
        max_rows:       int                 = 100,
        api_base:       Optional[str]       = None,
        api_key:        Optional[str]       = None,
        node_id:        Optional[str]       = None,
    ):
        try:
            import sqlalchemy  # noqa: F401
        except ImportError:
            raise ImportError(_SA_ERR)
        try:
            import litellm  # noqa: F401
        except ImportError:
            raise ImportError(_LITELLM_ERR)

        self._model          = model
        self._db_url         = db_url
        self._allowed_tables = allowed_tables
        self._max_rows       = max_rows
        self._api_base       = api_base
        self._api_key        = api_key
        self._node_id        = node_id or f"sql_{socket.gethostname()}"

        # Lazily populated schema string
        self._schema_cache: Optional[str] = None

    # ── async interface ───────────────────────────────────────────────────

    async def arun(self, request: AgentRequest) -> AgentResponse:
        t0      = time.perf_counter()
        model   = request.model or self._model
        question = request.last_user_message() or ""
        steps   = 0
        total_usage: Dict[str, int] = {"prompt_tokens": 0,
                                       "completion_tokens": 0, "total_tokens": 0}

        schema = await self._get_schema()

        # ── Step 1: generate SQL ──────────────────────────────────────────
        system_msg  = _SYSTEM_PROMPT_TMPL.format(schema=schema)
        llm_msgs = [
            {"role": "system",  "content": system_msg},
            {"role": "user",    "content": question},
        ]

        last_error: Optional[str] = None
        sql: Optional[str]        = None

        for attempt in range(request.max_steps):
            steps += 1
            if last_error and sql:
                # Self-correction: feed the error back
                llm_msgs.append({"role": "assistant", "content": sql})
                llm_msgs.append({
                    "role": "user",
                    "content": (
                        f"That query produced an error:\n{last_error}\n\n"
                        "Please fix the SQL and try again. Return ONLY the corrected SQL."
                    ),
                })

            resp = await self._llm_complete(llm_msgs, model, request.max_tokens,
                                            request.temperature)
            _accum_usage(total_usage, resp.get("usage"))
            sql = (resp.get("content") or "").strip()

            # Agent decided it can't answer
            if sql.upper().startswith("NO_SQL:"):
                reason = sql[7:].strip()
                return AgentResponse(
                    text          = f"Cannot answer from the database: {reason}",
                    agent         = "sql",
                    model         = model,
                    request_id    = request.request_id,
                    finish_reason = "no_sql",
                    steps         = steps,
                    usage         = total_usage,
                    latency_ms    = (time.perf_counter() - t0) * 1000,
                )

            # Strip markdown code fences if model wrapped the query
            sql = _strip_sql_fences(sql)

            # ── Step 2: execute SQL ───────────────────────────────────────
            result, exec_error = await self._execute_sql(sql)
            if exec_error is None:
                break
            last_error = exec_error

        if last_error and sql:
            return AgentResponse(
                text          = f"Failed to execute SQL after {steps} attempt(s).\n"
                                f"Last query: {sql}\nError: {last_error}",
                agent         = "sql",
                model         = model,
                request_id    = request.request_id,
                finish_reason = "error",
                steps         = steps,
                usage         = total_usage,
                latency_ms    = (time.perf_counter() - t0) * 1000,
            )

        # ── Step 3: synthesise a natural-language answer ──────────────────
        steps += 1
        answer_prompt = _ANSWER_PROMPT_TMPL.format(
            question = question,
            query    = sql,
            result   = result,
        )
        answer_resp = await self._llm_complete(
            [{"role": "user", "content": answer_prompt}],
            model, request.max_tokens, request.temperature,
        )
        _accum_usage(total_usage, answer_resp.get("usage"))
        answer = (answer_resp.get("content") or "").strip()

        return AgentResponse(
            text          = answer,
            agent         = "sql",
            model         = model,
            request_id    = request.request_id,
            finish_reason = "stop",
            steps         = steps,
            usage         = total_usage,
            latency_ms    = (time.perf_counter() - t0) * 1000,
            metadata      = {"sql": sql, "raw_result": result},
        )

    async def astream(self, request: AgentRequest) -> AsyncIterator[str]:
        """Non-streaming fallback — yields the final answer as a single chunk."""
        resp = await self.arun(request)
        if resp.text:
            yield resp.text

    async def ahealth(self) -> AgentHealthStatus:
        try:
            schema = await self._get_schema()
            tables = len(schema.splitlines())
            return AgentHealthStatus(
                healthy = True,
                agent   = "sql",
                model   = self._model,
                message = f"DB connected. Schema lines: {tables}",
            )
        except Exception as exc:
            return AgentHealthStatus(
                healthy = False,
                agent   = "sql",
                model   = self._model,
                message = str(exc),
            )

    def capabilities(self) -> AgentCapability:
        return AgentCapability(
            agent_type         = "sql",
            models             = [self._model],
            tools              = ["sql_query"],
            supports_streaming = False,
            node_id            = self._node_id,
            extra              = {
                "db_url":  _mask_db_url(self._db_url),
                "address": f"{socket.gethostname()}:0",
            },
        )

    # ── private helpers ───────────────────────────────────────────────────

    async def _get_schema(self) -> str:
        """Return (cached) CREATE-TABLE-style schema string."""
        if self._schema_cache is not None:
            return self._schema_cache

        import asyncio
        loop = asyncio.get_event_loop()
        self._schema_cache = await loop.run_in_executor(None, self._introspect_schema)
        return self._schema_cache

    def _introspect_schema(self) -> str:
        """Synchronously introspect the DB using SQLAlchemy reflection."""
        from sqlalchemy import create_engine, inspect

        engine  = create_engine(self._db_url)
        insp    = inspect(engine)
        tables  = insp.get_table_names()
        if self._allowed_tables:
            tables = [t for t in tables if t in self._allowed_tables]

        lines = []
        for table in tables:
            cols = insp.get_columns(table)
            col_defs = ", ".join(
                f"{c['name']} {c['type']}" for c in cols
            )
            lines.append(f"CREATE TABLE {table} ({col_defs});")
        engine.dispose()
        return "\n".join(lines)

    async def _execute_sql(self, sql: str):
        """Execute a SQL query, return (result_str, error_str)."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._run_sql, sql)

    def _run_sql(self, sql: str):
        from sqlalchemy import create_engine, text as sa_text

        engine = create_engine(self._db_url)
        try:
            with engine.connect() as conn:
                result = conn.execute(sa_text(sql))
                rows   = result.fetchmany(self._max_rows)
                cols   = list(result.keys())
                lines  = [", ".join(cols)]
                for row in rows:
                    lines.append(", ".join(str(v) for v in row))
                return "\n".join(lines), None
        except Exception as exc:
            return None, str(exc)
        finally:
            engine.dispose()

    async def _llm_complete(self, messages, model, max_tokens, temperature) -> dict:
        import litellm
        kwargs: dict = dict(
            model       = model,
            messages    = messages,
            max_tokens  = max_tokens,
            temperature = temperature,
        )
        if self._api_base:
            kwargs["api_base"] = self._api_base
        if self._api_key:
            kwargs["api_key"]  = self._api_key
        resp = await litellm.acompletion(**kwargs)
        return {
            "content": resp.choices[0].message.content,
            "usage":   resp.usage,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _strip_sql_fences(text: str) -> str:
    """Remove markdown ```sql ... ``` fences if present."""
    text = text.strip()
    # Remove opening fence
    text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
    # Remove closing fence
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def _mask_db_url(url: str) -> str:
    """Replace password in DB URL with *** for safe logging."""
    return re.sub(r"(:)[^:@]+(@)", r"\1***\2", url)


def _accum_usage(total: dict, usage) -> None:
    if usage is None:
        return
    total["prompt_tokens"]     += getattr(usage, "prompt_tokens",     0) or 0
    total["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
    total["total_tokens"]      += getattr(usage, "total_tokens",      0) or 0
