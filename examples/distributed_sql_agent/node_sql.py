"""
node_sql.py — SQL tool node (port 8766).

Exposes a SQLite database as a mesh data-source node.
The agent sends raw SQL; this node executes it and returns rows as DataChunks.

The NodeServer dispatch maps:
    NodeMessage(node_type="data_source", action="query", payload={"query": "SELECT ..."})
  → SQLToolBackend.aquery(DataRequest(query="SELECT ..."))
  → NodeResponse(result={"chunks": [{"content": "id: 1, name: Alice, ...", ...}], ...})

Usage
-----
    python3 node_sql.py [--port 8766] [--db ./shop.db] [--host 0.0.0.0]
"""

import argparse
import pathlib
import sqlite3
import sys
import time
import uuid
from typing import AsyncIterator, Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from ravnest.mesh.node_server      import NodeServer
from ravnest.data_sources.base     import (
    DataSourceBackend, DataRequest, DataResponse,
    DataChunk, DataSourceCapability, DataSourceHealthStatus,
)


class SQLToolBackend(DataSourceBackend):
    """
    DataSourceBackend that executes raw SQL against a SQLite database.

    DataRequest.query  → SQL string to execute
    DataResponse.chunks → one DataChunk per result row
                          chunk.content  = "col: val, col: val, ..."
                          chunk.metadata = {"col": val, ...}

    SQL errors are returned as a single chunk with score=0 and
    metadata.error set, so the agent can see what went wrong.
    """

    def __init__(self, db_path: str, node_id: Optional[str] = None):
        self._db_path = db_path
        self._node_id = node_id or f"sql-tool-{uuid.uuid4().hex[:8]}"

    async def aquery(self, request: DataRequest) -> DataResponse:
        t0  = time.perf_counter()
        sql = request.query.strip()

        chunks: list[DataChunk] = []
        try:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            cur  = conn.execute(sql)
            rows = cur.fetchall()
            conn.close()

            for i, row in enumerate(rows):
                row_dict = dict(row)
                content  = ", ".join(f"{k}: {v}" for k, v in row_dict.items())
                chunks.append(DataChunk(
                    content  = content,
                    chunk_id = str(i),
                    modality = "text",
                    score    = 1.0,
                    source   = self._db_path,
                    metadata = row_dict,
                ))
        except Exception as exc:
            chunks.append(DataChunk(
                content  = f"SQL Error: {exc}",
                chunk_id = "error",
                modality = "text",
                score    = 0.0,
                source   = self._db_path,
                metadata = {"error": str(exc), "sql": sql},
            ))

        return DataResponse(
            chunks      = chunks,
            source      = self._db_path,
            request_id  = request.request_id,
            total_found = len(chunks),
            latency_ms  = (time.perf_counter() - t0) * 1000,
            metadata    = {"sql": sql},
        )

    async def astream(self, request: DataRequest) -> AsyncIterator[DataChunk]:
        resp = await self.aquery(request)
        for chunk in resp.chunks:
            yield chunk

    async def ahealth(self) -> DataSourceHealthStatus:
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute("SELECT 1")
            tables = conn.execute(
                "SELECT count(*) FROM sqlite_master WHERE type='table'"
            ).fetchone()[0]
            conn.close()
            return DataSourceHealthStatus(
                healthy    = True,
                source     = "sql",
                message    = f"{tables} tables",
                item_count = tables,
            )
        except Exception as exc:
            return DataSourceHealthStatus(
                healthy = False,
                source  = "sql",
                message = str(exc),
            )

    def capabilities(self) -> DataSourceCapability:
        return DataSourceCapability(
            node_id     = self._node_id,
            source_type = "sql",
            modalities  = ["text"],
            item_count  = 0,
            extra       = {"db_path": self._db_path, "dialect": "sqlite"},
        )


def main():
    p = argparse.ArgumentParser(description="Ravnest SQL tool node")
    p.add_argument("--port", default=8766,       type=int)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--db",   default="./shop.db")
    args = p.parse_args()

    if not pathlib.Path(args.db).exists():
        print(f"[sql-tool] ERROR: database not found: {args.db}")
        print(f"[sql-tool]   Run:  python3 setup_db.py --db {args.db}")
        sys.exit(1)

    print(f"[sql-tool] Starting SQL tool node")
    print(f"[sql-tool]   db       = {args.db}")
    print(f"[sql-tool]   endpoint = http://{args.host}:{args.port}")

    backend = SQLToolBackend(db_path=args.db)
    server  = NodeServer(host=args.host, port=args.port)
    server.add_data_source(backend)
    server.run()


if __name__ == "__main__":
    main()
