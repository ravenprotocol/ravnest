"""
GraphDBSource — graph database node for Ravnest.

Supports two backends (auto-detected):
  1. Neo4j     — ``pip install neo4j``   — production, persistent, Cypher queries
  2. NetworkX  — ``pip install networkx`` — in-memory, great for development

Query interface:
  - Free-text Cypher (Neo4j)
  - Keyword / node-label search  → subgraph as JSON triples
  - Path queries: find all paths between two nodes

Usage
-----
    from ravnest.data_sources.graph_db import GraphDBSource
    from ravnest.data_sources.base import DataRequest

    # NetworkX in-memory
    source = GraphDBSource(backend="networkx")
    source.add_triples([
        ("Alice", "KNOWS", "Bob"),
        ("Bob",   "WORKS_AT", "Ravnest"),
        ("Alice", "WORKS_AT", "Ravnest"),
    ])
    resp = source.query(DataRequest(query="Alice", top_k=5))
    for chunk in resp.chunks:
        print(chunk.content)   # JSON triple

    # Neo4j
    source = GraphDBSource(
        backend  = "neo4j",
        neo4j_uri= "bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
    )
    resp = source.query(DataRequest(
        query = "MATCH (n:Person)-[:KNOWS]->(m) RETURN n.name, m.name LIMIT 5",
    ))
"""

from __future__ import annotations

import json
import socket
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from .base import (
    DataChunk, DataRequest, DataResponse, DataSourceBackend,
    DataSourceCapability, DataSourceHealthStatus,
)


class GraphDBSource(DataSourceBackend):
    """
    DataSourceBackend for graph data (triples, paths, neighbourhoods).

    Args:
        backend:         "auto" | "neo4j" | "networkx".
        neo4j_uri:       Bolt URI for Neo4j (default "bolt://localhost:7687").
        neo4j_user:      Neo4j username.
        neo4j_password:  Neo4j password.
        neo4j_database:  Neo4j database name (default "neo4j").
        node_id:         Registry node_id override.
    """

    def __init__(
        self,
        backend:        str           = "auto",
        neo4j_uri:      str           = "bolt://localhost:7687",
        neo4j_user:     str           = "neo4j",
        neo4j_password: str           = "",
        neo4j_database: str           = "neo4j",
        node_id:        Optional[str] = None,
    ):
        self._backend_name   = backend
        self._neo4j_uri      = neo4j_uri
        self._neo4j_user     = neo4j_user
        self._neo4j_password = neo4j_password
        self._neo4j_database = neo4j_database
        self._node_id        = node_id or f"graphdb_{socket.gethostname()}"

        self._graph:            Optional[object] = None
        self._resolved_backend: str              = ""

    # ── async interface ───────────────────────────────────────────────────

    async def aquery(self, request: DataRequest) -> DataResponse:
        import asyncio
        t0     = time.perf_counter()
        loop   = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(None, self._sync_query, request)
        return DataResponse(
            chunks      = chunks,
            source      = f"graph_db/{self._resolved_backend}",
            request_id  = request.request_id,
            total_found = len(chunks),
            latency_ms  = (time.perf_counter() - t0) * 1000,
        )

    async def astream(self, request: DataRequest) -> AsyncIterator[DataChunk]:
        import asyncio
        loop   = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(None, self._sync_query, request)
        for chunk in chunks:
            yield chunk

    async def ahealth(self) -> DataSourceHealthStatus:
        try:
            graph = self._get_graph()
            count = self._node_count(graph)
            return DataSourceHealthStatus(
                healthy    = True,
                source     = f"graph_db/{self._resolved_backend}",
                message    = f"Graph has {count} nodes",
                item_count = count,
            )
        except Exception as exc:
            return DataSourceHealthStatus(
                healthy = False,
                source  = f"graph_db/{self._backend_name}",
                message = str(exc),
            )

    def capabilities(self) -> DataSourceCapability:
        try:
            count = self._node_count(self._get_graph())
        except Exception:
            count = 0
        return DataSourceCapability(
            source_type = "graph_db",
            modalities  = ["graph", "text"],
            item_count  = count,
            node_id     = self._node_id,
            extra       = {
                "backend": self._resolved_backend or self._backend_name,
                "address": f"{socket.gethostname()}:0",
            },
        )

    # ── public data ingestion ─────────────────────────────────────────────

    def add_triples(
        self,
        triples:       List[Tuple[str, str, str]],
        node_attrs:    Optional[Dict[str, Dict]] = None,
        edge_attrs:    Optional[Dict[Tuple, Dict]] = None,
    ) -> None:
        """
        Add (subject, predicate, object) triples to the graph.

        Args:
            triples:    List of (src, rel, dst) tuples.
            node_attrs: Optional dict {node_name: {attr: val, …}}.
            edge_attrs: Optional dict {(src, dst): {attr: val, …}}.
        """
        graph = self._get_graph()
        b     = self._resolved_backend
        if b == "networkx":
            for src, rel, dst in triples:
                attrs = {"relation": rel}
                if edge_attrs and (src, dst) in edge_attrs:
                    attrs.update(edge_attrs[(src, dst)])
                graph.add_edge(src, dst, **attrs)
                if node_attrs:
                    for node, attrs_ in node_attrs.items():
                        if graph.has_node(node):
                            graph.nodes[node].update(attrs_)
        elif b == "neo4j":
            with graph.session(database=self._neo4j_database) as session:
                for src, rel, dst in triples:
                    session.run(
                        f"MERGE (a {{name: $src}}) "
                        f"MERGE (b {{name: $dst}}) "
                        f"MERGE (a)-[:{_safe_rel(rel)}]->(b)",
                        src=src, dst=dst,
                    )

    def clear(self) -> None:
        """Clear all data (NetworkX only; Neo4j: delete nodes yourself)."""
        graph = self._get_graph()
        if self._resolved_backend == "networkx":
            graph.clear()

    # ── private ───────────────────────────────────────────────────────────

    def _sync_query(self, request: DataRequest) -> List[DataChunk]:
        graph = self._get_graph()
        query = request.query.strip()

        # Path query: "path:Alice->Bob"
        if query.lower().startswith("path:"):
            path_spec = query[5:].strip()
            return self._path_query(graph, path_spec, request.top_k)

        # Cypher (Neo4j) or keyword search
        if self._resolved_backend == "neo4j":
            return self._neo4j_query(graph, query, request.top_k)
        else:
            return self._networkx_query(graph, query, request.top_k,
                                        request.filters)

    def _get_graph(self):
        if self._graph is None:
            self._graph, self._resolved_backend = self._init_graph()
        return self._graph

    def _init_graph(self):
        b = self._backend_name

        if b in ("auto", "neo4j"):
            try:
                return self._init_neo4j(), "neo4j"
            except ImportError:
                if b == "neo4j":
                    raise

        if b in ("auto", "networkx"):
            try:
                return self._init_networkx(), "networkx"
            except ImportError:
                raise ImportError(
                    "networkx is not installed. Run: pip install networkx"
                )

        raise ValueError(f"Unsupported graph backend: {b}")

    def _init_neo4j(self):
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            self._neo4j_uri,
            auth=(self._neo4j_user, self._neo4j_password),
        )
        driver.verify_connectivity()
        return driver

    def _init_networkx(self):
        import networkx as nx
        return nx.MultiDiGraph()

    # ── NetworkX queries ──────────────────────────────────────────────────

    def _networkx_query(self, graph, query: str, top_k: int,
                        filters: Dict) -> List[DataChunk]:
        import networkx as nx

        # Keyword match against node names
        query_lower = query.lower()
        if query_lower:
            matching = [n for n in graph.nodes
                        if query_lower in str(n).lower()]
        else:
            matching = list(graph.nodes)

        # Apply node attribute filters
        if filters:
            matching = [n for n in matching
                        if all(graph.nodes[n].get(k) == v
                               for k, v in filters.items())]

        # Return neighbourhood triples for each matched node
        chunks = []
        seen   = set()
        for node in matching[:top_k]:
            for src, dst, data in graph.out_edges(node, data=True):
                triple = (str(src), data.get("relation", "RELATED_TO"), str(dst))
                key    = triple
                if key not in seen:
                    seen.add(key)
                    chunks.append(_triple_to_chunk(triple, score=1.0,
                                                   source=self._resolved_backend))
            if not graph.out_edges(node):
                # Isolated node
                chunks.append(DataChunk(
                    content  = json.dumps({"node": str(node),
                                           "attrs": dict(graph.nodes[node])}),
                    modality = "graph",
                    score    = 1.0,
                    source   = self._resolved_backend,
                    metadata = {"type": "node"},
                ))
        return chunks[:top_k]

    def _path_query(self, graph, spec: str, top_k: int) -> List[DataChunk]:
        """Handle 'path:Alice->Bob' style queries."""
        parts = [p.strip() for p in spec.split("->")]
        if len(parts) != 2 or self._resolved_backend != "networkx":
            return []
        src, dst = parts[0], parts[1]
        import networkx as nx
        try:
            # Find all simple paths up to length 6
            paths = list(nx.all_simple_paths(graph, src, dst, cutoff=6))
            chunks = []
            for path in paths[:top_k]:
                chunks.append(DataChunk(
                    content  = json.dumps({"path": path}),
                    modality = "graph",
                    score    = 1.0 / len(path),   # shorter = better
                    source   = self._resolved_backend,
                    metadata = {"src": src, "dst": dst, "length": len(path)},
                ))
            return sorted(chunks, key=lambda c: -c.score)
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            return []

    # ── Neo4j queries ─────────────────────────────────────────────────────

    def _neo4j_query(self, driver, cypher: str, top_k: int) -> List[DataChunk]:
        # If it looks like a raw Cypher query, run it; else do keyword search
        if not _looks_like_cypher(cypher):
            cypher = (
                f"MATCH (n)-[r]->(m) "
                f"WHERE n.name CONTAINS $q OR m.name CONTAINS $q "
                f"RETURN n.name AS src, type(r) AS rel, m.name AS dst "
                f"LIMIT {top_k}"
            )
            params = {"q": cypher}  # cypher is the original keyword here
        else:
            params = {}

        with driver.session(database=self._neo4j_database) as session:
            result = session.run(cypher, **params)
            chunks = []
            for record in result:
                data = dict(record)
                # Try to extract (src, rel, dst) triple pattern
                if "src" in data and "rel" in data and "dst" in data:
                    triple = (str(data["src"]), str(data["rel"]), str(data["dst"]))
                    chunks.append(_triple_to_chunk(triple, 1.0, "neo4j"))
                else:
                    chunks.append(DataChunk(
                        content  = json.dumps(data),
                        modality = "graph",
                        score    = 1.0,
                        source   = "neo4j",
                        metadata = data,
                    ))
            return chunks[:top_k]

    def _node_count(self, graph) -> int:
        if self._resolved_backend == "networkx":
            return graph.number_of_nodes()
        elif self._resolved_backend == "neo4j":
            with graph.session(database=self._neo4j_database) as sess:
                result = sess.run("MATCH (n) RETURN count(n) AS c")
                return result.single()["c"]
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _triple_to_chunk(triple: Tuple[str, str, str], score: float,
                     source: str) -> DataChunk:
    src, rel, dst = triple
    return DataChunk(
        content  = json.dumps({"subject": src, "predicate": rel, "object": dst}),
        modality = "graph",
        score    = score,
        source   = source,
        metadata = {"subject": src, "predicate": rel, "object": dst},
    )


def _safe_rel(rel: str) -> str:
    """Make a relation name safe for use as a Neo4j relationship type."""
    import re
    return re.sub(r"[^A-Za-z0-9_]", "_", rel).upper()


def _looks_like_cypher(text: str) -> bool:
    """Heuristic: does the text look like a Cypher statement?"""
    keywords = ("match", "return", "create", "merge", "where", "with",
                 "call", "unwind")
    lower = text.lower().strip()
    return any(lower.startswith(k) for k in keywords)
