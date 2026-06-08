"""
examples/data_sources/demo.py
==============================
Smoke test for the Phase-4 Data Source Node framework.

What this covers
----------------
1.  DataChunk / DataRequest / DataResponse / DataSourceCapability dataclasses
2.  TextSource   — in-memory BM25 ranking via add_text()
3.  VectorDBSource (NumPy backend) — cosine similarity with raw vectors
4.  GraphDBSource (NetworkX backend) — triple ingestion + keyword + path query
5.  ImageSource  — scan a temp directory of placeholder files
6.  DataRouter   — offline mode via add_local_backend()
7.  LoadBasedDataStrategy / RoundRobinDataStrategy / SourceTypeStrategy
8.  Async streaming via astream()

Running
-------
# Offline tests (no external services needed)
python examples/data_sources/demo.py

# With a live registry
python examples/data_sources/demo.py --registry 127.0.0.1:50099
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
import time
import threading
import unittest.mock as mock
from typing import List

import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from ravnest.data_sources.base import (
    DataChunk, DataRequest, DataResponse,
    DataSourceBackend, DataSourceCapability, DataSourceHealthStatus,
)
from ravnest.data_sources.text_source  import TextSource
from ravnest.data_sources.vector_db    import VectorDBSource
from ravnest.data_sources.graph_db     import GraphDBSource
from ravnest.data_sources.image_source import ImageSource
from ravnest.data_sources.router import (
    DataRouter, LoadBasedDataStrategy, RoundRobinDataStrategy, SourceTypeStrategy,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sep(title: str = "") -> None:
    width = 64
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─' * pad} {title} {'─' * pad}")
    else:
        print("─" * width)


def _ok(msg: str) -> None:   print(f"  ✓  {msg}")
def _skip(msg: str) -> None: print(f"  ○  SKIP  {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data-class construction
# ─────────────────────────────────────────────────────────────────────────────

def test_dataclasses() -> None:
    _sep("Data classes")

    chunk = DataChunk(content="hello world", modality="text", score=0.9,
                      source="test.txt")
    assert chunk.content == "hello world" and chunk.score == 0.9
    _ok("DataChunk")

    req = DataRequest(query="distributed training", top_k=5)
    assert req.top_k == 5 and req.request_id  # auto UUID
    _ok("DataRequest — auto request_id")

    resp = DataResponse(
        chunks     = [chunk],
        source     = "text",
        request_id = req.request_id,
        latency_ms = 12.3,
    )
    assert resp.top().content == "hello world"
    assert resp.texts() == ["hello world"]
    _ok("DataResponse — top() and texts()")

    cap = DataSourceCapability(source_type="text", modalities=["text"],
                               node_id="n1", item_count=42)
    assert cap.item_count == 42
    _ok("DataSourceCapability")

    hs = DataSourceHealthStatus(healthy=True, source="text", item_count=10)
    assert hs.healthy
    _ok("DataSourceHealthStatus")


# ─────────────────────────────────────────────────────────────────────────────
# 2. TextSource — in-memory BM25 ranking
# ─────────────────────────────────────────────────────────────────────────────

def test_text_source() -> None:
    _sep("TextSource — BM25 ranking")

    source = TextSource(paths=[], node_id="test-text")

    # Populate via add_text()
    source.add_text(
        "Ravnest is a distributed framework for training and inference of "
        "large language models across heterogeneous nodes.",
        source="ravnest_docs.txt",
    )
    source.add_text(
        "vLLM provides high-throughput and memory-efficient inference for LLMs "
        "using PagedAttention and continuous batching.",
        source="vllm_docs.txt",
    )
    source.add_text(
        "SQLAlchemy is a Python SQL toolkit and Object-Relational Mapper.",
        source="sqlalchemy_docs.txt",
    )

    resp = source.query(DataRequest(query="distributed inference", top_k=2))
    assert len(resp.chunks) > 0
    assert resp.chunks[0].score > 0
    _ok(f"BM25 top result: score={resp.chunks[0].score}  "
        f"src={resp.chunks[0].source!r}")

    # No-query listing
    resp2 = source.query(DataRequest(query="", top_k=10))
    assert len(resp2.chunks) >= 1
    _ok(f"No-query listing returns {len(resp2.chunks)} chunk(s)")

    # Health check
    hs = source.health()
    assert hs.healthy
    _ok(f"health() → {hs.message}")

    # Capabilities
    cap = source.capabilities()
    assert cap.source_type == "text"
    _ok(f"capabilities() → source_type={cap.source_type}, items={cap.item_count}")


def test_text_source_file() -> None:
    _sep("TextSource — file scanning")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write two text files
        for name, content in [
            ("alpha.txt", "Ravnest supports pipeline parallelism and ring all-reduce."),
            ("beta.txt",  "Ollama provides a simple local LLM runtime."),
        ]:
            (pathlib.Path(tmpdir) / name).write_text(content)

        source = TextSource(paths=[tmpdir], chunk_mode="paragraph",
                            node_id="test-text-files")
        resp = source.query(DataRequest(query="pipeline parallelism", top_k=3))
        assert any("alpha" in c.source for c in resp.chunks), \
            "Expected to find content from alpha.txt"
        _ok(f"File scan: top chunk from {resp.chunks[0].source!r}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. VectorDBSource — NumPy backend (no external deps)
# ─────────────────────────────────────────────────────────────────────────────

def test_vector_db_numpy() -> None:
    _sep("VectorDBSource — NumPy cosine similarity")

    vdb = VectorDBSource(backend="numpy", collection="test-vdb",
                         embed_model=None, node_id="test-vdb")

    # Add pre-computed 4-D vectors
    vdb.add_vectors(
        vectors  = [[1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.9, 0.1, 0.0, 0.0]],
        texts    = ["alpha",  "beta",  "near-alpha"],
        ids      = ["id-a",   "id-b",  "id-c"],
        metadata = [{"tag": "a"}, {"tag": "b"}, {"tag": "c"}],
    )

    resp = vdb.query(DataRequest(vector=[1.0, 0.0, 0.0, 0.0], top_k=2))
    assert len(resp.chunks) == 2
    assert resp.chunks[0].content in ("alpha", "near-alpha"), \
        f"Expected 'alpha' or 'near-alpha' to be top result, got: {resp.chunks[0].content}"
    _ok(f"Top result: '{resp.chunks[0].content}'  score={resp.chunks[0].score}")

    # Filter by metadata tag
    resp2 = vdb.query(DataRequest(vector=[1.0, 0.0, 0.0, 0.0], top_k=3,
                                  filters={"tag": "b"}))
    assert all(c.content == "beta" for c in resp2.chunks)
    _ok(f"Metadata filter works: {[c.content for c in resp2.chunks]}")

    # Health
    hs = vdb.health()
    assert hs.healthy and hs.item_count == 3
    _ok(f"health() → {hs.message}")

    # Capabilities
    cap = vdb.capabilities()
    assert cap.source_type == "vector_db"
    _ok(f"capabilities() → source_type={cap.source_type}, items={cap.item_count}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. GraphDBSource — NetworkX backend
# ─────────────────────────────────────────────────────────────────────────────

def test_graph_db() -> None:
    _sep("GraphDBSource — NetworkX backend")

    try:
        import networkx  # noqa: F401
    except ImportError:
        _skip("networkx not installed — pip install networkx")
        return

    gdb = GraphDBSource(backend="networkx", node_id="test-graph")
    gdb.add_triples([
        ("Alice",   "KNOWS",    "Bob"),
        ("Bob",     "KNOWS",    "Carol"),
        ("Alice",   "WORKS_AT", "Ravnest"),
        ("Bob",     "WORKS_AT", "Ravnest"),
        ("Carol",   "WORKS_AT", "OpenAI"),
    ])

    # Keyword search — find neighbours of "Alice"
    resp = gdb.query(DataRequest(query="Alice", top_k=5))
    assert len(resp.chunks) > 0
    subjects = [c.metadata.get("subject") for c in resp.chunks]
    assert "Alice" in subjects, f"Expected Alice in subjects, got: {subjects}"
    _ok(f"Keyword query 'Alice' → {len(resp.chunks)} triple(s)")

    # Path query
    resp2 = gdb.query(DataRequest(query="path:Alice->Carol", top_k=3))
    assert len(resp2.chunks) > 0
    _ok(f"Path query Alice→Carol → {len(resp2.chunks)} path(s)")

    # Health
    hs = gdb.health()
    assert hs.healthy
    _ok(f"health() → nodes={hs.item_count}")

    # Capabilities
    cap = gdb.capabilities()
    assert cap.source_type == "graph_db"
    _ok(f"capabilities() → source_type={cap.source_type}, items={cap.item_count}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. ImageSource — structural test (no Pillow needed)
# ─────────────────────────────────────────────────────────────────────────────

def test_image_source() -> None:
    _sep("ImageSource — structural test")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create minimal placeholder PNG (1x1 pixel, raw bytes)
        import struct, zlib
        def _png_1x1(r, g, b):
            def chunk(name, data):
                c = struct.pack(">I", len(data)) + name + data
                return c + struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)
            sig    = b"\x89PNG\r\n\x1a\n"
            ihdr   = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            raw    = b"\x00" + bytes([r, g, b])
            idat   = chunk(b"IDAT", zlib.compress(raw))
            iend   = chunk(b"IEND", b"")
            return sig + ihdr + idat + iend

        for name, pixel in [("red.png", (255,0,0)), ("blue.png", (0,0,255))]:
            (pathlib.Path(tmpdir) / name).write_bytes(_png_1x1(*pixel))

        source = ImageSource(paths=[tmpdir], node_id="test-images")

        cap = source.capabilities()
        assert cap.source_type == "image"
        assert cap.item_count == 2
        _ok(f"capabilities() → items={cap.item_count}")

        hs = source.health()
        assert hs.healthy and hs.item_count == 2
        _ok(f"health() → {hs.message}")

        resp = source.query(DataRequest(top_k=2))
        assert len(resp.chunks) == 2
        assert all(c.modality == "image" for c in resp.chunks)
        _ok(f"query(top_k=2) → {len(resp.chunks)} image chunk(s)")

        # base64 content should be non-empty
        assert all(len(c.content) > 0 for c in resp.chunks)
        _ok("All chunks have non-empty base64 content")


# ─────────────────────────────────────────────────────────────────────────────
# 6. DataRouter — offline mode
# ─────────────────────────────────────────────────────────────────────────────

class _StubDataSource(DataSourceBackend):
    def __init__(self, source_type: str, node_id: str, ram_pct: float = 0.0):
        self._source_type = source_type
        self._node_id     = node_id
        self._ram_pct     = ram_pct

    async def aquery(self, request: DataRequest) -> DataResponse:
        return DataResponse(
            chunks     = [DataChunk(content=f"result from {self._node_id}",
                                    modality="text", score=1.0,
                                    source=self._node_id)],
            source     = self._source_type,
            request_id = request.request_id,
        )

    async def astream(self, request: DataRequest):
        yield DataChunk(content=f"stream from {self._node_id}",
                        modality="text", source=self._node_id)

    async def ahealth(self) -> DataSourceHealthStatus:
        return DataSourceHealthStatus(healthy=True, source=self._source_type)

    def capabilities(self) -> DataSourceCapability:
        return DataSourceCapability(
            source_type = self._source_type,
            modalities  = ["text"],
            node_id     = self._node_id,
            extra       = {"address": "local"},
        )


def _make_offline_router() -> DataRouter:
    with mock.patch("ravnest.data_sources.router.DataRouter._refresh_backends"):
        with mock.patch("ravnest.data_sources.router.DataRouter._start_refresh_thread"):
            router = DataRouter.__new__(DataRouter)
            router._strategy         = LoadBasedDataStrategy()
            router._refresh_interval = 30.0
            router._max_retries      = 3
            router._backends         = {}
            router._local_backends   = []
            router._last_refresh     = time.monotonic()
            router._lock             = threading.RLock()
    return router


def test_router_load_based() -> None:
    _sep("DataRouter — LoadBasedDataStrategy")

    router = _make_offline_router()

    from ravnest.data_sources.router import _DataCapWrapper

    class _LoadedCap:
        def __init__(self, cap, ram):
            self.node_id     = cap.node_id
            self.source_type = cap.source_type
            self.subtype     = cap.source_type
            self.current_load = {"ram_percent": ram}
            self.address     = "local"
            self.metadata    = {}
            self.extra       = cap.extra

    s_busy = _StubDataSource("text", "node-busy")
    s_idle = _StubDataSource("text", "node-idle")

    with router._lock:
        router._local_backends = [
            (s_busy, _LoadedCap(s_busy.capabilities(), 85.0)),
            (s_idle, _LoadedCap(s_idle.capabilities(),  3.0)),
        ]

    resp = router.query(DataRequest(query="test"))
    assert "node-idle" in resp.chunks[0].source, \
        f"Expected node-idle, got: {resp.chunks[0].source}"
    _ok("Routes to least-loaded data source (node-idle)")

    info = router.list_backends()
    assert len(info) == 2
    _ok(f"list_backends() returns {len(info)} entries")


def test_router_source_type_strategy() -> None:
    _sep("DataRouter — SourceTypeStrategy")

    router = _make_offline_router()
    router._strategy = SourceTypeStrategy("vector_db")

    router.add_local_backend(_StubDataSource("text",      "node-text"))
    router.add_local_backend(_StubDataSource("vector_db", "node-vdb"))

    resp = router.query(DataRequest(
        query = "test",
        extra = {"source_type": "vector_db"},
    ))
    assert "node-vdb" in resp.chunks[0].source, \
        f"Expected node-vdb, got: {resp.chunks[0].source}"
    _ok("SourceTypeStrategy routes to vector_db node")


def test_router_no_backends() -> None:
    _sep("DataRouter — no backends raises RuntimeError")
    router = _make_offline_router()
    try:
        router.query(DataRequest(query="test"))
        assert False, "Should have raised"
    except RuntimeError as exc:
        assert "No data source" in str(exc)
        _ok(f"Raised RuntimeError: {exc}")


def test_router_remove_backend() -> None:
    _sep("DataRouter — remove_local_backend")
    router = _make_offline_router()
    s = _StubDataSource("text", "node-X")
    router.add_local_backend(s)
    assert len(router._local_backends) == 1
    router.remove_local_backend("node-X")
    assert len(router._local_backends) == 0
    _ok("remove_local_backend() clears the entry")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Async streaming
# ─────────────────────────────────────────────────────────────────────────────

async def _async_stream_test() -> List[DataChunk]:
    router = _make_offline_router()
    router.add_local_backend(_StubDataSource("text", "node-stream"))
    chunks = []
    async for chunk in router.astream(DataRequest(query="stream test")):
        chunks.append(chunk)
    return chunks


def test_async_streaming() -> None:
    _sep("DataRouter — async streaming")
    chunks = asyncio.run(_async_stream_test())
    assert chunks, "No chunks received"
    _ok(f"astream() yielded {len(chunks)} chunk(s): "
        f"'{chunks[0].content[:40]}'")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Sync wrappers on abstract base
# ─────────────────────────────────────────────────────────────────────────────

def test_sync_wrappers() -> None:
    _sep("DataSourceBackend sync wrappers")

    s   = _StubDataSource("text", "sync-test")
    req = DataRequest(query="hi")

    resp = s.query(req)
    assert resp.chunks[0].content == "result from sync-test"
    _ok("query() wraps aquery() correctly")

    chunks = list(s.stream(req))
    assert chunks[0].content == "stream from sync-test"
    _ok("stream() wraps astream() correctly")

    hs = s.health()
    assert hs.healthy
    _ok("health() wraps ahealth() correctly")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Registry-backed router (optional)
# ─────────────────────────────────────────────────────────────────────────────

def test_registry_router(registry_address: str) -> None:
    _sep(f"Registry-backed DataRouter ({registry_address})")
    try:
        router   = DataRouter(registry_address=registry_address, max_retries=1)
        backends = router.list_backends()
        _ok(f"Discovered {len(backends)} data source backend(s)")
        for b in backends:
            print(f"       {b['node_id']:30s}  type={b['source_type']}")
    except Exception as exc:
        _skip(f"Registry unreachable ({exc})")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Ravnest data sources smoke test")
    parser.add_argument("--registry", default=None,
                        help="host:port of a live Ravnest registry")
    args = parser.parse_args()

    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║     Ravnest · Phase-4 Data Source Nodes  smoke test         ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    test_dataclasses()
    test_text_source()
    test_text_source_file()
    test_vector_db_numpy()
    test_graph_db()
    test_image_source()
    test_sync_wrappers()
    test_router_load_based()
    test_router_source_type_strategy()
    test_router_no_backends()
    test_router_remove_backend()
    test_async_streaming()

    if args.registry:
        test_registry_router(args.registry)
    else:
        _sep("Registry-backed DataRouter")
        _skip("pass --registry host:port to run against a live registry")

    _sep()
    print("\n  All offline tests passed ✓\n")


if __name__ == "__main__":
    main()
