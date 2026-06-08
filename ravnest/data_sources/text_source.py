"""
TextSource — serve text files and directories as a data source node.

Reads plain text, Markdown, and other text-based files from a local path
(or list of paths).  Supports:

  - Recursive directory scanning with glob patterns
  - Three chunking modes: "paragraph", "sentence", "fixed" (fixed-size windows)
  - Simple keyword / BM25-style relevance scoring for ranked retrieval
  - Streaming chunk delivery

Install (optional, for better sentence splitting):
    pip install nltk

Usage
-----
    from ravnest.data_sources.text_source import TextSource
    from ravnest.data_sources.base import DataRequest

    source = TextSource(
        paths      = ["/data/docs", "/data/readme.md"],
        chunk_mode = "paragraph",
        chunk_size = 500,        # tokens / chars (for "fixed" mode)
    )

    resp = source.query(DataRequest(query="distributed inference", top_k=3))
    for chunk in resp.chunks:
        print(chunk.score, chunk.source, chunk.content[:80])
"""

from __future__ import annotations

import math
import os
import re
import socket
import time
from collections import defaultdict
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional, Set

from .base import (
    DataChunk, DataRequest, DataResponse, DataSourceBackend,
    DataSourceCapability, DataSourceHealthStatus,
)

_TEXT_EXTS: Set[str] = {
    ".txt", ".md", ".rst", ".csv", ".log", ".json",
    ".yaml", ".yml", ".toml", ".html", ".htm", ".xml",
    ".py", ".js", ".ts", ".go", ".rs", ".cpp", ".c", ".h",
}


class TextSource(DataSourceBackend):
    """
    DataSourceBackend that serves chunked text from local files/directories.

    Args:
        paths:       File or directory path(s) to serve.
        extensions:  File extensions to include (default: common text types).
        chunk_mode:  "paragraph" | "sentence" | "fixed".
        chunk_size:  Character count for "fixed" mode (default 800).
        chunk_overlap: Overlap chars between fixed chunks (default 100).
        recursive:   Recurse into sub-directories (default True).
        node_id:     Registry node_id override.
        encoding:    File encoding (default "utf-8", errors ignored).
    """

    def __init__(
        self,
        paths:         str | List[str],
        extensions:    Optional[Set[str]] = None,
        chunk_mode:    str               = "paragraph",
        chunk_size:    int               = 800,
        chunk_overlap: int               = 100,
        recursive:     bool              = True,
        node_id:       Optional[str]     = None,
        encoding:      str               = "utf-8",
    ):
        self._paths        = [paths] if isinstance(paths, str) else list(paths)
        self._extensions   = extensions or _TEXT_EXTS
        self._chunk_mode   = chunk_mode
        self._chunk_size   = chunk_size
        self._chunk_overlap = chunk_overlap
        self._recursive    = recursive
        self._node_id      = node_id or f"text_{socket.gethostname()}"
        self._encoding     = encoding

        # Build index lazily on first query
        self._index: Optional[_TextIndex] = None

    # ── async interface ───────────────────────────────────────────────────

    async def aquery(self, request: DataRequest) -> DataResponse:
        import asyncio
        t0     = time.perf_counter()
        loop   = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(None, self._sync_query, request)
        return DataResponse(
            chunks      = chunks,
            source      = "text",
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
            idx = self._get_index()
            return DataSourceHealthStatus(
                healthy    = True,
                source     = "text",
                message    = f"Indexed {idx.num_chunks} chunks from {idx.num_files} files",
                item_count = idx.num_chunks,
            )
        except Exception as exc:
            return DataSourceHealthStatus(healthy=False, source="text",
                                          message=str(exc))

    def capabilities(self) -> DataSourceCapability:
        try:
            idx = self._get_index()
            count = idx.num_chunks
        except Exception:
            count = 0
        return DataSourceCapability(
            source_type = "text",
            modalities  = ["text"],
            item_count  = count,
            node_id     = self._node_id,
            extra       = {
                "paths":      self._paths,
                "chunk_mode": self._chunk_mode,
                "address":    f"{socket.gethostname()}:0",
            },
        )

    # ── private ───────────────────────────────────────────────────────────

    def _get_index(self) -> "_TextIndex":
        if self._index is None:
            self._index = _TextIndex(
                paths         = self._paths,
                extensions    = self._extensions,
                chunk_mode    = self._chunk_mode,
                chunk_size    = self._chunk_size,
                chunk_overlap = self._chunk_overlap,
                recursive     = self._recursive,
                encoding      = self._encoding,
            )
        return self._index

    def _sync_query(self, request: DataRequest) -> List[DataChunk]:
        idx    = self._get_index()
        chunks = idx.search(request.query, request.top_k, request.filters)
        return chunks

    def add_text(self, text: str, source: str = "inline") -> None:
        """Add a raw text string to the index at runtime."""
        idx = self._get_index()
        chunks = _split_text(text, self._chunk_mode, self._chunk_size,
                              self._chunk_overlap, source)
        idx.add_chunks(chunks)

    def reload(self) -> None:
        """Force a full re-scan of all paths."""
        self._index = None


# ─────────────────────────────────────────────────────────────────────────────
# In-memory BM25-style text index
# ─────────────────────────────────────────────────────────────────────────────

class _TextIndex:
    def __init__(self, paths, extensions, chunk_mode, chunk_size,
                 chunk_overlap, recursive, encoding):
        self._chunks:    List[DataChunk] = []
        self._tf:        List[Dict[str, float]] = []  # per-chunk term freq
        self._df:        Dict[str, int] = defaultdict(int)   # doc freq
        self.num_files   = 0
        self.num_chunks  = 0

        for path_str in paths:
            path = Path(path_str)
            if path.is_file():
                self._index_file(path, encoding, chunk_mode, chunk_size,
                                  chunk_overlap)
            elif path.is_dir():
                pattern = "**/*" if recursive else "*"
                for fpath in path.glob(pattern):
                    if fpath.is_file() and fpath.suffix.lower() in extensions:
                        self._index_file(fpath, encoding, chunk_mode,
                                          chunk_size, chunk_overlap)

        self.num_chunks = len(self._chunks)
        self._N         = self.num_chunks  # for IDF

    def _index_file(self, path: Path, encoding, mode, size, overlap) -> None:
        try:
            text = path.read_text(encoding=encoding, errors="ignore")
        except Exception:
            return
        self.num_files += 1
        chunks = _split_text(text, mode, size, overlap, str(path))
        self.add_chunks(chunks)

    def add_chunks(self, chunks: List[DataChunk]) -> None:
        for chunk in chunks:
            terms = _tokenize(chunk.content)
            tf: Dict[str, float] = defaultdict(float)
            for t in terms:
                tf[t] += 1.0
            # Normalise TF
            total = sum(tf.values()) or 1
            for t in tf:
                tf[t] /= total
                self._df[t] += 1
            self._tf.append(dict(tf))
            self._chunks.append(chunk)
        self._N         = len(self._chunks)
        self.num_chunks = self._N  # keep the public counter in sync

    def search(self, query: str, top_k: int,
               filters: Dict) -> List[DataChunk]:
        if not query.strip():
            # No query — return top_k chunks in order
            result = self._chunks[:top_k]
            return [_copy_chunk(c, 0.0) for c in result]

        query_terms = _tokenize(query)
        scores: List[tuple[float, int]] = []

        for idx, (chunk, tf) in enumerate(zip(self._chunks, self._tf)):
            # Filter
            if filters:
                if not all(chunk.metadata.get(k) == v
                           for k, v in filters.items()):
                    continue
            score = _bm25_score(query_terms, tf, self._df, self._N)
            if score > 0:
                scores.append((score, idx))

        scores.sort(key=lambda x: -x[0])
        # Normalise scores to [0, 1]
        max_score = scores[0][0] if scores else 1.0
        result = []
        for score, idx in scores[:top_k]:
            result.append(_copy_chunk(self._chunks[idx],
                                      round(score / max_score, 4)))
        return result


def _split_text(text: str, mode: str, size: int, overlap: int,
                source: str) -> List[DataChunk]:
    if mode == "paragraph":
        parts = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    elif mode == "sentence":
        parts = _split_sentences(text)
    else:  # fixed
        parts = []
        step  = max(1, size - overlap)
        for i in range(0, len(text), step):
            part = text[i: i + size].strip()
            if part:
                parts.append(part)

    chunks = []
    for i, part in enumerate(parts):
        chunks.append(DataChunk(
            content  = part,
            modality = "text",
            source   = source,
            metadata = {"chunk_index": i, "chunk_mode": mode},
        ))
    return chunks


def _split_sentences(text: str) -> List[str]:
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        sents = nltk.sent_tokenize(text)
        return [s.strip() for s in sents if s.strip()]
    except ImportError:
        pass
    # Fallback: split on . ! ? boundaries
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b[a-z]{2,}\b", text.lower())


def _bm25_score(query_terms: List[str], tf: Dict[str, float],
                df: Dict[str, int], N: int,
                k1: float = 1.5, b: float = 0.75) -> float:
    score = 0.0
    doc_len = sum(tf.values())
    avg_len = 1.0  # normalised already
    for term in query_terms:
        if term not in tf:
            continue
        df_t  = max(df.get(term, 0), 1)
        idf   = math.log((N - df_t + 0.5) / (df_t + 0.5) + 1)
        freq  = tf[term]
        score += idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * doc_len / avg_len))
    return score


def _copy_chunk(chunk: DataChunk, score: float) -> DataChunk:
    return DataChunk(
        content  = chunk.content,
        chunk_id = chunk.chunk_id,
        modality = chunk.modality,
        score    = score,
        source   = chunk.source,
        metadata = dict(chunk.metadata),
    )
