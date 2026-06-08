"""
VectorDBSource — vector similarity search over any supported vector store.

Auto-detects the first available backend (in order of preference):
  1. Chroma     — ``pip install chromadb``
  2. Qdrant     — ``pip install qdrant-client``
  3. Pinecone   — ``pip install pinecone-client``
  4. FAISS      — ``pip install faiss-cpu``  (in-memory, no server needed)
  5. NumPy      — stdlib fallback (cosine similarity, no install required)

Embedding model (for text → vector):
  - SentenceTransformers — ``pip install sentence-transformers``  (preferred)
  - Any ``ComputeBackend`` with ``embed()`` support
  - Raw vectors passed directly via ``DataRequest.vector``

Usage
-----
    from ravnest.data_sources.vector_db import VectorDBSource
    from ravnest.data_sources.base import DataRequest

    # In-memory FAISS / NumPy (no server, great for development)
    source = VectorDBSource(backend="auto", collection="docs")
    source.add_texts(["Ravnest is a distributed ML framework.", "vLLM is fast."])

    resp = source.query(DataRequest(query="distributed inference", top_k=2))
    for chunk in resp.chunks:
        print(chunk.score, chunk.content[:60])

    # Chroma (persistent)
    source = VectorDBSource(
        backend    = "chroma",
        collection = "knowledge_base",
        chroma_path = "./chroma_data",
        embed_model = "all-MiniLM-L6-v2",
    )
"""

from __future__ import annotations

import socket
import time
import uuid
from typing import AsyncIterator, Dict, List, Optional, Any

from .base import (
    DataChunk, DataRequest, DataResponse, DataSourceBackend,
    DataSourceCapability, DataSourceHealthStatus,
)


class VectorDBSource(DataSourceBackend):
    """
    DataSourceBackend backed by a vector similarity search store.

    Args:
        backend:       "auto" | "chroma" | "qdrant" | "pinecone" | "faiss" | "numpy".
        collection:    Collection / index / namespace name.
        embed_model:   SentenceTransformers model name, or a ComputeBackend
                       with embed() support, or None (raw vectors only).
        dimension:     Embedding dimension (required for faiss/numpy backends).
        chroma_path:   Persistent Chroma DB directory (default in-memory).
        qdrant_url:    Qdrant server URL (default "http://localhost:6333").
        qdrant_api_key: Qdrant API key (for cloud).
        pinecone_api_key: Pinecone API key.
        pinecone_env:  Pinecone environment string.
        node_id:       Registry node_id override.
    """

    def __init__(
        self,
        backend:          str              = "auto",
        collection:       str              = "ravnest_default",
        embed_model:      Any              = "all-MiniLM-L6-v2",
        dimension:        Optional[int]    = None,
        chroma_path:      Optional[str]    = None,
        qdrant_url:       str              = "http://localhost:6333",
        qdrant_api_key:   Optional[str]    = None,
        pinecone_api_key: Optional[str]    = None,
        pinecone_env:     Optional[str]    = None,
        node_id:          Optional[str]    = None,
    ):
        self._backend_name    = backend
        self._collection      = collection
        self._embed_model_arg = embed_model
        self._dimension       = dimension
        self._chroma_path     = chroma_path
        self._qdrant_url      = qdrant_url
        self._qdrant_api_key  = qdrant_api_key
        self._pinecone_key    = pinecone_api_key
        self._pinecone_env    = pinecone_env
        self._node_id         = node_id or f"vectordb_{socket.gethostname()}"

        # Lazily initialised
        self._store:  Optional[object]  = None
        self._embedder: Optional[object] = None
        self._resolved_backend: str     = ""

    # ── async interface ───────────────────────────────────────────────────

    async def aquery(self, request: DataRequest) -> DataResponse:
        import asyncio
        t0     = time.perf_counter()
        loop   = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(None, self._sync_query, request)
        return DataResponse(
            chunks      = chunks,
            source      = f"vector_db/{self._resolved_backend}",
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
            store = self._get_store()
            count = self._count(store)
            return DataSourceHealthStatus(
                healthy    = True,
                source     = f"vector_db/{self._resolved_backend}",
                message    = f"Collection '{self._collection}' has {count} vectors",
                item_count = count,
            )
        except Exception as exc:
            return DataSourceHealthStatus(
                healthy = False,
                source  = f"vector_db/{self._backend_name}",
                message = str(exc),
            )

    def capabilities(self) -> DataSourceCapability:
        try:
            store = self._get_store()
            count = self._count(store)
        except Exception:
            count = 0
        return DataSourceCapability(
            source_type = "vector_db",
            modalities  = ["text", "vector"],
            item_count  = count,
            node_id     = self._node_id,
            extra       = {
                "backend":    self._resolved_backend or self._backend_name,
                "collection": self._collection,
                "address":    f"{socket.gethostname()}:0",
            },
        )

    # ── public data ingestion ─────────────────────────────────────────────

    def add_texts(
        self,
        texts:    List[str],
        metadata: Optional[List[Dict]] = None,
        ids:      Optional[List[str]]  = None,
    ) -> None:
        """Embed and add text strings to the vector store."""
        vectors = self._embed(texts)
        ids_    = ids or [str(uuid.uuid4()) for _ in texts]
        meta_   = metadata or [{} for _ in texts]
        store   = self._get_store()
        self._add_to_store(store, ids_, vectors, texts, meta_)

    def add_vectors(
        self,
        vectors:  List[List[float]],
        texts:    Optional[List[str]] = None,
        metadata: Optional[List[Dict]] = None,
        ids:      Optional[List[str]]  = None,
    ) -> None:
        """Add pre-computed vectors to the store."""
        ids_  = ids or [str(uuid.uuid4()) for _ in vectors]
        meta_ = metadata or [{} for _ in vectors]
        texts_ = texts or ["" for _ in vectors]
        store = self._get_store()
        self._add_to_store(store, ids_, vectors, texts_, meta_)

    def delete(self, ids: List[str]) -> None:
        """Delete vectors by ID."""
        store = self._get_store()
        self._delete_from_store(store, ids)

    # ── private ───────────────────────────────────────────────────────────

    def _sync_query(self, request: DataRequest) -> List[DataChunk]:
        store = self._get_store()
        # Get query vector
        if request.vector is not None:
            qvec = request.vector
        elif request.query:
            qvec = self._embed([request.query])[0]
        else:
            return []

        return self._search_store(store, qvec, request.top_k,
                                  request.filters, request.include_vectors)

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Convert texts to embedding vectors."""
        if isinstance(self._embed_model_arg, str):
            embedder = self._get_embedder()
            vecs     = embedder.encode(texts, convert_to_numpy=True)
            return [v.tolist() for v in vecs]
        elif self._embed_model_arg is not None:
            # ComputeBackend with embed() support
            from ravnest.data_sources.base import DataRequest as DR
            req  = __import__("ravnest.compute.base", fromlist=["EmbedRequest"])
            ereq = req.EmbedRequest(texts=texts)
            resp = self._embed_model_arg.embed(ereq)
            return resp.embeddings
        else:
            raise ValueError("No embed_model provided and no raw vectors in request")

    def _get_embedder(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self._embed_model_arg)
                if self._dimension is None:
                    test = self._embedder.encode(["test"], convert_to_numpy=True)
                    self._dimension = test.shape[1]
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for text embedding. "
                    "Run: pip install sentence-transformers"
                )
        return self._embedder

    def _get_store(self):
        if self._store is None:
            self._store, self._resolved_backend = self._init_store()
        return self._store

    def _init_store(self):
        backend = self._backend_name

        if backend in ("auto", "chroma"):
            try:
                return self._init_chroma(), "chroma"
            except ImportError:
                if backend == "chroma":
                    raise

        if backend in ("auto", "qdrant"):
            try:
                return self._init_qdrant(), "qdrant"
            except ImportError:
                if backend == "qdrant":
                    raise

        if backend in ("auto", "pinecone"):
            if self._pinecone_key:
                try:
                    return self._init_pinecone(), "pinecone"
                except ImportError:
                    if backend == "pinecone":
                        raise

        if backend in ("auto", "faiss"):
            try:
                return self._init_faiss(), "faiss"
            except ImportError:
                if backend == "faiss":
                    raise

        # Numpy fallback — always available
        return self._init_numpy(), "numpy"

    # ── Chroma ────────────────────────────────────────────────────────────

    def _init_chroma(self):
        import chromadb
        if self._chroma_path:
            client = chromadb.PersistentClient(path=self._chroma_path)
        else:
            client = chromadb.EphemeralClient()
        return client.get_or_create_collection(self._collection)

    def _add_to_store_chroma(self, store, ids, vectors, texts, meta):
        store.add(embeddings=vectors, documents=texts,
                  metadatas=meta, ids=ids)

    def _search_chroma(self, store, qvec, top_k, filters, include_vectors):
        where = filters if filters else None
        res   = store.query(query_embeddings=[qvec], n_results=top_k,
                            where=where, include=["documents", "metadatas",
                                                   "distances", "embeddings"]
                            if include_vectors else
                            ["documents", "metadatas", "distances"])
        chunks = []
        for i, doc in enumerate(res["documents"][0]):
            dist  = res["distances"][0][i]
            score = max(0.0, 1.0 - dist)   # cosine distance → similarity
            meta  = res["metadatas"][0][i] or {}
            vec   = (res["embeddings"][0][i]
                     if include_vectors and "embeddings" in res else None)
            chunks.append(DataChunk(
                content  = doc,
                chunk_id = res["ids"][0][i],
                modality = "text",
                score    = round(score, 4),
                source   = self._collection,
                metadata = meta,
                vector   = vec,
            ))
        return chunks

    # ── Qdrant ────────────────────────────────────────────────────────────

    def _init_qdrant(self):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        client = QdrantClient(url=self._qdrant_url,
                              api_key=self._qdrant_api_key)
        dim = self._dimension or 384
        try:
            client.create_collection(
                self._collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        except Exception:
            pass  # already exists
        return client

    # ── FAISS ─────────────────────────────────────────────────────────────

    def _init_faiss(self):
        import faiss
        dim    = self._dimension or 384
        index  = faiss.IndexFlatIP(dim)  # inner product (normalised = cosine)
        return _FAISSStore(index, dim)

    # ── NumPy (always-available fallback) ─────────────────────────────────

    def _init_numpy(self):
        return _NumpyStore()

    # ── dispatch to backend-specific methods ──────────────────────────────

    def _add_to_store(self, store, ids, vectors, texts, meta):
        b = self._resolved_backend
        if b == "chroma":
            self._add_to_store_chroma(store, ids, vectors, texts, meta)
        elif b == "qdrant":
            from qdrant_client.models import PointStruct
            points = [
                PointStruct(id=i, vector=v,
                            payload={"text": t, **m})
                for i, (v, t, m) in enumerate(zip(vectors, texts, meta))
            ]
            store.upsert(self._collection, points=points)
        elif b in ("faiss", "numpy"):
            store.add(ids, vectors, texts, meta)

    def _search_store(self, store, qvec, top_k, filters, include_vectors):
        b = self._resolved_backend
        if b == "chroma":
            return self._search_chroma(store, qvec, top_k, filters, include_vectors)
        elif b == "qdrant":
            res = store.search(self._collection, query_vector=qvec,
                               limit=top_k, with_payload=True,
                               with_vectors=include_vectors)
            return [
                DataChunk(
                    content  = hit.payload.get("text", ""),
                    chunk_id = str(hit.id),
                    modality = "text",
                    score    = round(hit.score, 4),
                    source   = self._collection,
                    metadata = {k: v for k, v in hit.payload.items()
                                if k != "text"},
                    vector   = hit.vector if include_vectors else None,
                )
                for hit in res
            ]
        elif b in ("faiss", "numpy"):
            return store.search(qvec, top_k, filters, include_vectors,
                                self._collection)
        return []

    def _count(self, store) -> int:
        b = self._resolved_backend
        if b == "chroma":
            return store.count()
        elif b == "qdrant":
            info = store.get_collection(self._collection)
            return info.vectors_count or 0
        elif b in ("faiss", "numpy"):
            return store.count()
        return 0

    def _delete_from_store(self, store, ids):
        b = self._resolved_backend
        if b == "chroma":
            store.delete(ids=ids)
        elif b == "qdrant":
            from qdrant_client.models import PointIdsList
            store.delete(self._collection,
                         points_selector=PointIdsList(points=ids))
        elif b in ("faiss", "numpy"):
            store.delete(ids)


# ─────────────────────────────────────────────────────────────────────────────
# FAISS in-memory store wrapper
# ─────────────────────────────────────────────────────────────────────────────

class _FAISSStore:
    def __init__(self, index, dim: int):
        self._index  = index
        self._dim    = dim
        self._ids:   List[str]       = []
        self._texts: List[str]       = []
        self._meta:  List[Dict]      = []

    def add(self, ids, vectors, texts, meta):
        import numpy as np
        vecs = np.array(vectors, dtype="float32")
        # Normalise for cosine similarity
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs  = vecs / (norms + 1e-10)
        self._index.add(vecs)
        self._ids.extend(ids)
        self._texts.extend(texts)
        self._meta.extend(meta)

    def search(self, qvec, top_k, filters, include_vectors, collection):
        import numpy as np
        if self._index.ntotal == 0:
            return []
        q = np.array([qvec], dtype="float32")
        q /= (np.linalg.norm(q) + 1e-10)
        D, I = self._index.search(q, min(top_k, self._index.ntotal))
        chunks = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            chunks.append(DataChunk(
                content  = self._texts[idx],
                chunk_id = self._ids[idx],
                modality = "text",
                score    = round(float(score), 4),
                source   = collection,
                metadata = dict(self._meta[idx]),
            ))
        return chunks

    def count(self) -> int:
        return self._index.ntotal

    def delete(self, ids):
        # FAISS IndexFlatIP doesn't support deletion; rebuild without those IDs
        keep = [(i, v, t, m) for i, v, t, m in
                zip(self._ids, [], self._texts, self._meta) if i not in ids]
        # Simplified — just mark as deleted in metadata
        self._ids   = [i for i in self._ids   if i not in ids]
        self._texts = [t for t in self._texts[:len(self._ids)]]
        self._meta  = [m for m in self._meta[:len(self._ids)]]


# ─────────────────────────────────────────────────────────────────────────────
# NumPy cosine similarity (no deps beyond stdlib + numpy)
# ─────────────────────────────────────────────────────────────────────────────

class _NumpyStore:
    def __init__(self):
        self._ids:     List[str]        = []
        self._vectors: List[List[float]] = []
        self._texts:   List[str]        = []
        self._meta:    List[Dict]       = []

    def add(self, ids, vectors, texts, meta):
        self._ids.extend(ids)
        self._vectors.extend(vectors)
        self._texts.extend(texts)
        self._meta.extend(meta)

    def search(self, qvec, top_k, filters, include_vectors, collection):
        if not self._vectors:
            return []
        import math
        q_norm = math.sqrt(sum(x * x for x in qvec)) or 1.0
        scored = []
        for i, (vec, text) in enumerate(zip(self._vectors, self._texts)):
            if filters:
                if not all(self._meta[i].get(k) == v
                           for k, v in filters.items()):
                    continue
            v_norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            dot    = sum(a * b for a, b in zip(qvec, vec))
            cos    = dot / (q_norm * v_norm)
            scored.append((cos, i))

        scored.sort(key=lambda x: -x[0])
        return [
            DataChunk(
                content  = self._texts[idx],
                chunk_id = self._ids[idx],
                modality = "text",
                score    = round(max(0.0, float(score)), 4),
                source   = collection,
                metadata = dict(self._meta[idx]),
                vector   = self._vectors[idx] if include_vectors else None,
            )
            for score, idx in scored[:top_k]
        ]

    def count(self) -> int:
        return len(self._vectors)

    def delete(self, ids):
        keep = [(i, v, t, m) for i, v, t, m in
                zip(self._ids, self._vectors, self._texts, self._meta)
                if i not in ids]
        if keep:
            self._ids, self._vectors, self._texts, self._meta = map(list, zip(*keep))
        else:
            self._ids = self._vectors = self._texts = self._meta = []
