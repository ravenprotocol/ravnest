"""
ImageSource — serve images from local directories or URLs.

Supported formats: JPEG, PNG, GIF, BMP, WEBP, TIFF, SVG.
Returns images as base64-encoded strings inside DataChunks.

Optional enrichment (auto-detected, first available wins):
  - PIL / Pillow  — ``pip install Pillow``  — thumbnail generation, EXIF
  - CLIP          — ``pip install open-clip-torch`` — semantic image search
  - URLlib        — stdlib — fetch remote images (no extra install)

Usage
-----
    from ravnest.data_sources.image_source import ImageSource
    from ravnest.data_sources.base import DataRequest

    # Serve images from a local directory
    source = ImageSource(paths=["/data/photos", "/data/diagrams"])

    # List all images (no query)
    resp = source.query(DataRequest(top_k=10))
    for chunk in resp.chunks:
        print(chunk.source, chunk.metadata.get("width"), chunk.metadata.get("height"))

    # Semantic search (requires CLIP)
    resp = source.query(DataRequest(query="a dog playing in the park", top_k=3))
"""

from __future__ import annotations

import base64
import os
import socket
import time
import uuid
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional, Set

from .base import (
    DataChunk, DataRequest, DataResponse, DataSourceBackend,
    DataSourceCapability, DataSourceHealthStatus,
)

_IMAGE_EXTS: Set[str] = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp",
    ".webp", ".tiff", ".tif", ".svg",
}


class ImageSource(DataSourceBackend):
    """
    DataSourceBackend that serves images from local paths and/or URLs.

    Args:
        paths:       Local file or directory path(s) to scan.
        urls:        Remote image URLs to include.
        extensions:  Image file extensions to include.
        recursive:   Recurse into sub-directories (default True).
        thumbnail:   Max thumbnail size in pixels (None = return full image).
        semantic:    Enable CLIP-based semantic search (requires open-clip-torch).
        clip_model:  CLIP model name (default "ViT-B-32" / "openai" pretrained).
        node_id:     Registry node_id override.
    """

    def __init__(
        self,
        paths:      str | List[str]  = None,
        urls:       List[str]        = None,
        extensions: Optional[Set[str]] = None,
        recursive:  bool             = True,
        thumbnail:  Optional[int]    = None,
        semantic:   bool             = False,
        clip_model: str              = "ViT-B-32",
        node_id:    Optional[str]    = None,
    ):
        self._paths      = ([paths] if isinstance(paths, str) else list(paths or []))
        self._urls       = urls or []
        self._extensions = extensions or _IMAGE_EXTS
        self._recursive  = recursive
        self._thumbnail  = thumbnail
        self._semantic   = semantic
        self._clip_model = clip_model
        self._node_id    = node_id or f"image_{socket.gethostname()}"

        # Lazy index
        self._entries:  Optional[List[dict]] = None
        self._clip:     Optional[object]     = None

    # ── async interface ───────────────────────────────────────────────────

    async def aquery(self, request: DataRequest) -> DataResponse:
        import asyncio
        t0     = time.perf_counter()
        loop   = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(None, self._sync_query, request)
        return DataResponse(
            chunks      = chunks,
            source      = "image",
            request_id  = request.request_id,
            total_found = len(self._get_entries()),
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
            entries = self._get_entries()
            return DataSourceHealthStatus(
                healthy    = True,
                source     = "image",
                message    = f"Indexed {len(entries)} images",
                item_count = len(entries),
            )
        except Exception as exc:
            return DataSourceHealthStatus(healthy=False, source="image",
                                          message=str(exc))

    def capabilities(self) -> DataSourceCapability:
        try:
            count = len(self._get_entries())
        except Exception:
            count = 0
        return DataSourceCapability(
            source_type = "image",
            modalities  = ["image", "text"],  # text if semantic search available
            item_count  = count,
            node_id     = self._node_id,
            extra       = {
                "paths":    self._paths,
                "semantic": self._semantic,
                "address":  f"{socket.gethostname()}:0",
            },
        )

    # ── private ───────────────────────────────────────────────────────────

    def _get_entries(self) -> List[dict]:
        if self._entries is None:
            self._entries = self._scan()
        return self._entries

    def _scan(self) -> List[dict]:
        entries = []
        for path_str in self._paths:
            path = Path(path_str)
            if path.is_file() and path.suffix.lower() in self._extensions:
                entries.append({"path": str(path), "type": "local"})
            elif path.is_dir():
                pattern = "**/*" if self._recursive else "*"
                for fpath in sorted(path.glob(pattern)):
                    if fpath.is_file() and fpath.suffix.lower() in self._extensions:
                        entries.append({"path": str(fpath), "type": "local"})
        for url in self._urls:
            entries.append({"path": url, "type": "remote"})
        return entries

    def _sync_query(self, request: DataRequest) -> List[DataChunk]:
        entries = self._get_entries()

        # Apply path-based filter
        if request.filters.get("path_contains"):
            pattern = request.filters["path_contains"]
            entries = [e for e in entries if pattern in e["path"]]

        # Semantic search with CLIP
        if request.query and self._semantic:
            entries = self._clip_rank(request.query, entries, request.top_k)
            return [self._load_chunk(e, score=e.get("score", 0.0))
                    for e in entries[:request.top_k]]

        # No query — return first top_k images in scan order
        return [self._load_chunk(e) for e in entries[:request.top_k]]

    def _load_chunk(self, entry: dict, score: float = 0.0) -> DataChunk:
        """Load an image entry into a DataChunk with base64 content."""
        path = entry["path"]
        meta: Dict = {"path": path}

        if entry["type"] == "local":
            try:
                data, meta = _load_local_image(path, self._thumbnail)
            except Exception as exc:
                data = ""
                meta["error"] = str(exc)
        else:
            try:
                data, meta = _load_remote_image(path, self._thumbnail)
            except Exception as exc:
                data = ""
                meta["error"] = str(exc)

        return DataChunk(
            content  = data,
            modality = "image",
            score    = score,
            source   = path,
            metadata = meta,
        )

    def _clip_rank(self, query: str, entries: List[dict],
                   top_k: int) -> List[dict]:
        """Rank images by CLIP text-image similarity."""
        try:
            import open_clip
            import torch

            if self._clip is None:
                model, _, preprocess = open_clip.create_model_and_transforms(
                    self._clip_model, pretrained="openai"
                )
                model.eval()
                tokenizer = open_clip.get_tokenizer(self._clip_model)
                self._clip = (model, preprocess, tokenizer)

            model, preprocess, tokenizer = self._clip

            with torch.no_grad():
                text_tokens = tokenizer([query])
                text_feat   = model.encode_text(text_tokens)
                text_feat   /= text_feat.norm(dim=-1, keepdim=True)

            scored = []
            for entry in entries:
                try:
                    from PIL import Image as PILImage
                    img   = PILImage.open(entry["path"]).convert("RGB")
                    img_t = preprocess(img).unsqueeze(0)
                    with torch.no_grad():
                        img_feat  = model.encode_image(img_t)
                        img_feat  /= img_feat.norm(dim=-1, keepdim=True)
                        similarity = (text_feat @ img_feat.T).item()
                    entry = dict(entry, score=float(similarity))
                    scored.append(entry)
                except Exception:
                    scored.append(dict(entry, score=0.0))

            scored.sort(key=lambda x: -x.get("score", 0.0))
            return scored[:top_k]

        except ImportError:
            # CLIP not available — return all entries unscored
            return entries[:top_k]

    def reload(self) -> None:
        """Force a re-scan of all paths."""
        self._entries = None


# ─────────────────────────────────────────────────────────────────────────────
# Image loading utilities
# ─────────────────────────────────────────────────────────────────────────────

def _load_local_image(path: str, thumbnail: Optional[int]) -> tuple[str, dict]:
    """Read a local image; return (base64_str, metadata_dict)."""
    meta: Dict = {"path": path}
    try:
        from PIL import Image as PILImage
        import io
        img = PILImage.open(path)
        meta["width"]  = img.width
        meta["height"] = img.height
        meta["format"] = img.format or Path(path).suffix.lstrip(".")
        if thumbnail:
            img.thumbnail((thumbnail, thumbnail))
        buf = io.BytesIO()
        img.save(buf, format=img.format or "PNG")
        return base64.b64encode(buf.getvalue()).decode(), meta
    except ImportError:
        pass

    # Fallback: raw read
    with open(path, "rb") as f:
        raw = f.read()
    meta["format"] = Path(path).suffix.lstrip(".")
    return base64.b64encode(raw).decode(), meta


def _load_remote_image(url: str, thumbnail: Optional[int]) -> tuple[str, dict]:
    """Fetch a remote image; return (base64_str, metadata_dict)."""
    import urllib.request
    meta: Dict = {"url": url}
    with urllib.request.urlopen(url, timeout=10) as resp:
        raw = resp.read()
    meta["content_type"] = resp.headers.get("Content-Type", "image/jpeg")

    try:
        from PIL import Image as PILImage
        import io
        img = PILImage.open(io.BytesIO(raw))
        meta["width"]  = img.width
        meta["height"] = img.height
        if thumbnail:
            img.thumbnail((thumbnail, thumbnail))
            buf = io.BytesIO()
            img.save(buf, format=img.format or "PNG")
            raw = buf.getvalue()
    except ImportError:
        pass

    return base64.b64encode(raw).decode(), meta
