"""
ravnest.federated.serializer — Tensor serialisation for FL gradient updates.

All model updates cross the mesh as JSON.  Tensors are serialised to a single
base64-gzipped blob so that:
  - The full state dict round-trips faithfully (dtype, shape preserved).
  - The payload stays JSON-serialisable.
  - gzip reduces typical bandwidth by 30–60 % for float16/float32 weights.

For very large models (>1 GB deltas) consider chunked upload or a binary
transport (gRPC, NCCL direct) instead.
"""

from __future__ import annotations

import base64
import gzip
import io
from typing import Dict

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _require_torch():
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is required for federated serialisation. pip install torch")


# ── pack / unpack ─────────────────────────────────────────────────────────────

def pack_state_dict(state_dict: Dict[str, "torch.Tensor"]) -> str:
    """
    Serialise an entire state dict to a single base64-gzipped string.

    Uses ``torch.save`` so all dtypes and shapes are preserved exactly.

    Returns
    -------
    str
        Base64-encoded gzip-compressed blob, safe to embed in JSON.
    """
    _require_torch()
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    compressed = gzip.compress(buf.getvalue(), compresslevel=6)
    return base64.b64encode(compressed).decode("ascii")


def unpack_state_dict(blob: str) -> Dict[str, "torch.Tensor"]:
    """
    Deserialise a state dict produced by ``pack_state_dict``.

    Returns tensors on CPU regardless of where they were when packed.
    """
    _require_torch()
    raw = gzip.decompress(base64.b64decode(blob))
    # weights_only=True when available (torch >= 2.0) for security
    try:
        return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(io.BytesIO(raw), map_location="cpu")


# ── delta helpers ─────────────────────────────────────────────────────────────

def compute_delta(
    old_sd: Dict[str, "torch.Tensor"],
    new_sd: Dict[str, "torch.Tensor"],
) -> Dict[str, "torch.Tensor"]:
    """
    Compute the model update: delta[k] = new_sd[k] − old_sd[k].

    Only keys present in both dicts are included.  Both inputs must be on CPU.
    """
    _require_torch()
    return {k: (new_sd[k] - old_sd[k]).clone() for k in old_sd if k in new_sd}


def apply_delta(
    model: "torch.nn.Module",
    delta: Dict[str, "torch.Tensor"],
    alpha: float = 1.0,
) -> None:
    """
    Add a scaled delta to ``model``'s weights in-place.

    Parameters
    ----------
    model:  The model to update (in-place).
    delta:  Layer-name → tensor update (on any device; moved to model device).
    alpha:  Scaling factor (default 1.0 = full update).
    """
    _require_torch()
    sd = model.state_dict()
    for k, v in delta.items():
        if k in sd:
            sd[k] = sd[k] + alpha * v.to(sd[k].device)
    model.load_state_dict(sd)
