"""
ravnest.federated.participant — Training hooks for FL participants.

``FLParticipant`` wraps a ``NodeClient`` to the coordinator and provides
the four hooks you weave into a standard PyTorch training loop:

    participant = FLParticipant(
        coordinator_url = "http://coordinator:8768",
        node_id         = "node-1",
        dp_config       = DPConfig(clip_norm=1.0, noise_multiplier=1.1),
    )

    for fl_round in range(config.num_rounds):
        # 1. Pull global model
        round_num = await participant.apull_model(model)

        # 2. Local training
        for epoch in range(config.local_epochs):
            for x, y in local_loader:
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                participant.clip_and_noise(model, num_samples=len(x))
                optimizer.step()

        # 3. Push update + wait for aggregation
        result = await participant.apush_update(model, num_samples=len(local_dataset))
        await participant.await_for_aggregation(round_num)

Sync wrappers (``pull_model``, ``push_update``, ``wait_for_aggregation``)
are available for non-async training loops.  They run the coroutine in a
separate thread to avoid blocking a running event loop.

DP note
-------
``clip_and_noise`` applies DP to *gradients* during local SGD (called before
``optimizer.step()``).  ``apush_update`` *also* applies DP to the model
*update* (delta) before sending it to the coordinator.  Using both gives
stronger privacy guarantees but may harm accuracy.  For most use-cases,
only one mechanism is needed:
  - Use ``clip_and_noise=False`` in ``apush_update`` if you've already
    noised gradients during training.
  - Skip ``clip_and_noise()`` during training if you want output-perturbation
    only (simpler, better accuracy).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import logging
import threading
import uuid
from typing import Optional

from ..mesh.base import NodeMessage
from ..mesh.node_client import NodeClient
from .base import DPConfig, GradientUpdate
from .privacy import (
    apply_dp_to_delta,
    clip_gradients,
    add_dp_noise,
)
from .serializer import compute_delta, pack_state_dict, unpack_state_dict

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:
    _TORCH = False


def _run_sync(coro):
    """Run a coroutine synchronously, safe to call from both sync and async contexts."""
    try:
        loop = asyncio.get_running_loop()
        # Running inside an event loop — offload to a thread
        result_box = []
        error_box  = []

        def _thread():
            try:
                result_box.append(asyncio.run(coro))
            except Exception as e:
                error_box.append(e)

        t = threading.Thread(target=_thread, daemon=True)
        t.start()
        t.join(timeout=600)
        if error_box:
            raise error_box[0]
        return result_box[0] if result_box else None
    except RuntimeError:
        return asyncio.run(coro)


class FLParticipant:
    """
    Training hooks for a federated learning participant node.

    Manages the FL round protocol on behalf of a training process:
    pulling the global model, applying DP during local training,
    computing and uploading the model update, and synchronising with
    the coordinator after aggregation.

    Parameters
    ----------
    coordinator_url: HTTP address of the ``FLCoordinator`` node server.
    node_id:         Unique participant identifier sent with every upload.
    dp_config:       Differential privacy config.  ``None`` disables DP.
    timeout:         Per-request timeout in seconds.
    """

    def __init__(
        self,
        coordinator_url: str,
        node_id:         Optional[str] = None,
        dp_config:       Optional[DPConfig] = None,
        timeout:         float = 120.0,
    ) -> None:
        if not _TORCH:
            raise ImportError("torch required. pip install torch")

        self._client    = NodeClient(coordinator_url, timeout=timeout)
        self._node_id   = node_id or f"participant-{uuid.uuid4().hex[:8]}"
        self._dp        = dp_config
        self._round_num = 0

        # Snapshot of the model at the start of the current round
        # (used to compute the update delta before uploading)
        self._reference_sd: dict = {}

    # ── async API ─────────────────────────────────────────────────────────

    async def apull_model(self, model: "nn.Module", round_num: int = -1) -> int:
        """
        Download the current global model from the coordinator and load
        its weights into ``model``.

        Parameters
        ----------
        model:     The local model to overwrite.
        round_num: Round to request (pass -1 to use current round).

        Returns
        -------
        int
            Round number of the downloaded model.
        """
        msg = NodeMessage(
            node_type = "federated",
            action    = "download",
            payload   = {"round": round_num if round_num >= 0 else self._round_num},
        )
        resp = await self._client.asend(msg)
        if not resp.ok:
            raise RuntimeError(f"pull_model failed: {resp.error}")

        sd = unpack_state_dict(resp.result["model"])
        device = next(model.parameters()).device
        model.load_state_dict({k: v.to(device) for k, v in sd.items()})

        # Save CPU snapshot as the starting point for this round's delta
        self._reference_sd = {k: v.cpu().clone() for k, v in sd.items()}
        self._round_num    = resp.result.get("round", self._round_num)

        logger.debug("[%s] Pulled model for round %d", self._node_id, self._round_num)
        return self._round_num

    async def apush_update(
        self,
        model:       "nn.Module",
        num_samples: int,
        round_num:   int = -1,
        apply_dp:    bool = True,
    ) -> dict:
        """
        Compute the model update (delta = current − reference), optionally
        apply DP, and upload it to the coordinator.

        Parameters
        ----------
        model:       The locally trained model.
        num_samples: Number of samples used in local training (FedAvg weight).
        round_num:   Round this update belongs to (-1 = current round).
        apply_dp:    Apply DP clipping + noise to the delta before uploading.
                     Set to ``False`` if you applied DP via ``clip_and_noise()``
                     during training and don't want double-noising.

        Returns
        -------
        dict
            Coordinator response: ``{"ok": True, "aggregated": bool, ...}``.
        """
        r = round_num if round_num >= 0 else self._round_num

        # Compute update delta (CPU)
        current_sd = {k: v.cpu() for k, v in model.state_dict().items()}
        delta = compute_delta(self._reference_sd, current_sd)

        # Apply DP to the delta
        if apply_dp and self._dp:
            norm = apply_dp_to_delta(delta, self._dp)
            logger.debug("[%s] Applied DP to delta (pre-clip norm=%.4f)", self._node_id, norm)

        update = GradientUpdate(
            round_num   = r,
            node_id     = self._node_id,
            num_samples = num_samples,
            delta       = pack_state_dict(delta),
            metadata    = {"apply_dp": apply_dp and self._dp is not None},
        )

        msg = NodeMessage(
            node_type = "federated",
            action    = "upload",
            payload   = update.to_dict(),
        )
        resp = await self._client.asend(msg)
        if not resp.ok:
            raise RuntimeError(f"push_update failed: {resp.error}")

        if resp.result.get("aggregated"):
            self._round_num = resp.result.get("global_round", r + 1)

        logger.debug("[%s] Pushed update for round %d → %s",
                     self._node_id, r, resp.result)
        return resp.result

    async def await_for_aggregation(
        self,
        round_num: int = -1,
        timeout:   float = 300.0,
    ) -> dict:
        """
        Block until the coordinator finishes aggregating round ``round_num``.

        Returns the RoundResult dict from the coordinator.
        """
        r = round_num if round_num >= 0 else self._round_num - 1
        msg = NodeMessage(
            node_type = "federated",
            action    = "wait",
            payload   = {"round": r, "timeout": timeout},
        )
        resp = await self._client.asend(msg, timeout=timeout + 10)
        if not resp.ok:
            raise RuntimeError(f"wait_for_aggregation failed: {resp.error}")
        return resp.result

    async def astatus(self) -> dict:
        """Fetch coordinator status."""
        msg  = NodeMessage(node_type="federated", action="status", payload={})
        resp = await self._client.asend(msg)
        return resp.result if resp.ok else {"error": resp.error}

    # ── DP training hook (call before optimizer.step()) ───────────────────

    def clip_and_noise(self, model: "nn.Module", num_samples: int = 1) -> float:
        """
        Apply DP to gradients accumulated in the current backward pass.

        Call this after ``loss.backward()`` and before ``optimizer.step()``.

        1. Clips all gradients so the total L2 norm ≤ dp_config.clip_norm.
        2. Adds Gaussian noise scaled to noise_multiplier × clip_norm / √n.

        Parameters
        ----------
        model:       Model with accumulated ``.grad`` attributes.
        num_samples: Batch size (scales noise for mini-batch SGD).

        Returns
        -------
        float
            Pre-clipping gradient norm (0.0 if DP is disabled).
        """
        if self._dp is None:
            return 0.0
        norm = clip_gradients(model, self._dp.clip_norm)
        if self._dp.noise_multiplier > 0:
            add_dp_noise(model, self._dp.noise_multiplier, self._dp.clip_norm, num_samples)
        return norm

    # ── sync wrappers ─────────────────────────────────────────────────────

    def pull_model(self, model: "nn.Module", round_num: int = -1) -> int:
        """Sync wrapper for ``apull_model``."""
        return _run_sync(self.apull_model(model, round_num))

    def push_update(
        self,
        model:       "nn.Module",
        num_samples: int,
        round_num:   int = -1,
        apply_dp:    bool = True,
    ) -> dict:
        """Sync wrapper for ``apush_update``."""
        return _run_sync(self.apush_update(model, num_samples, round_num, apply_dp))

    def wait_for_aggregation(self, round_num: int = -1, timeout: float = 300.0) -> dict:
        """Sync wrapper for ``await_for_aggregation``."""
        return _run_sync(self.await_for_aggregation(round_num, timeout))

    def status(self) -> dict:
        """Sync wrapper for ``astatus``."""
        return _run_sync(self.astatus())

    # ── helpers ───────────────────────────────────────────────────────────

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def current_round(self) -> int:
        return self._round_num
