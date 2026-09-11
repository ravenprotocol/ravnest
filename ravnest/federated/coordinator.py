"""
ravnest.federated.coordinator — FL coordinator node.

The coordinator is the trusted aggregation server in a federated learning
setup.  It:
  - Stores and serves the current global model.
  - Accepts gradient updates from participants.
  - Triggers aggregation when enough updates arrive.
  - Optionally adds coordinator-side DP noise to the aggregated update.
  - Signals round completion to waiting participants.

Usage
-----
    import torch.nn as nn
    from ravnest.federated import FLConfig, DPConfig, FLCoordinator, FedAvgAggregator

    model = nn.Linear(784, 10)
    config = FLConfig(num_rounds=5, min_participants=3,
                      dp=DPConfig(clip_norm=1.0, noise_multiplier=1.1))

    coordinator = FLCoordinator(model, config)

    # Expose via the mesh
    from ravnest.mesh.node_server import NodeServer
    server = NodeServer(port=8768)
    server.add_federated(coordinator)
    server.run()
"""

from __future__ import annotations

import asyncio
import copy
import logging
import time
import uuid
from typing import Dict, List, Optional

from .base import (
    DPConfig, FLConfig, FederatedBackend, FederatedCapability,
    FederatedHealthStatus, GradientUpdate, RoundResult,
)
from .aggregator import FedAvgAggregator
from .privacy import PrivacyAccountant, add_noise_to_delta
from .serializer import pack_state_dict, unpack_state_dict, compute_delta

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:
    _TORCH = False


class FLCoordinator(FederatedBackend):
    """
    Federated learning coordinator — aggregation server.

    Manages FL rounds.  Participants upload model updates; once
    ``config.min_participants`` updates arrive the coordinator runs FedAvg
    (or the supplied aggregator), optionally adds coordinator-side DP noise,
    updates the global model, and unblocks any waiting participants.

    Thread-safety: all state mutations are protected by an asyncio Lock.
    The coordinator is single-process; for distributed coordinators use
    the Ravnest registry to shard participants across multiple coordinators.

    Parameters
    ----------
    model:       The global model (any ``nn.Module``).  Weights are updated
                 in-place after each round.
    config:      FL training configuration.
    aggregator:  Aggregation strategy.  Defaults to ``FedAvgAggregator``.
    node_id:     Unique ID for this coordinator node.
    """

    def __init__(
        self,
        model:      "nn.Module",
        config:     FLConfig,
        aggregator: Optional[FedAvgAggregator] = None,
        node_id:    Optional[str]              = None,
    ) -> None:
        if not _TORCH:
            raise ImportError("torch required. pip install torch")

        self._model      = model
        self._config     = config
        self._aggregator = aggregator or FedAvgAggregator()
        self._node_id    = node_id or f"fl-coordinator-{uuid.uuid4().hex[:8]}"

        # Round state
        self._current_round: int = 0
        self._reference_sd:  Dict = {}     # model snapshot at start of current round
        self._updates: Dict[int, List[GradientUpdate]] = {}
        self._round_events:  Dict[int, asyncio.Event] = {}
        self._round_results: Dict[int, RoundResult]   = {}
        self._lock = asyncio.Lock()

        # Privacy
        self._accountant: Optional[PrivacyAccountant] = None
        if config.dp:
            self._accountant = PrivacyAccountant(config.dp)

        # Start time
        self._start_time = time.time()

        # Snapshot the initial model
        self._snapshot()

    # ── FederatedBackend interface ─────────────────────────────────────────

    async def aupload(self, update: GradientUpdate) -> dict:
        """Accept a model update from a participant."""
        async with self._lock:
            r = update.round_num

            if r > self._current_round:
                return {
                    "ok": False,
                    "error": f"Round {r} not started yet (current={self._current_round})",
                }
            if r < self._current_round:
                # Stale upload — already aggregated
                return {
                    "ok": True,
                    "aggregated": True,
                    "stale": True,
                    "global_round": self._current_round,
                }

            # Register the update
            if r not in self._updates:
                self._updates[r] = []
            if r not in self._round_events:
                self._round_events[r] = asyncio.Event()

            # Dedup by node_id
            existing_ids = {u.node_id for u in self._updates[r]}
            if update.node_id in existing_ids:
                return {
                    "ok": True,
                    "aggregated": False,
                    "duplicate": True,
                    "received": len(self._updates[r]),
                }

            self._updates[r].append(update)
            n = len(self._updates[r])
            logger.info("Round %d: received update from %s (%d/%d)",
                        r, update.node_id, n, self._config.min_participants)

            if n >= self._config.min_participants:
                result = await self._aggregate(r)
                return {
                    "ok": True,
                    "aggregated": True,
                    "global_round": result.global_round,
                    "num_participants": result.num_participants,
                    "privacy_epsilon": result.privacy_epsilon,
                }

            return {
                "ok": True,
                "aggregated": False,
                "received": n,
                "needed": self._config.min_participants,
            }

    async def adownload(self, round_num: int) -> dict:
        """Return the serialised global model (reference snapshot for this round)."""
        # If asking for a future round's model, wait until that round starts.
        # For round 0 always available immediately.
        return {
            "ok":    True,
            "round": self._current_round,
            "model": pack_state_dict(self._reference_sd),
        }

    async def astatus(self) -> dict:
        r = self._current_round
        received = len(self._updates.get(r, []))
        eps = self._accountant.epsilon if self._accountant else None
        return {
            "ok":                True,
            "current_round":     r,
            "num_rounds":        self._config.num_rounds,
            "updates_received":  received,
            "min_participants":  self._config.min_participants,
            "done":              r >= self._config.num_rounds,
            "privacy_epsilon":   eps,
            "uptime_s":          time.time() - self._start_time,
        }

    async def await_for_round(self, round_num: int, timeout: float = 300.0) -> dict:
        """Block until round ``round_num`` is complete."""
        # Already done
        if round_num in self._round_results:
            res = self._round_results[round_num]
            return {"ok": True, **res.to_dict()}
        if round_num < self._current_round:
            # Completed before this call
            res = self._round_results.get(round_num)
            if res:
                return {"ok": True, **res.to_dict()}
            return {"ok": True, "round_num": round_num, "global_round": self._current_round}

        # Wait for the event
        async with self._lock:
            if round_num not in self._round_events:
                self._round_events[round_num] = asyncio.Event()
        event = self._round_events[round_num]

        try:
            await asyncio.wait_for(asyncio.shield(event.wait()), timeout=timeout)
            res = self._round_results.get(round_num)
            if res:
                return {"ok": True, **res.to_dict()}
            return {"ok": True, "round_num": round_num, "global_round": self._current_round}
        except asyncio.TimeoutError:
            return {
                "ok":    False,
                "error": f"Timeout waiting for round {round_num} after {timeout}s",
            }

    async def ahealth(self) -> FederatedHealthStatus:
        return FederatedHealthStatus(
            healthy       = True,
            role          = "coordinator",
            current_round = self._current_round,
            message       = (
                f"round={self._current_round}/{self._config.num_rounds}, "
                f"ε={self._accountant.epsilon:.4f}" if self._accountant else
                f"round={self._current_round}/{self._config.num_rounds}, dp=off"
            ),
        )

    def capabilities(self) -> FederatedCapability:
        return FederatedCapability(
            node_id = self._node_id,
            role    = "coordinator",
            config  = {
                "num_rounds":       self._config.num_rounds,
                "min_participants": self._config.min_participants,
                "local_epochs":     self._config.local_epochs,
                "dp_enabled":       self._config.dp is not None,
            },
        )

    # ── public helpers ────────────────────────────────────────────────────

    @property
    def model(self) -> "nn.Module":
        """The current global model."""
        return self._model

    @property
    def current_round(self) -> int:
        return self._current_round

    @property
    def is_done(self) -> bool:
        return self._current_round >= self._config.num_rounds

    def save_model(self, path: str) -> None:
        """Save the global model's state dict to disk."""
        import torch
        torch.save(self._model.state_dict(), path)
        logger.info("Saved global model to %s", path)

    # ── internals ─────────────────────────────────────────────────────────

    def _snapshot(self) -> None:
        """Deep-copy the current model to CPU as the reference for this round."""
        self._reference_sd = copy.deepcopy(
            {k: v.cpu() for k, v in self._model.state_dict().items()}
        )

    async def _aggregate(self, round_num: int) -> RoundResult:
        """Run aggregation for round ``round_num`` and advance the coordinator."""
        updates = self._updates[round_num]

        # Aggregate
        try:
            new_sd = self._aggregator.aggregate(updates, self._reference_sd)
        except Exception as exc:
            logger.error("Aggregation failed for round %d: %s", round_num, exc)
            raise

        # Optional coordinator-side DP noise on the aggregated delta
        # (in addition to any per-participant DP already applied)
        if self._config.dp and self._config.dp.noise_multiplier > 0:
            delta_sd = compute_delta(self._reference_sd, new_sd)
            add_noise_to_delta(
                delta_sd,
                self._config.dp.noise_multiplier,
                self._config.dp.clip_norm,
            )
            new_sd = {k: self._reference_sd[k] + delta_sd[k] for k in self._reference_sd}

        # Update global model (move to original device)
        device = next(self._model.parameters()).device
        self._model.load_state_dict(
            {k: v.to(device) for k, v in new_sd.items()}
        )

        # Privacy accounting
        eps = self._accountant.step() if self._accountant else None

        result = RoundResult(
            round_num        = round_num,
            global_round     = round_num + 1,
            num_participants  = len(updates),
            aggregated       = True,
            privacy_epsilon  = eps,
        )
        self._round_results[round_num] = result

        # Advance round
        self._current_round = round_num + 1
        self._snapshot()

        # Signal waiting participants
        if round_num in self._round_events:
            self._round_events[round_num].set()

        logger.info(
            "Round %d complete: %d participants, ε=%.4f",
            round_num, len(updates), eps or 0.0,
        )
        return result
