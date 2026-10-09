"""
ravnest.federated.async_coordinator — Asynchronous FL coordinator.

Unlike ``FLCoordinator`` (which blocks until *all* ``min_participants``
updates arrive), ``AsyncFLCoordinator`` advances rounds on whichever
condition fires first:

  1. **Threshold path** — ``min_participants`` updates received (fast path,
     identical to the sync coordinator).
  2. **Timer path** — ``round_timeout`` seconds elapsed since the round
     started, provided at least ``min_updates_for_advance`` updates are
     queued.  If the minimum is not met the timer restarts until it is.

Bounded staleness
-----------------
Participants that fall behind by up to ``staleness_bound`` rounds can still
contribute.  Their updates are remapped to the current round and optionally
discounted:

    effective_samples = num_samples × staleness_discount ** age

where ``age = current_round − update.round_num`` (0 = current round).
Set ``staleness_bound=0`` (default) to reject stale updates entirely.

Concurrency safety
------------------
All state mutations go through ``self._lock`` (inherited asyncio.Lock).
Both the threshold path (in ``aupload``) and the timer path hold the lock
before calling ``_aggregate()``, so only one can ever run it per round.
The guard ``if self._current_round != round_num: return`` in
``_timer_advance`` covers any residual race.

Usage
-----
    from ravnest.federated import FLConfig, AsyncFLCoordinator
    from ravnest.mesh.node_server import NodeServer
    import torch.nn as nn

    model  = nn.Linear(784, 10)
    config = FLConfig(num_rounds=20, min_participants=5)

    coordinator = AsyncFLCoordinator(
        model, config,
        round_timeout           = 30.0,   # advance after 30 s even with < 5 updates
        min_updates_for_advance = 2,      # but only if >= 2 arrived
        staleness_bound         = 2,      # accept updates <= 2 rounds behind
        staleness_discount      = 0.9,    # discount by 0.9 per stale round
    )
    server = NodeServer(port=8768)
    server.add_federated(coordinator)
    server.run()
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, Optional

from .base import (
    FLConfig, FederatedCapability, FederatedHealthStatus,
    GradientUpdate, RoundResult,
)
from .aggregator import FedAvgAggregator
from .coordinator import FLCoordinator

logger = logging.getLogger(__name__)


class AsyncFLCoordinator(FLCoordinator):
    """
    FL coordinator with timer-based round advancement and bounded staleness.

    Parameters
    ----------
    model, config, aggregator, node_id:
        Forwarded to ``FLCoordinator``.
    round_timeout:
        Seconds after a round starts before the timer fires.
        0 disables the timer entirely (pure threshold mode).
    min_updates_for_advance:
        Minimum updates that must be queued before the timer will aggregate.
        Prevents advancing on a completely empty round.
    staleness_bound:
        Accept updates from up to this many rounds behind current (0 = off).
    staleness_discount:
        Per-stale-round weight decay applied to ``num_samples``.
        1.0 = no discount.
    """

    def __init__(
        self,
        model,
        config:                   FLConfig,
        aggregator:               Optional[FedAvgAggregator] = None,
        node_id:                  Optional[str]              = None,
        round_timeout:            float                      = 60.0,
        min_updates_for_advance:  int                        = 1,
        staleness_bound:          int                        = 0,
        staleness_discount:       float                      = 1.0,
    ) -> None:
        super().__init__(model, config, aggregator, node_id)
        self._round_timeout           = round_timeout
        self._min_updates_for_advance = max(1, min_updates_for_advance)
        self._staleness_bound         = max(0, staleness_bound)
        self._staleness_discount      = max(0.0, min(1.0, staleness_discount))

        self._round_start_times: Dict[int, float] = {}
        self._timer_tasks:       Dict[int, asyncio.Task] = {}

    # ── FederatedBackend overrides ─────────────────────────────────────────

    async def adownload(self, round_num: int) -> dict:
        """Start the round timer on the first download for each round."""
        r = self._current_round
        if r not in self._round_start_times:
            self._round_start_times[r] = time.time()
            self._arm_timer(r)
        return await super().adownload(round_num)

    async def aupload(self, update: GradientUpdate) -> dict:
        """Accept updates, remapping stale ones to the current round."""
        age = self._current_round - update.round_num
        if age > 0:
            if self._staleness_bound == 0 or age > self._staleness_bound:
                # Let parent return stale=True unchanged
                return await super().aupload(update)
            update = self._remap_stale(update, age)
        return await super().aupload(update)

    async def _aggregate(self, round_num: int) -> RoundResult:
        """Cancel the timer, aggregate, then arm the timer for the next round."""
        # Cancel the timer for this round (threshold fired first)
        task = self._timer_tasks.pop(round_num, None)
        if task and not task.done():
            task.cancel()

        result = await super()._aggregate(round_num)

        # Arm the next round's timer immediately so slow participants
        # that haven't downloaded yet don't delay the clock.
        next_round = round_num + 1
        if next_round < self._config.num_rounds:
            self._round_start_times[next_round] = time.time()
            self._arm_timer(next_round)

        return result

    async def astatus(self) -> dict:
        base = await super().astatus()
        r       = self._current_round
        elapsed = time.time() - self._round_start_times.get(r, time.time())
        remaining = (
            max(0.0, self._round_timeout - elapsed)
            if self._round_timeout > 0 else None
        )
        base.update({
            "async_mode":             True,
            "round_timeout_s":        self._round_timeout,
            "round_elapsed_s":        round(elapsed, 1),
            "round_remaining_s":      round(remaining, 1) if remaining is not None else None,
            "min_updates_for_advance": self._min_updates_for_advance,
            "staleness_bound":        self._staleness_bound,
            "staleness_discount":     self._staleness_discount,
        })
        return base

    async def ahealth(self) -> FederatedHealthStatus:
        hs      = await super().ahealth()
        r       = self._current_round
        elapsed = time.time() - self._round_start_times.get(r, time.time())
        n       = len(self._updates.get(r, []))
        hs.message = (
            f"async round={r}/{self._config.num_rounds}  "
            f"updates={n}/{self._config.min_participants}  "
            f"elapsed={elapsed:.0f}s/{self._round_timeout:.0f}s"
        )
        return hs

    def capabilities(self) -> FederatedCapability:
        cap = super().capabilities()
        cap.config.update({
            "async_mode":             True,
            "round_timeout_s":        self._round_timeout,
            "min_updates_for_advance": self._min_updates_for_advance,
            "staleness_bound":        self._staleness_bound,
            "staleness_discount":     self._staleness_discount,
        })
        return cap

    # ── timer internals ────────────────────────────────────────────────────

    def _arm_timer(self, round_num: int) -> None:
        """Create the per-round timer task if the timer is enabled."""
        if self._round_timeout <= 0:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        existing = self._timer_tasks.get(round_num)
        if existing and not existing.done():
            return
        self._timer_tasks[round_num] = loop.create_task(
            self._timer_advance(round_num),
            name=f"fl-timer-r{round_num}",
        )
        logger.debug("Timer armed: round=%d timeout=%.1fs", round_num, self._round_timeout)

    async def _timer_advance(self, round_num: int) -> None:
        """
        Sleep for round_timeout, then aggregate or restart.

        Holds self._lock during the decision so it cannot race with the
        threshold path in aupload → _aggregate.
        """
        try:
            await asyncio.sleep(self._round_timeout)
        except asyncio.CancelledError:
            logger.debug("Timer cancelled: round=%d", round_num)
            return

        async with self._lock:
            if self._current_round != round_num:
                logger.debug("Timer: round=%d already advanced", round_num)
                return

            n = len(self._updates.get(round_num, []))
            if n >= self._min_updates_for_advance:
                logger.info(
                    "Timer fired: aggregating round=%d  updates=%d/%d",
                    round_num, n, self._config.min_participants,
                )
                await self._aggregate(round_num)
            else:
                logger.info(
                    "Timer fired: round=%d  updates=%d < min_advance=%d — restarting",
                    round_num, n, self._min_updates_for_advance,
                )
                self._round_start_times[round_num] = time.time()
                self._arm_timer(round_num)

    # ── staleness helpers ──────────────────────────────────────────────────

    def _remap_stale(self, update: GradientUpdate, age: int) -> GradientUpdate:
        """Return update remapped to current round with discounted sample count."""
        if self._staleness_discount < 1.0:
            discounted = max(1, int(update.num_samples * (self._staleness_discount ** age)))
        else:
            discounted = update.num_samples

        logger.debug(
            "Stale update: node=%s age=%d samples %d→%d",
            update.node_id, age, update.num_samples, discounted,
        )
        return GradientUpdate(
            round_num   = self._current_round,
            node_id     = update.node_id,
            num_samples = discounted,
            delta       = update.delta,
            metadata    = {
                **update.metadata,
                "staleness_age":  age,
                "original_round": update.round_num,
                "sample_factor":  self._staleness_discount ** age,
            },
        )

    # ── public helpers ─────────────────────────────────────────────────────

    @property
    def round_timeout(self) -> float:
        return self._round_timeout

    @property
    def staleness_bound(self) -> int:
        return self._staleness_bound

    def active_timers(self) -> list:
        """Round numbers with a live timer task."""
        return [r for r, t in self._timer_tasks.items() if not t.done()]
