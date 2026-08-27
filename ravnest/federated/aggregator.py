"""
ravnest.federated.aggregator — Gradient aggregation strategies for FL.

All aggregators receive a list of ``GradientUpdate`` objects plus the
coordinator's current reference state dict and return a new global state dict.

Available aggregators
---------------------
``FedAvgAggregator``  — weighted average by local dataset size (McMahan 2017).
``FedProxAggregator`` — FedAvg + proximal-term update clipping (Li 2020).
``TrimmedMeanAggregator`` — coordinate-wise trimmed mean for robustness.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from .base import GradientUpdate
from .serializer import unpack_state_dict
from .privacy import clip_delta

logger = logging.getLogger(__name__)

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False


def _chk():
    if not _TORCH:
        raise ImportError("torch required for aggregation. pip install torch")


class FedAvgAggregator:
    """
    Federated Averaging (McMahan et al., 2017).

    Computes:
        Δ_global = Σ_i  (n_i / N) × Δ_i
        new_global = reference + Δ_global

    where n_i = update.num_samples, N = Σ n_i.

    All deltas are moved to CPU before aggregation.
    """

    def aggregate(
        self,
        updates: List[GradientUpdate],
        reference_sd: Dict[str, "torch.Tensor"],
    ) -> Dict[str, "torch.Tensor"]:
        """
        Aggregate participant updates into a new global state dict.

        Parameters
        ----------
        updates:      Non-empty list of GradientUpdate from participants.
        reference_sd: The coordinator's model state dict at the start of the round
                      (CPU tensors).  Used as the base for applying deltas.

        Returns
        -------
        Dict[str, torch.Tensor]
            New global state dict (CPU).
        """
        _chk()
        if not updates:
            raise ValueError("No updates to aggregate")

        total_samples = sum(u.num_samples for u in updates)
        if total_samples == 0:
            raise ValueError("All updates have num_samples=0")

        # Accumulate weighted deltas
        agg_delta: Dict[str, torch.Tensor] = {
            k: torch.zeros_like(v) for k, v in reference_sd.items()
        }

        for update in updates:
            try:
                delta = unpack_state_dict(update.delta)
            except Exception as exc:
                logger.warning("Skipping update from %s — deserialise error: %s",
                               update.node_id, exc)
                continue

            weight = update.num_samples / total_samples
            for k in agg_delta:
                if k in delta:
                    agg_delta[k].add_(delta[k].to(agg_delta[k].device), alpha=weight)

        # Apply aggregated delta to reference
        return {k: reference_sd[k] + agg_delta[k] for k in reference_sd}


class FedProxAggregator(FedAvgAggregator):
    """
    FedProx (Li et al., 2020).

    Clips each participant's update to ``max_delta_norm`` before FedAvg.
    This limits the influence of participants with very large local updates
    (e.g. those with non-i.i.d. data or many local epochs), improving
    convergence in heterogeneous settings.

    Parameters
    ----------
    max_delta_norm: Maximum L2 norm for each participant's delta.
                    Updates larger than this are scaled down.
    """

    def __init__(self, max_delta_norm: float = 10.0) -> None:
        self.max_delta_norm = max_delta_norm

    def aggregate(
        self,
        updates: List[GradientUpdate],
        reference_sd: Dict[str, "torch.Tensor"],
    ) -> Dict[str, "torch.Tensor"]:
        _chk()
        import copy

        clipped_updates: List[GradientUpdate] = []
        for u in updates:
            try:
                delta = unpack_state_dict(u.delta)
            except Exception as exc:
                logger.warning("Skipping update from %s: %s", u.node_id, exc)
                continue
            pre_norm = clip_delta(delta, self.max_delta_norm)
            if pre_norm > self.max_delta_norm:
                logger.debug("Clipped update from %s: norm %.3f → %.3f",
                             u.node_id, pre_norm, self.max_delta_norm)
            u2 = copy.copy(u)
            from .serializer import pack_state_dict
            u2.delta = pack_state_dict(delta)
            clipped_updates.append(u2)

        return super().aggregate(clipped_updates, reference_sd)


class TrimmedMeanAggregator:
    """
    Coordinate-wise trimmed mean (Byzantine-robust aggregation).

    For each parameter coordinate, sorts the values across participants,
    drops the top-``trim_fraction`` and bottom-``trim_fraction`` of values,
    then averages the rest.  Provides robustness against a minority of
    malicious or misbehaving participants.

    Parameters
    ----------
    trim_fraction: Fraction of extreme values to drop on each side.
                   Must be < 0.5.  E.g. 0.1 drops the top and bottom 10%.
    """

    def __init__(self, trim_fraction: float = 0.1) -> None:
        if not 0.0 <= trim_fraction < 0.5:
            raise ValueError("trim_fraction must be in [0, 0.5)")
        self.trim_fraction = trim_fraction

    def aggregate(
        self,
        updates: List[GradientUpdate],
        reference_sd: Dict[str, "torch.Tensor"],
    ) -> Dict[str, "torch.Tensor"]:
        _chk()
        if not updates:
            raise ValueError("No updates to aggregate")

        deltas: List[Dict[str, "torch.Tensor"]] = []
        for u in updates:
            try:
                deltas.append(unpack_state_dict(u.delta))
            except Exception as exc:
                logger.warning("Skipping update from %s: %s", u.node_id, exc)

        if not deltas:
            raise ValueError("All updates failed to deserialise")

        n = len(deltas)
        k = max(0, int(n * self.trim_fraction))    # elements to trim each side

        agg_delta: Dict[str, "torch.Tensor"] = {}
        for key in reference_sd:
            # Stack deltas for this key: shape [n, *param_shape]
            stacked = torch.stack(
                [d[key] for d in deltas if key in d], dim=0
            )
            if stacked.shape[0] == 0:
                agg_delta[key] = torch.zeros_like(reference_sd[key])
                continue

            if k > 0 and stacked.shape[0] > 2 * k:
                sorted_vals, _ = stacked.sort(dim=0)
                trimmed = sorted_vals[k : stacked.shape[0] - k]
                agg_delta[key] = trimmed.mean(dim=0)
            else:
                agg_delta[key] = stacked.mean(dim=0)

        return {k: reference_sd[k] + agg_delta.get(k, torch.zeros_like(v))
                for k, v in reference_sd.items()}
