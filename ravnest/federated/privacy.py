"""
ravnest.federated.privacy — Differential privacy primitives for FL.

Implements the Gaussian mechanism applied to model updates:

  1. Clip the update to a maximum L2 norm (sensitivity bounding).
  2. Add Gaussian noise N(0, σ²) where σ = noise_multiplier × clip_norm.
  3. Track cumulative ε via analytical Gaussian mechanism accounting.

Privacy guarantee per round
---------------------------
For a single participant's update after clipping + noise:

    ε_round = sqrt(2 × ln(1.25/δ)) / noise_multiplier

Composition over T rounds (naive):

    ε_total ≤ T × ε_round

For tighter bounds use the moments accountant (Abadi et al., 2016) or RDP
(Mironov, 2017).  For production, integrate Opacus (pip install opacus) which
provides per-sample gradient clipping and tight DP-SGD accounting.

Model-update perturbation vs. gradient perturbation
----------------------------------------------------
This module applies DP to the *model update* (new_weights − old_weights),
not to per-batch gradients.  This is "output perturbation" and is simpler to
implement — it perturbs the final result of all local training steps rather
than individual SGD steps.  The L2 sensitivity equals clip_norm (one update
per round per participant), so the accounting is straightforward.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:
    _TORCH = False

from .base import DPConfig


def _chk():
    if not _TORCH:
        raise ImportError("torch is required. pip install torch")


# ── gradient-level DP (applied during local training) ─────────────────────────

def clip_gradients(model: "nn.Module", max_norm: float) -> float:
    """
    Clip *accumulated* gradients to a maximum L2 norm.

    This is the fast "global clipping" approach.  For proper per-sample
    clipping (DP-SGD), use Opacus instead.

    Parameters
    ----------
    model:    Model whose ``.grad`` attributes are clipped in-place.
    max_norm: Maximum L2 norm.

    Returns
    -------
    float
        Pre-clipping gradient norm.
    """
    _chk()
    params = [p for p in model.parameters() if p.grad is not None]
    if not params:
        return 0.0
    total_norm = torch.norm(
        torch.stack([p.grad.detach().norm(2) for p in params]), 2
    ).item()
    clip_coef = max_norm / (total_norm + 1e-8)
    if clip_coef < 1.0:
        for p in params:
            p.grad.detach().mul_(clip_coef)
    return total_norm


def add_dp_noise(
    model: "nn.Module",
    noise_multiplier: float,
    clip_norm: float,
    num_samples: int = 1,
) -> None:
    """
    Add Gaussian noise to gradients for differential privacy.

    σ_grad = noise_multiplier × clip_norm / √num_samples

    Call after ``clip_gradients`` and before ``optimizer.step()``.

    Parameters
    ----------
    model:            Model whose ``.grad`` attributes are noised in-place.
    noise_multiplier: Ratio σ/clip_norm.
    clip_norm:        The clip bound used in ``clip_gradients``.
    num_samples:      Local batch size (scales noise down for larger batches).
    """
    _chk()
    sigma = noise_multiplier * clip_norm / math.sqrt(max(num_samples, 1))
    for p in model.parameters():
        if p.grad is not None:
            p.grad.add_(torch.randn_like(p.grad) * sigma)


# ── update-level DP (applied to the model delta before upload) ─────────────────

def clip_delta(
    delta: Dict[str, "torch.Tensor"],
    max_norm: float,
) -> float:
    """
    Clip a model update delta to a maximum L2 norm (in-place).

    Returns the pre-clipping norm.
    """
    _chk()
    tensors = list(delta.values())
    if not tensors:
        return 0.0
    total_norm = torch.norm(
        torch.stack([v.norm(2) for v in tensors]), 2
    ).item()
    clip_coef = max_norm / (total_norm + 1e-8)
    if clip_coef < 1.0:
        for k in delta:
            delta[k] = delta[k].mul(clip_coef)
    return total_norm


def add_noise_to_delta(
    delta: Dict[str, "torch.Tensor"],
    noise_multiplier: float,
    clip_norm: float,
) -> None:
    """
    Add Gaussian noise to a model update delta (in-place).

    σ_update = noise_multiplier × clip_norm

    Call after ``clip_delta`` and before uploading to the coordinator.
    """
    _chk()
    sigma = noise_multiplier * clip_norm
    for k in delta:
        delta[k] = delta[k] + torch.randn_like(delta[k]) * sigma


def apply_dp_to_delta(
    delta: Dict[str, "torch.Tensor"],
    dp: DPConfig,
) -> float:
    """
    Convenience wrapper: clip + noise a model delta in-place.

    Returns
    -------
    float
        Pre-clipping L2 norm of the delta.
    """
    norm = clip_delta(delta, dp.clip_norm)
    if dp.noise_multiplier > 0:
        add_noise_to_delta(delta, dp.noise_multiplier, dp.clip_norm)
    return norm


# ── privacy accounting ────────────────────────────────────────────────────────

class PrivacyAccountant:
    """
    Tracks cumulative (ε, δ)-DP consumption for the Gaussian mechanism.

    Each call to ``step()`` represents one FL round where the coordinator
    adds Gaussian noise with the configured σ to the aggregated update.

    Analytical bound per round (Gaussian mechanism, output perturbation):

        ε_round = sqrt(2 × ln(1.25 / δ)) / noise_multiplier

    Composition (naive, T rounds):

        ε_total ≤ T × ε_round

    This is conservative.  Use RDP composition for tighter bounds.
    """

    def __init__(self, dp_config: DPConfig) -> None:
        self._cfg   = dp_config
        self._steps = 0
        # ε per FL round for the Gaussian mechanism
        self._eps_per_round = (
            math.sqrt(2.0 * math.log(1.25 / dp_config.delta))
            / dp_config.noise_multiplier
        ) if dp_config.noise_multiplier > 0 else float("inf")

    def step(self) -> float:
        """Record one FL round and return the new cumulative ε."""
        self._steps += 1
        return self.epsilon

    @property
    def epsilon(self) -> float:
        """Current cumulative ε (naive composition)."""
        return self._steps * self._eps_per_round

    @property
    def delta(self) -> float:
        return self._cfg.delta

    @property
    def rounds_completed(self) -> int:
        return self._steps

    def __repr__(self) -> str:
        return (
            f"PrivacyAccountant(ε={self.epsilon:.4f}, δ={self.delta}, "
            f"rounds={self._steps}, noise_mult={self._cfg.noise_multiplier})"
        )
