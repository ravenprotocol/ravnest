"""
ravnest.federated.base — Core types for federated learning over the Ravnest mesh.

Protocol summary
----------------
Every FL round:
  1. Participants call ``FLParticipant.pull_model()``  → coordinator sends global model.
  2. Participants train locally for ``FLConfig.local_epochs`` epochs.
  3. Participants optionally apply DP (clip + noise) to their model update.
  4. Participants call ``FLParticipant.push_update()`` → coordinator receives ``GradientUpdate``.
  5. Coordinator aggregates when ``min_participants`` updates arrive (FedAvg / FedProx).
  6. Participants call ``FLParticipant.wait_for_aggregation()`` to sync before next round.

Wire protocol (NodeMessage payloads)
-------------------------------------
  action="download"  payload={"round": N}
  action="upload"    payload={...GradientUpdate fields...}
  action="status"    payload={}
  action="wait"      payload={"round": N, "timeout": seconds}
  action="health"    payload={}
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# ── DP / FL configuration ─────────────────────────────────────────────────────

@dataclass
class DPConfig:
    """
    Differential privacy configuration for participant gradient updates.

    Uses the Gaussian mechanism:
      sigma = noise_multiplier * clip_norm
      epsilon_per_round = sqrt(2 * ln(1.25/delta)) / noise_multiplier

    For production use, replace with Opacus for tighter DP-SGD guarantees.

    Attributes
    ----------
    clip_norm:         L2 norm bound — model updates are clipped to this norm.
    noise_multiplier:  σ / clip_norm.  Higher = more privacy, less accuracy.
    delta:             Failure probability δ (typ. 1e-5 for privacy analysis).
    """
    clip_norm:        float = 1.0
    noise_multiplier: float = 1.1
    delta:            float = 1e-5


@dataclass
class FLConfig:
    """
    Configuration for an FL training run.

    Attributes
    ----------
    num_rounds:       Total FL rounds to run.
    min_participants: Minimum uploads before coordinator runs aggregation.
    fraction_fit:     Fraction of registered participants selected per round
                      (multiplied by total known — unused unless you register
                      participants with the coordinator explicitly).
    local_epochs:     Suggested local epochs per round (informational; each
                      participant controls its own training).
    dp:               Optional differential privacy configuration.
    timeout:          Seconds coordinator waits for ``min_participants``
                      uploads before timing out a round.
    """
    num_rounds:       int            = 10
    min_participants: int            = 2
    fraction_fit:     float          = 1.0
    local_epochs:     int            = 1
    dp:               Optional[DPConfig] = None
    timeout:          float          = 300.0


# ── wire types ────────────────────────────────────────────────────────────────

@dataclass
class GradientUpdate:
    """
    A model update (delta = new_weights − old_weights) from one participant.

    ``delta`` is the ENTIRE delta state dict serialised as a single
    base64-gzipped blob via ``ravnest.federated.serializer.pack_state_dict``.

    Attributes
    ----------
    round_num:   FL round this update belongs to.
    node_id:     Unique participant identifier.
    num_samples: Number of local training samples used — drives FedAvg weight.
    delta:       Serialised model-update blob (see serializer.pack_state_dict).
    metadata:    Arbitrary key-value pairs (e.g. local loss, training time).
    """
    round_num:   int
    node_id:     str
    num_samples: int
    delta:       str                     # base64-gzipped packed state dict
    metadata:    Dict[str, Any]          = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "round_num":   self.round_num,
            "node_id":     self.node_id,
            "num_samples": self.num_samples,
            "delta":       self.delta,
            "metadata":    self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GradientUpdate":
        return cls(
            round_num   = d["round_num"],
            node_id     = d["node_id"],
            num_samples = d["num_samples"],
            delta       = d["delta"],
            metadata    = d.get("metadata", {}),
        )


@dataclass
class RoundResult:
    """
    Summary returned to participants after a round completes.

    Attributes
    ----------
    round_num:         The round that just finished.
    global_round:      The new current round (round_num + 1).
    num_participants:  How many updates were aggregated.
    aggregated:        True if FedAvg ran; False if the round timed out.
    privacy_epsilon:   Cumulative ε consumed so far (None if DP disabled).
    metadata:          Additional coordinator-supplied info.
    """
    round_num:        int
    global_round:     int
    num_participants: int
    aggregated:       bool                 = True
    privacy_epsilon:  Optional[float]      = None
    metadata:         Dict[str, Any]       = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "round_num":        self.round_num,
            "global_round":     self.global_round,
            "num_participants":  self.num_participants,
            "aggregated":        self.aggregated,
            "privacy_epsilon":   self.privacy_epsilon,
            "metadata":          self.metadata,
        }


# ── capability / health ───────────────────────────────────────────────────────

@dataclass
class FederatedCapability:
    node_id:  str
    role:     str            = "coordinator"   # "coordinator" | "participant"
    config:   Optional[dict] = None            # serialized FLConfig


@dataclass
class FederatedHealthStatus:
    healthy:       bool
    role:          str
    current_round: int  = 0
    message:       str  = ""


# ── abstract base ─────────────────────────────────────────────────────────────

class FederatedBackend(ABC):
    """
    Abstract interface for a federated learning node exposed over the mesh.

    Implementations:
      ``FLCoordinator``  — aggregates updates from participants.
    """

    @abstractmethod
    async def aupload(self, update: GradientUpdate) -> dict:
        """Accept a gradient update from a participant."""

    @abstractmethod
    async def adownload(self, round_num: int) -> dict:
        """Return the serialised global model for round ``round_num``."""

    @abstractmethod
    async def astatus(self) -> dict:
        """Return current round info (round number, uploads received, etc.)."""

    @abstractmethod
    async def await_for_round(self, round_num: int, timeout: float) -> dict:
        """Block until round ``round_num`` is complete or timeout elapses."""

    @abstractmethod
    async def ahealth(self) -> FederatedHealthStatus:
        """Return health status of this federated backend."""

    @abstractmethod
    def capabilities(self) -> FederatedCapability:
        """Return capability descriptor for this node."""

    # ── registry helpers ──────────────────────────────────────────────────

    def register_with_registry(
        self,
        registry_address:   str,
        mesh_address:       str  = "",
        heartbeat_interval: float = 10.0,
    ) -> None:
        """
        Register this federated backend with the Ravnest node registry
        and start sending heartbeats.

        Parameters
        ----------
        registry_address:   Host:port of the gRPC registry server.
        mesh_address:       Public HTTP URL of this node's NodeServer
                            (e.g. "http://coordinator:8768").
                            Stored in metadata so participants can discover
                            the coordinator via NodeClient.from_registry().
        heartbeat_interval: Seconds between heartbeat calls.
        """
        from ravnest.registry import RegistryClient, HeartbeatSender
        from ravnest.registry.capability import NodeCapability, NodeType

        cap  = self.capabilities()
        node = NodeCapability(
            node_id   = cap.node_id,
            node_type = NodeType.FEDERATED,
            subtype   = cap.role,          # "coordinator" | "participant"
            address   = mesh_address.replace("http://", "").replace("https://", ""),
            metadata  = {
                "mesh_address": mesh_address,
                "role":         cap.role,
                **(cap.config or {}),
            },
        )
        self._registry_client  = RegistryClient(registry_address, cache_ttl=30.0)
        self._registry_client.register(node)
        self._heartbeat_sender = HeartbeatSender(
            self._registry_client, cap.node_id,
            interval=heartbeat_interval, cap=node,
        )
        self._heartbeat_sender.start()

    def deregister_from_registry(self) -> None:
        """Gracefully remove this backend from the registry."""
        sender = getattr(self, "_heartbeat_sender", None)
        if sender:
            sender.stop()
        client = getattr(self, "_registry_client", None)
        if client:
            try:
                cap = self.capabilities()
                client.deregister(cap.node_id)
                client.close()
            except Exception:
                pass
