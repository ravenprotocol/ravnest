"""
ravnest.federated — Federated learning hooks for privacy-preserving training.

Quick start
-----------
    # Coordinator side
    from ravnest.federated import FLConfig, DPConfig, FLCoordinator
    from ravnest.mesh.node_server import NodeServer
    import torch.nn as nn

    model  = nn.Linear(784, 10)
    config = FLConfig(num_rounds=10, min_participants=3,
                      dp=DPConfig(clip_norm=1.0, noise_multiplier=1.1))
    coord  = FLCoordinator(model, config)
    server = NodeServer(port=8768)
    server.add_federated(coord)
    server.run()

    # Participant side (in each training process)
    from ravnest.federated import FLParticipant, DPConfig

    participant = FLParticipant(
        coordinator_url = "http://localhost:8768",
        node_id         = "node-0",
        dp_config       = DPConfig(clip_norm=1.0, noise_multiplier=1.1),
    )

    for fl_round in range(config.num_rounds):
        round_num = participant.pull_model(model)
        # ... local training ...
        participant.push_update(model, num_samples=len(local_dataset))
        participant.wait_for_aggregation(round_num)
"""

from .base import (
    DPConfig,
    FLConfig,
    FederatedBackend,
    FederatedCapability,
    FederatedHealthStatus,
    GradientUpdate,
    RoundResult,
)
from .aggregator import (
    FedAvgAggregator,
    FedProxAggregator,
    TrimmedMeanAggregator,
)
from .coordinator import FLCoordinator
from .participant import FLParticipant
from .privacy import (
    PrivacyAccountant,
    clip_gradients,
    add_dp_noise,
    clip_delta,
    add_noise_to_delta,
    apply_dp_to_delta,
)
from .serializer import (
    pack_state_dict,
    unpack_state_dict,
    compute_delta,
    apply_delta,
)

__all__ = [
    # config
    "DPConfig",
    "FLConfig",
    # base types
    "FederatedBackend",
    "FederatedCapability",
    "FederatedHealthStatus",
    "GradientUpdate",
    "RoundResult",
    # aggregators
    "FedAvgAggregator",
    "FedProxAggregator",
    "TrimmedMeanAggregator",
    # core classes
    "FLCoordinator",
    "FLParticipant",
    # privacy
    "PrivacyAccountant",
    "clip_gradients",
    "add_dp_noise",
    "clip_delta",
    "add_noise_to_delta",
    "apply_dp_to_delta",
    # serializer
    "pack_state_dict",
    "unpack_state_dict",
    "compute_delta",
    "apply_delta",
]
