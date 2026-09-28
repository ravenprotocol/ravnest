"""
Federated Learning Coordinator Node
====================================
Hosts the global model and runs FedAvg aggregation over the Ravnest mesh.

Usage
-----
    python3 node_coordinator.py --port 8768 --num-rounds 10 --min-participants 3

    # With differential privacy
    python3 node_coordinator.py --port 8768 --num-rounds 10 --min-participants 3 \
        --dp-clip-norm 1.0 --dp-noise-multiplier 1.1 --dp-delta 1e-5

    # Save the final global model
    python3 node_coordinator.py --port 8768 --save-path /tmp/global_model.pt
"""

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fl.coordinator")


def build_model(arch: str, input_size: int, hidden_size: int, output_size: int):
    """Build the global model architecture."""
    import torch.nn as nn
    if arch == "mlp":
        return nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
    elif arch == "logistic":
        return nn.Linear(input_size, output_size)
    else:
        raise ValueError(f"Unknown arch: {arch}")


def main():
    parser = argparse.ArgumentParser(description="FL coordinator node")
    parser.add_argument("--host",              default="0.0.0.0")
    parser.add_argument("--port",              type=int, default=8768)
    parser.add_argument("--node-id",           default=None)
    parser.add_argument("--num-rounds",        type=int, default=10)
    parser.add_argument("--min-participants",  type=int, default=2)
    parser.add_argument("--fraction-fit",      type=float, default=1.0)
    parser.add_argument("--local-epochs",      type=int, default=1)
    parser.add_argument("--timeout",           type=float, default=300.0,
                        help="Per-round aggregation timeout (seconds)")
    # Model
    parser.add_argument("--arch",              default="mlp",
                        choices=["mlp", "logistic"])
    parser.add_argument("--input-size",        type=int, default=784)
    parser.add_argument("--hidden-size",       type=int, default=128)
    parser.add_argument("--output-size",       type=int, default=10)
    # Differential privacy
    parser.add_argument("--dp-clip-norm",          type=float, default=0.0,
                        help="DP clip norm (0 = DP disabled)")
    parser.add_argument("--dp-noise-multiplier",   type=float, default=1.1)
    parser.add_argument("--dp-delta",              type=float, default=1e-5)
    # Aggregation
    parser.add_argument("--aggregator",        default="fedavg",
                        choices=["fedavg", "fedprox", "trimmed_mean"])
    parser.add_argument("--fedprox-max-norm",  type=float, default=10.0)
    parser.add_argument("--trim-fraction",     type=float, default=0.1)
    # Checkpointing
    parser.add_argument("--save-path",         default=None,
                        help="Path to save the global model when training completes")
    # Registry
    parser.add_argument("--registry",          default=None,
                        help="Ravnest registry address, e.g. localhost:50099")
    args = parser.parse_args()

    from ravnest.federated import (
        FLConfig, DPConfig, FLCoordinator,
        FedAvgAggregator, FedProxAggregator, TrimmedMeanAggregator,
    )
    from ravnest.mesh.node_server import NodeServer

    dp = None
    if args.dp_clip_norm > 0:
        dp = DPConfig(
            clip_norm        = args.dp_clip_norm,
            noise_multiplier = args.dp_noise_multiplier,
            delta            = args.dp_delta,
        )
        logger.info("DP enabled: clip_norm=%.2f noise_mult=%.2f delta=%.0e",
                    dp.clip_norm, dp.noise_multiplier, dp.delta)
    else:
        logger.info("DP disabled (pass --dp-clip-norm > 0 to enable)")

    config = FLConfig(
        num_rounds       = args.num_rounds,
        min_participants = args.min_participants,
        fraction_fit     = args.fraction_fit,
        local_epochs     = args.local_epochs,
        dp               = dp,
        timeout          = args.timeout,
    )

    if args.aggregator == "fedprox":
        aggregator = FedProxAggregator(max_delta_norm=args.fedprox_max_norm)
    elif args.aggregator == "trimmed_mean":
        aggregator = TrimmedMeanAggregator(trim_fraction=args.trim_fraction)
    else:
        aggregator = FedAvgAggregator()

    model = build_model(args.arch, args.input_size, args.hidden_size, args.output_size)
    logger.info("Global model: arch=%s  params=%d",
                args.arch, sum(p.numel() for p in model.parameters()))

    coordinator = FLCoordinator(
        model      = model,
        config     = config,
        aggregator = aggregator,
        node_id    = args.node_id,
    )

    server = NodeServer(host=args.host, port=args.port, node_id=args.node_id)
    server.add_federated(coordinator)

    logger.info("FL coordinator starting on http://%s:%d", args.host, args.port)
    logger.info("Config: rounds=%d  min_participants=%d  aggregator=%s",
                args.num_rounds, args.min_participants, args.aggregator)

    try:
        server.run(registry_address=args.registry)
    finally:
        if args.save_path and coordinator.is_done:
            coordinator.save_model(args.save_path)
            logger.info("Global model saved to %s", args.save_path)


if __name__ == "__main__":
    main()
