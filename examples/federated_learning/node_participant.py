"""
Federated Learning Participant Node
=====================================
Trains a local model on synthetic data and participates in FL rounds
coordinated by ``node_coordinator.py``.

Usage
-----
    # Start after the coordinator is running
    python3 node_participant.py --coordinator http://localhost:8768 \
        --node-id node-0 --num-samples 1000

    # With local DP on gradients (output perturbation via coordinator is also on by default)
    python3 node_participant.py --coordinator http://localhost:8768 \
        --node-id node-1 --num-samples 800 \
        --dp-clip-norm 1.0 --dp-noise-multiplier 1.1

    # Run multiple participants in separate terminals (node-0 … node-N)
"""

import argparse
import logging
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fl.participant")


def make_synthetic_data(num_samples: int, input_size: int, num_classes: int):
    """Generate a simple Gaussian synthetic dataset."""
    import torch
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y


def build_model(input_size: int, hidden_size: int, output_size: int):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
    )


def train_local(model, X, y, epochs: int, lr: float, batch_size: int, participant):
    """Run local training for one FL round, applying DP if configured."""
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    loader    = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

    model.train()
    total_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            # Gradient-level DP (optional — only if dp_config set)
            participant.clip_and_noise(model, num_samples=len(xb))
            optimizer.step()
            epoch_loss += loss.item()
        avg = epoch_loss / len(loader)
        total_loss += avg
        logger.debug("  epoch %d loss=%.4f", epoch + 1, avg)

    return total_loss / epochs


def evaluate(model, X, y) -> float:
    import torch
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
        return (preds == y).float().mean().item()


def main():
    parser = argparse.ArgumentParser(description="FL participant node")
    parser.add_argument("--coordinator",       default="http://localhost:8768",
                        help="Coordinator NodeServer URL")
    parser.add_argument("--node-id",           default=None,
                        help="Unique participant ID (auto-generated if omitted)")
    parser.add_argument("--num-samples",       type=int, default=1000,
                        help="Number of synthetic training samples")
    parser.add_argument("--input-size",        type=int, default=784)
    parser.add_argument("--hidden-size",       type=int, default=128)
    parser.add_argument("--output-size",       type=int, default=10)
    parser.add_argument("--local-epochs",      type=int, default=1)
    parser.add_argument("--lr",                type=float, default=0.01)
    parser.add_argument("--batch-size",        type=int, default=64)
    parser.add_argument("--num-rounds",        type=int, default=10)
    parser.add_argument("--timeout",           type=float, default=300.0,
                        help="Wait timeout per round (seconds)")
    parser.add_argument("--retry-interval",    type=float, default=5.0,
                        help="Seconds to wait between coordinator connection retries")
    # Gradient-level DP
    parser.add_argument("--dp-clip-norm",          type=float, default=0.0,
                        help="Gradient clip norm for local DP (0 = disabled)")
    parser.add_argument("--dp-noise-multiplier",   type=float, default=1.1)
    parser.add_argument("--dp-delta",              type=float, default=1e-5)
    # Skip coordinator-side DP (useful when gradient DP is enough)
    parser.add_argument("--no-upload-dp",      action="store_true",
                        help="Disable output-perturbation DP on the uploaded delta")
    args = parser.parse_args()

    from ravnest.federated import FLParticipant, DPConfig

    dp = None
    if args.dp_clip_norm > 0:
        dp = DPConfig(
            clip_norm        = args.dp_clip_norm,
            noise_multiplier = args.dp_noise_multiplier,
            delta            = args.dp_delta,
        )
        logger.info("Local gradient DP: clip_norm=%.2f noise_mult=%.2f",
                    dp.clip_norm, dp.noise_multiplier)

    participant = FLParticipant(
        coordinator_url = args.coordinator,
        node_id         = args.node_id,
        dp_config       = dp,
        timeout         = args.timeout,
    )
    logger.info("Participant %s  →  coordinator %s",
                participant.node_id, args.coordinator)

    # Build local model and dataset
    model = build_model(args.input_size, args.hidden_size, args.output_size)
    X, y  = make_synthetic_data(args.num_samples, args.input_size, args.output_size)
    logger.info("Local dataset: %d samples  model params: %d",
                args.num_samples, sum(p.numel() for p in model.parameters()))

    # Wait for coordinator to be ready
    for attempt in range(20):
        try:
            status = participant.status()
            if status.get("ok"):
                logger.info("Coordinator ready: round=%d/%d",
                            status.get("current_round", 0),
                            status.get("num_rounds", args.num_rounds))
                break
        except Exception as exc:
            logger.info("Waiting for coordinator (%d/20): %s", attempt + 1, exc)
            time.sleep(args.retry_interval)
    else:
        logger.error("Could not connect to coordinator after 20 attempts. Exiting.")
        sys.exit(1)

    # FL training loop
    for fl_round in range(args.num_rounds):
        logger.info("=== FL round %d/%d ===", fl_round + 1, args.num_rounds)

        # 1. Pull global model
        round_num = participant.pull_model(model)
        logger.info("Pulled global model for round %d", round_num)

        # 2. Local training
        avg_loss = train_local(
            model, X, y,
            epochs     = args.local_epochs,
            lr         = args.lr,
            batch_size = args.batch_size,
            participant = participant,
        )
        acc = evaluate(model, X, y)
        logger.info("Local training done: loss=%.4f  acc=%.3f", avg_loss, acc)

        # 3. Push update (with optional output-perturbation DP)
        result = participant.push_update(
            model,
            num_samples = args.num_samples,
            apply_dp    = not args.no_upload_dp,
        )
        logger.info("Update pushed: %s", result)

        # 4. Wait for round aggregation
        if not result.get("aggregated"):
            logger.info("Waiting for aggregation of round %d…", round_num)
            wait_result = participant.wait_for_aggregation(round_num, timeout=args.timeout)
            logger.info("Aggregation complete: %s", wait_result)

    # Final evaluation
    round_num = participant.pull_model(model)
    acc = evaluate(model, X, y)
    logger.info("Final accuracy after %d rounds: %.3f  (round %d)",
                args.num_rounds, acc, round_num)


if __name__ == "__main__":
    main()
