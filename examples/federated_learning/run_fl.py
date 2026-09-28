"""
Federated Learning Launcher
==============================
Starts one coordinator and N participant processes on the local machine.

Usage
-----
    # 3 participants, 10 rounds, no DP
    python3 run_fl.py --num-participants 3 --num-rounds 10

    # 5 participants with coordinator + participant DP
    python3 run_fl.py --num-participants 5 --num-rounds 20 \
        --dp-clip-norm 1.0 --dp-noise-multiplier 1.1

    # Custom ports
    python3 run_fl.py --num-participants 3 --coordinator-port 9000

    # Stop all with Ctrl-C; processes also clean up on exit.
"""

import argparse
import signal
import subprocess
import sys
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def start_coordinator(args) -> subprocess.Popen:
    cmd = [
        sys.executable, os.path.join(HERE, "node_coordinator.py"),
        "--host",             "0.0.0.0",
        "--port",             str(args.coordinator_port),
        "--num-rounds",       str(args.num_rounds),
        "--min-participants", str(args.min_participants or args.num_participants),
        "--local-epochs",     str(args.local_epochs),
        "--aggregator",       args.aggregator,
        "--arch",             args.arch,
        "--input-size",       str(args.input_size),
        "--hidden-size",      str(args.hidden_size),
        "--output-size",      str(args.output_size),
        "--timeout",          str(args.timeout),
    ]
    if args.dp_clip_norm > 0:
        cmd += [
            "--dp-clip-norm",        str(args.dp_clip_norm),
            "--dp-noise-multiplier", str(args.dp_noise_multiplier),
            "--dp-delta",            str(args.dp_delta),
        ]
    if args.save_path:
        cmd += ["--save-path", args.save_path]

    print(f"[launcher] Starting coordinator on port {args.coordinator_port}")
    return subprocess.Popen(cmd)


def start_participant(node_id: str, args) -> subprocess.Popen:
    coordinator_url = f"http://127.0.0.1:{args.coordinator_port}"
    # Stagger sample counts slightly so FedAvg weights differ
    import random
    num_samples = args.num_samples + random.randint(-args.num_samples // 10,
                                                     args.num_samples // 10)
    cmd = [
        sys.executable, os.path.join(HERE, "node_participant.py"),
        "--coordinator",    coordinator_url,
        "--node-id",        node_id,
        "--num-samples",    str(max(100, num_samples)),
        "--input-size",     str(args.input_size),
        "--hidden-size",    str(args.hidden_size),
        "--output-size",    str(args.output_size),
        "--local-epochs",   str(args.local_epochs),
        "--lr",             str(args.lr),
        "--batch-size",     str(args.batch_size),
        "--num-rounds",     str(args.num_rounds),
        "--timeout",        str(args.timeout),
    ]
    if args.dp_clip_norm > 0:
        cmd += [
            "--dp-clip-norm",        str(args.dp_clip_norm),
            "--dp-noise-multiplier", str(args.dp_noise_multiplier),
            "--dp-delta",            str(args.dp_delta),
        ]
    if args.no_upload_dp:
        cmd.append("--no-upload-dp")

    print(f"[launcher] Starting participant {node_id}")
    return subprocess.Popen(cmd)


def main():
    parser = argparse.ArgumentParser(description="FL local launcher")
    parser.add_argument("--num-participants",  type=int, default=3)
    parser.add_argument("--coordinator-port",  type=int, default=8768)
    parser.add_argument("--num-rounds",        type=int, default=10)
    parser.add_argument("--min-participants",  type=int, default=None,
                        help="Min updates before aggregation (default = num-participants)")
    parser.add_argument("--local-epochs",      type=int, default=1)
    parser.add_argument("--num-samples",       type=int, default=1000)
    parser.add_argument("--lr",                type=float, default=0.01)
    parser.add_argument("--batch-size",        type=int, default=64)
    parser.add_argument("--timeout",           type=float, default=300.0)
    parser.add_argument("--participant-delay", type=float, default=3.0,
                        help="Seconds to wait before starting participants")
    # Model
    parser.add_argument("--arch",              default="mlp",
                        choices=["mlp", "logistic"])
    parser.add_argument("--input-size",        type=int, default=784)
    parser.add_argument("--hidden-size",       type=int, default=128)
    parser.add_argument("--output-size",       type=int, default=10)
    # Aggregation
    parser.add_argument("--aggregator",        default="fedavg",
                        choices=["fedavg", "fedprox", "trimmed_mean"])
    # DP
    parser.add_argument("--dp-clip-norm",          type=float, default=0.0)
    parser.add_argument("--dp-noise-multiplier",   type=float, default=1.1)
    parser.add_argument("--dp-delta",              type=float, default=1e-5)
    parser.add_argument("--no-upload-dp",      action="store_true")
    # Misc
    parser.add_argument("--save-path",         default=None)
    args = parser.parse_args()

    procs = []

    def _cleanup(sig=None, frame=None):
        print("\n[launcher] Shutting down all processes…")
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    # Start coordinator
    procs.append(start_coordinator(args))

    # Give it time to bind before participants connect
    print(f"[launcher] Waiting {args.participant_delay}s for coordinator to start…")
    time.sleep(args.participant_delay)

    # Start participants
    for i in range(args.num_participants):
        procs.append(start_participant(f"node-{i}", args))
        time.sleep(0.5)   # slight stagger to avoid simultaneous first pulls

    print(f"[launcher] {args.num_participants} participant(s) + 1 coordinator running.")
    print("[launcher] Press Ctrl-C to stop.")

    # Wait for all processes
    for p in procs:
        try:
            p.wait()
        except KeyboardInterrupt:
            _cleanup()


if __name__ == "__main__":
    main()
