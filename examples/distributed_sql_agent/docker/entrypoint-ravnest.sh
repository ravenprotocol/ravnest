#!/bin/bash
# Entrypoint for the Ravnest pipeline-parallel compute node.
#
# Supports two launch contexts:
#
#   Docker Compose / bare docker run:
#     Set RANK=0|1|2 and MASTER_ADDR=<ip> explicitly in the environment.
#
#   Kubernetes StatefulSet (via Helm chart):
#     POD_NAME is injected via the downward API (e.g. "release-compute-2").
#     The ordinal (last segment after "-") is extracted and used as RANK.
#     MASTER_ADDR defaults to the headless-service DNS of pod 0.
set -euo pipefail

# Extract rank from pod ordinal when running inside a K8s StatefulSet
if [[ -n "${POD_NAME:-}" ]]; then
    RANK="${POD_NAME##*-}"
    export RANK
fi

echo "[ravnest] rank=${RANK}  world=${NUM_NODES}  master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[ravnest] model=${COMPUTE_MODEL}  port=${COMPUTE_PORT}"

exec torchrun \
    --nnodes="${NUM_NODES}" \
    --nproc_per_node=1 \
    --node_rank="${RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    examples/distributed_sql_agent/node_compute_ravnest.py \
        --port      "${COMPUTE_PORT}" \
        --host      "${COMPUTE_HOST}" \
        --model     "${COMPUTE_MODEL}" \
        --num-nodes "${NUM_NODES}"
