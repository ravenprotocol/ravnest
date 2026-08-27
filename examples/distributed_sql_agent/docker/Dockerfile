# ─────────────────────────────────────────────────────────────────────────────
# Multi-stage Dockerfile for CPU-only Ravnest nodes (Ollama backend, SQL, Agent)
#
# Build context: repo root (ravnest/)
#
# Build individual targets:
#   docker build --target compute -t ravnest-compute -f examples/distributed_sql_agent/docker/Dockerfile .
#   docker build --target sql     -t ravnest-sql     -f examples/distributed_sql_agent/docker/Dockerfile .
#   docker build --target agent   -t ravnest-agent   -f examples/distributed_sql_agent/docker/Dockerfile .
#
# Or via docker compose (from examples/distributed_sql_agent/):
#   docker compose build
# ─────────────────────────────────────────────────────────────────────────────

# ── shared base ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Runtime deps for mesh HTTP layer and Ollama client
# (torch / gRPC not needed — the CPU-only nodes use aiohttp + httpx)
RUN pip install --no-cache-dir \
        aiohttp \
        httpx \
        numpy \
        psutil

# Copy the ravnest package and example files.
# The example scripts add the repo root to sys.path at import time, so no
# pip-install of ravnest is required here.
COPY ravnest/     ./ravnest/
COPY examples/    ./examples/

# ── compute node — Ollama backend ─────────────────────────────────────────────
FROM base AS compute

EXPOSE 8765
ENV COMPUTE_PORT=8765 \
    COMPUTE_HOST=0.0.0.0 \
    COMPUTE_MODEL=llama3.2 \
    OLLAMA_BASE_URL=http://ollama:11434

HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=6 \
    CMD curl -sf http://localhost:${COMPUTE_PORT}/health || exit 1

CMD python3 examples/distributed_sql_agent/node_compute.py \
        --port     ${COMPUTE_PORT} \
        --host     ${COMPUTE_HOST} \
        --model    ${COMPUTE_MODEL} \
        --base-url ${OLLAMA_BASE_URL}

# ── SQL tool node ──────────────────────────────────────────────────────────────
FROM base AS sql

EXPOSE 8766
VOLUME ["/data"]
ENV SQL_PORT=8766 \
    SQL_HOST=0.0.0.0 \
    DB_PATH=/data/shop.db

HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=6 \
    CMD curl -sf http://localhost:${SQL_PORT}/health || exit 1

CMD python3 examples/distributed_sql_agent/node_sql.py \
        --port ${SQL_PORT} \
        --host ${SQL_HOST} \
        --db   ${DB_PATH}

# ── agent node ─────────────────────────────────────────────────────────────────
FROM base AS agent

EXPOSE 8767
VOLUME ["/data"]
ENV AGENT_PORT=8767 \
    AGENT_HOST=0.0.0.0 \
    AGENT_MODEL=llama3.2 \
    COMPUTE_URL=http://compute:8765 \
    SQL_URL=http://sql:8766 \
    DB_PATH=/data/shop.db

HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=6 \
    CMD curl -sf http://localhost:${AGENT_PORT}/health || exit 1

CMD python3 examples/distributed_sql_agent/node_agent.py \
        --port        ${AGENT_PORT} \
        --host        ${AGENT_HOST} \
        --model       ${AGENT_MODEL} \
        --compute-url ${COMPUTE_URL} \
        --sql-url     ${SQL_URL} \
        --db          ${DB_PATH}
