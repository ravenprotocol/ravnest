# Ravnest Distributed Inference Demo

Run Llama-3.2-3B distributed across 2 GPU containers with an OpenAI-compatible API.

## Prerequisites

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- At least 1 NVIDIA GPU with 8GB+ VRAM
- HuggingFace account with access to `meta-llama/Llama-3.2-3B`

## Quick Start

```bash
# 1. Build and start (first run downloads the model, takes a few minutes)
cd deploy
docker compose up --build

# 2. Wait for "Starting API server on 0.0.0.0:8000" in the logs

# 3. Send a request
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ravnest",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "max_tokens": 50
  }'

# 4. Stop
docker compose down
```

## Configuration

Edit `docker-compose.yml` to change:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `meta-llama/Llama-3.2-3B` | HuggingFace model ID |
| `WORLD_SIZE` | `2` | Number of pipeline stages |
| `MASTER_PORT` | `29500` | torch.distributed rendezvous port |

## API

OpenAI-compatible chat completions endpoint:

- `POST /v1/chat/completions` - Non-streaming chat completion
- `GET /health` - Health check

Works with Open WebUI, LangChain, Continue.dev, and any tool that speaks the OpenAI protocol.
Point your tool at `http://localhost:8000`.

## Running the Smoke Test

```bash
# With the containers running:
bash deploy/test.sh
```

## How It Works

```
User → HTTP POST → node-0 (root)
                    ├── Tokenize prompt
                    ├── Forward through layers 0-N
                    ├── Send activations → node-1 (leaf)
                    │                      ├── Forward through layers N+1-M
                    │                      └── Broadcast next token back
                    ├── Receive token
                    ├── Repeat until done
                    └── Detokenize → HTTP response
```

Both containers load the full model checkpoint, then each prunes to its own layers
using Ravnest's pipeline split spec. Communication uses PyTorch distributed (Gloo backend).

## Limitations (v0.1)

- Non-streaming only (streaming in v0.2)
- Single request at a time (returns 503 if busy)
- Same-machine only (cross-machine in v0.2)
- No authentication
