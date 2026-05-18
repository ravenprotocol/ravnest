"""
OllamaBackend — calls the Ollama REST API.

Install Ollama: https://ollama.com/download
Pull a model:   ollama pull llama3.2

Usage:
    backend = OllamaBackend(model="llama3.2")
    resp = backend.generate(GenerateRequest(prompt="Why is the sky blue?", max_tokens=128))
    print(resp.text)
"""

from __future__ import annotations

import json
import socket
import time
from typing import AsyncIterator, List, Optional

from .base import (
    ComputeBackend, ComputeCapability, EmbedRequest, EmbedResponse,
    GenerateRequest, GenerateResponse, HealthStatus,
)

_HTTPX_ERR = "httpx is required for OllamaBackend. Run: pip install httpx"
_DEFAULT_BASE = "http://localhost:11434"


class OllamaBackend(ComputeBackend):
    """
    ComputeBackend that talks to a running Ollama server.

    Args:
        model:    Ollama model tag, e.g. "llama3.2", "mistral", "phi3".
        base_url: Ollama server address (default "http://localhost:11434").
        node_id:  Registry node_id override.
        timeout:  HTTP timeout in seconds.
    """

    def __init__(
        self,
        model:    str,
        base_url: str           = _DEFAULT_BASE,
        node_id:  Optional[str] = None,
        timeout:  float         = 120.0,
    ):
        try:
            import httpx  # noqa: F401
        except ImportError:
            raise ImportError(_HTTPX_ERR)

        self._model   = model
        self._base    = base_url.rstrip("/")
        self._timeout = timeout
        self._node_id = node_id or f"ollama_{socket.gethostname()}"
        self._address = base_url

    # ── async native ─────────────────────────────────────────────────────

    async def agenerate(self, request: GenerateRequest) -> GenerateResponse:
        import httpx
        t0 = time.perf_counter()

        payload = self._build_payload(request, stream=False)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(f"{self._base}/api/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()

        return GenerateResponse(
            text          = data.get("response", ""),
            model         = data.get("model", self._model),
            backend       = "ollama",
            request_id    = request.request_id,
            finish_reason = "stop" if data.get("done") else "length",
            usage         = {
                "prompt_tokens":     data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens":      data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            },
            latency_ms    = (time.perf_counter() - t0) * 1000,
        )

    async def agenerate_stream(self, request: GenerateRequest) -> AsyncIterator[str]:
        """Stream tokens from Ollama (NDJSON, one JSON object per line)."""
        import httpx
        payload = self._build_payload(request, stream=True)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream("POST", f"{self._base}/api/generate", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        if token:
                            yield token
                        if chunk.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue

    async def aembed(self, request: EmbedRequest) -> EmbedResponse:
        """Call Ollama's /api/embeddings endpoint."""
        import httpx
        embeddings = []
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for text in request.texts:
                resp = await client.post(
                    f"{self._base}/api/embeddings",
                    json={"model": request.model or self._model, "prompt": text},
                )
                resp.raise_for_status()
                embeddings.append(resp.json().get("embedding", []))
        return EmbedResponse(
            embeddings = embeddings,
            model      = request.model or self._model,
            backend    = "ollama",
            request_id = request.request_id,
        )

    async def ahealth(self) -> HealthStatus:
        import httpx
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base}/api/tags")
            if resp.status_code == 200:
                tags  = resp.json()
                names = [m["name"] for m in tags.get("models", [])]
                return HealthStatus(
                    healthy = True,
                    backend = "ollama",
                    model   = self._model,
                    message = f"Available models: {names}",
                )
            return HealthStatus(healthy=False, backend="ollama", model=self._model)
        except Exception as exc:
            return HealthStatus(healthy=False, backend="ollama", model=self._model, message=str(exc))

    # ── capabilities ─────────────────────────────────────────────────────

    def capabilities(self) -> ComputeCapability:
        return ComputeCapability(
            backend             = "ollama",
            models              = [self._model],
            max_context_length  = 4096,
            supports_streaming  = True,
            supports_embeddings = True,
            node_id             = self._node_id,
            extra               = {"base_url": self._base},
        )

    # ── helpers ───────────────────────────────────────────────────────────

    def _build_payload(self, request: GenerateRequest, stream: bool) -> dict:
        payload: dict = {
            "model":  request.model or self._model,
            "stream": stream,
            "options": {
                "num_predict": request.max_tokens,
                "temperature": request.temperature,
                "top_p":       request.top_p,
                "top_k":       request.top_k,
            },
        }
        if request.stop:
            payload["options"]["stop"] = request.stop

        # Prefer native chat endpoint when messages are provided
        if request.messages:
            payload["messages"] = [
                {"role": m.role, "content": m.content}
                for m in request.messages
            ]
            # /api/chat instead of /api/generate
            payload["_endpoint"] = "/api/chat"
        else:
            payload["prompt"] = request.flat_prompt()
        return payload

    async def _chat_or_generate(self, client, payload: dict, stream: bool):
        """Route to /api/chat or /api/generate based on payload."""
        endpoint = payload.pop("_endpoint", "/api/generate")
        if stream:
            return client.stream("POST", f"{self._base}{endpoint}", json=payload)
        return await client.post(f"{self._base}{endpoint}", json=payload)
