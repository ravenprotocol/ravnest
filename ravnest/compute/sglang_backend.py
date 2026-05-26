"""
SGLangBackend — calls a running SGLang server via its HTTP API.

Start the SGLang server first:
    python -m sglang.launch_server \
        --model-path meta-llama/Llama-3.1-8B-Instruct \
        --host 0.0.0.0 --port 30000

Then:
    backend = SGLangBackend(
        model     = "meta-llama/Llama-3.1-8B-Instruct",
        base_url  = "http://localhost:30000",
    )
    resp = backend.generate(GenerateRequest(prompt="Hi!", max_tokens=64))
    print(resp.text)
"""

from __future__ import annotations

import socket
import time
from typing import AsyncIterator, Optional

from .base import (
    ComputeBackend, ComputeCapability, EmbedRequest, EmbedResponse,
    GenerateRequest, GenerateResponse, HealthStatus,
)

_HTTPX_ERR = "httpx is required for SGLangBackend. Run: pip install httpx"


class SGLangBackend(ComputeBackend):
    """
    ComputeBackend that calls a running SGLang HTTP server.

    Args:
        model:    Model name/path (used for logging and registry metadata).
        base_url: Base URL of the SGLang server (default "http://localhost:30000").
        node_id:  Registry node_id override.
        timeout:  Per-request HTTP timeout in seconds.
    """

    def __init__(
        self,
        model:    str,
        base_url: str            = "http://localhost:30000",
        node_id:  Optional[str]  = None,
        timeout:  float          = 120.0,
    ):
        try:
            import httpx  # noqa: F401
        except ImportError:
            raise ImportError(_HTTPX_ERR)

        self._model   = model
        self._base    = base_url.rstrip("/")
        self._timeout = timeout
        self._node_id = node_id or f"sglang_{socket.gethostname()}"
        self._address = base_url

    # ── async native ─────────────────────────────────────────────────────

    async def agenerate(self, request: GenerateRequest) -> GenerateResponse:
        import httpx
        t0     = time.perf_counter()
        prompt = request.flat_prompt()
        payload = {
            "text":        prompt,
            "sampling_params": {
                "max_new_tokens": request.max_tokens,
                "temperature":    request.temperature,
                "top_p":          request.top_p,
                "top_k":          request.top_k,
                "stop":           request.stop or [],
            },
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(f"{self._base}/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()

        # SGLang returns {"text": "...", "meta_info": {...}}
        generated_text = data.get("text", "")
        # Strip the input prompt from the output if echoed back
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):]

        meta = data.get("meta_info", {})
        return GenerateResponse(
            text          = generated_text,
            model         = self._model,
            backend       = "sglang",
            request_id    = request.request_id,
            finish_reason = meta.get("finish_reason", "stop"),
            usage         = {
                "prompt_tokens":     meta.get("prompt_tokens", 0),
                "completion_tokens": meta.get("completion_tokens", 0),
                "total_tokens":      meta.get("total_tokens", 0),
            },
            latency_ms    = (time.perf_counter() - t0) * 1000,
        )

    async def agenerate_stream(self, request: GenerateRequest) -> AsyncIterator[str]:
        """Stream tokens via SGLang's /generate endpoint with stream=true."""
        import httpx
        import json
        prompt  = request.flat_prompt()
        payload = {
            "text":        prompt,
            "stream":      True,
            "sampling_params": {
                "max_new_tokens": request.max_tokens,
                "temperature":    request.temperature,
                "top_p":          request.top_p,
                "top_k":          request.top_k,
                "stop":           request.stop or [],
            },
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream("POST", f"{self._base}/generate", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    line = line.removeprefix("data:").strip()
                    if line in ("", "[DONE]"):
                        continue
                    try:
                        chunk = json.loads(line)
                        delta = chunk.get("text", "")
                        if delta:
                            yield delta
                    except json.JSONDecodeError:
                        continue

    async def aembed(self, request: EmbedRequest) -> EmbedResponse:
        """Call SGLang's /encode endpoint (if the server supports it)."""
        import httpx
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base}/encode",
                json={"text": request.texts},
            )
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings", [])
        return EmbedResponse(
            embeddings = embeddings,
            model      = self._model,
            backend    = "sglang",
            request_id = request.request_id,
        )

    async def ahealth(self) -> HealthStatus:
        import httpx
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base}/health")
            return HealthStatus(
                healthy = resp.status_code == 200,
                backend = "sglang",
                model   = self._model,
            )
        except Exception as exc:
            return HealthStatus(healthy=False, backend="sglang", model=self._model, message=str(exc))

    # ── capabilities ─────────────────────────────────────────────────────

    def capabilities(self) -> ComputeCapability:
        return ComputeCapability(
            backend             = "sglang",
            models              = [self._model],
            max_context_length  = 4096,
            supports_streaming  = True,
            supports_embeddings = True,
            node_id             = self._node_id,
            extra               = {"base_url": self._base},
        )
