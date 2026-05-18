"""
OpenAICompatBackend — any OpenAI-compatible HTTP API.

Works with: OpenAI, Together AI, Groq, Fireworks, Anyscale, local vLLM/SGLang
OpenAI-compat servers, and anything else that speaks the OpenAI Chat / Completions
protocol.

Install dependency:  pip install openai

Usage:
    # OpenAI
    backend = OpenAICompatBackend(model="gpt-4o", api_key="sk-...")

    # Together AI
    backend = OpenAICompatBackend(
        model    = "meta-llama/Llama-3-8b-chat-hf",
        base_url = "https://api.together.xyz/v1",
        api_key  = "...",
    )

    # Local vLLM server
    backend = OpenAICompatBackend(
        model    = "meta-llama/Llama-3.1-8B-Instruct",
        base_url = "http://localhost:8000/v1",
        api_key  = "EMPTY",
    )

    resp = backend.generate(GenerateRequest(
        messages=[Message("user", "Tell me a joke.")],
        max_tokens=128,
    ))
    print(resp.text)
"""

from __future__ import annotations

import socket
import time
from typing import AsyncIterator, List, Optional

from .base import (
    ComputeBackend, ComputeCapability, EmbedRequest, EmbedResponse,
    GenerateRequest, GenerateResponse, HealthStatus, Message,
)

_OPENAI_ERR = "openai is not installed. Run: pip install openai"
_DEFAULT_BASE = "https://api.openai.com/v1"


class OpenAICompatBackend(ComputeBackend):
    """
    ComputeBackend that talks to any OpenAI-compatible REST API.

    Prefers the Chat Completions endpoint when ``messages`` are supplied;
    falls back to Completions for raw ``prompt`` strings.

    Args:
        model:       Model identifier, e.g. "gpt-4o", "llama-3-8b".
        base_url:    API base URL (default: OpenAI production).
        api_key:     API key (or "EMPTY" for local servers that don't require one).
        org_id:      Optional organisation ID (OpenAI-specific).
        timeout:     HTTP timeout in seconds.
        node_id:     Registry node_id override.
        extra_kwargs: Passed verbatim to ``AsyncOpenAI()``.
    """

    def __init__(
        self,
        model:        str,
        base_url:     str           = _DEFAULT_BASE,
        api_key:      str           = "EMPTY",
        org_id:       Optional[str] = None,
        timeout:      float         = 60.0,
        node_id:      Optional[str] = None,
        extra_kwargs: dict          = None,
    ):
        try:
            import openai  # noqa: F401
        except ImportError:
            raise ImportError(_OPENAI_ERR)

        self._model    = model
        self._base_url = base_url
        self._api_key  = api_key
        self._org_id   = org_id
        self._timeout  = timeout
        self._node_id  = node_id or f"openai_compat_{socket.gethostname()}"
        self._address  = base_url
        self._extra    = extra_kwargs or {}

    def _client(self):
        """Create a fresh AsyncOpenAI client (lightweight, no persistent state)."""
        from openai import AsyncOpenAI
        kwargs = dict(
            api_key  = self._api_key,
            base_url = self._base_url,
            timeout  = self._timeout,
            **self._extra,
        )
        if self._org_id:
            kwargs["organization"] = self._org_id
        return AsyncOpenAI(**kwargs)

    # ── async native ─────────────────────────────────────────────────────

    async def agenerate(self, request: GenerateRequest) -> GenerateResponse:
        t0     = time.perf_counter()
        model  = request.model or self._model
        client = self._client()

        if request.messages:
            # Chat Completions
            msgs = [{"role": m.role, "content": m.content} for m in request.messages]
            resp = await client.chat.completions.create(
                model       = model,
                messages    = msgs,
                max_tokens  = request.max_tokens,
                temperature = request.temperature,
                top_p       = request.top_p,
                stop        = request.stop,
                stream      = False,
            )
            text          = resp.choices[0].message.content or ""
            finish_reason = resp.choices[0].finish_reason or "stop"
            usage         = {
                "prompt_tokens":     resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens":      resp.usage.total_tokens,
            }
        else:
            # Legacy Completions
            resp = await client.completions.create(
                model       = model,
                prompt      = request.flat_prompt(),
                max_tokens  = request.max_tokens,
                temperature = request.temperature,
                top_p       = request.top_p,
                stop        = request.stop,
                stream      = False,
            )
            text          = resp.choices[0].text or ""
            finish_reason = resp.choices[0].finish_reason or "stop"
            usage         = {
                "prompt_tokens":     resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens":      resp.usage.total_tokens,
            }

        return GenerateResponse(
            text          = text,
            model         = model,
            backend       = "openai_compat",
            request_id    = request.request_id,
            finish_reason = finish_reason,
            usage         = usage,
            latency_ms    = (time.perf_counter() - t0) * 1000,
        )

    async def agenerate_stream(self, request: GenerateRequest) -> AsyncIterator[str]:
        model  = request.model or self._model
        client = self._client()

        if request.messages:
            msgs   = [{"role": m.role, "content": m.content} for m in request.messages]
            stream = await client.chat.completions.create(
                model       = model,
                messages    = msgs,
                max_tokens  = request.max_tokens,
                temperature = request.temperature,
                top_p       = request.top_p,
                stop        = request.stop,
                stream      = True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        else:
            stream = await client.completions.create(
                model       = model,
                prompt      = request.flat_prompt(),
                max_tokens  = request.max_tokens,
                temperature = request.temperature,
                top_p       = request.top_p,
                stop        = request.stop,
                stream      = True,
            )
            async for chunk in stream:
                text = chunk.choices[0].text
                if text:
                    yield text

    async def aembed(self, request: EmbedRequest) -> EmbedResponse:
        client = self._client()
        model  = request.model or "text-embedding-3-small"
        resp   = await client.embeddings.create(input=request.texts, model=model)
        embeddings = [item.embedding for item in resp.data]
        return EmbedResponse(
            embeddings = embeddings,
            model      = model,
            backend    = "openai_compat",
            request_id = request.request_id,
            usage      = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "total_tokens":  resp.usage.total_tokens,
            },
        )

    async def ahealth(self) -> HealthStatus:
        try:
            client = self._client()
            models = await client.models.list()
            names  = [m.id for m in models.data]
            return HealthStatus(
                healthy = True,
                backend = "openai_compat",
                model   = self._model,
                message = f"Available: {names[:5]}{'...' if len(names) > 5 else ''}",
            )
        except Exception as exc:
            return HealthStatus(
                healthy = False,
                backend = "openai_compat",
                model   = self._model,
                message = str(exc),
            )

    # ── capabilities ─────────────────────────────────────────────────────

    def capabilities(self) -> ComputeCapability:
        return ComputeCapability(
            backend             = "openai_compat",
            models              = [self._model],
            max_context_length  = 128000,   # conservative; real limit is model-specific
            supports_streaming  = True,
            supports_embeddings = True,
            node_id             = self._node_id,
            extra               = {"base_url": self._base_url},
        )
