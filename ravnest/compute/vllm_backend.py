"""
VLLMBackend — wraps vLLM's AsyncLLMEngine.

Install dependency:  pip install vllm

Usage:
    backend = VLLMBackend(
        model="meta-llama/Llama-3.1-8B-Instruct",
        tensor_parallel_size=1,
        dtype="float16",
    )
    backend.register_with_registry("registry_host:50099")

    resp = backend.generate(GenerateRequest(prompt="Hello!", max_tokens=64))
    print(resp.text)
"""

from __future__ import annotations

import socket
import time
import uuid
from typing import AsyncIterator, Optional

from .base import (
    ComputeBackend, ComputeCapability, EmbedRequest, EmbedResponse,
    GenerateRequest, GenerateResponse, HealthStatus,
)


class VLLMBackend(ComputeBackend):
    """
    ComputeBackend powered by vLLM's AsyncLLMEngine.

    Args:
        model:                HuggingFace model id or local path.
        tensor_parallel_size: Number of GPUs for tensor parallelism (default 1).
        dtype:                Model dtype — "auto", "float16", "bfloat16", etc.
        max_model_len:        Override the model's max context length.
        gpu_memory_utilization: Fraction of GPU VRAM to use (default 0.90).
        node_id:              Registry node_id override.
        extra_engine_args:    Dict of additional AsyncEngineArgs kwargs.
    """

    def __init__(
        self,
        model:                    str,
        tensor_parallel_size:     int   = 1,
        dtype:                    str   = "auto",
        max_model_len:            Optional[int] = None,
        gpu_memory_utilization:   float = 0.90,
        node_id:                  Optional[str] = None,
        extra_engine_args:        dict  = None,
    ):
        try:
            from vllm import AsyncLLMEngine, AsyncEngineArgs
        except ImportError:
            raise ImportError(
                "vllm is not installed. Run: pip install vllm"
            )

        engine_kwargs = dict(
            model                  = model,
            tensor_parallel_size   = tensor_parallel_size,
            dtype                  = dtype,
            gpu_memory_utilization = gpu_memory_utilization,
        )
        if max_model_len is not None:
            engine_kwargs["max_model_len"] = max_model_len
        if extra_engine_args:
            engine_kwargs.update(extra_engine_args)

        self._model    = model
        self._engine   = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**engine_kwargs))
        self._node_id  = node_id or f"vllm_{socket.gethostname()}"
        self._address  = f"{socket.gethostname()}:8000"

        # Pre-fetch context length for capability reporting
        self._max_ctx  = max_model_len or 4096

    # ── async native ─────────────────────────────────────────────────────

    async def agenerate(self, request: GenerateRequest) -> GenerateResponse:
        from vllm import SamplingParams
        t0      = time.perf_counter()
        prompt  = request.flat_prompt()
        req_id  = request.request_id or str(uuid.uuid4())
        params  = SamplingParams(
            max_tokens  = request.max_tokens,
            temperature = request.temperature,
            top_p       = request.top_p,
            top_k       = request.top_k,
            stop        = request.stop or [],
        )
        final_output = None
        async for output in self._engine.generate(prompt, params, request_id=req_id):
            final_output = output

        text           = final_output.outputs[0].text
        finish_reason  = final_output.outputs[0].finish_reason or "stop"
        prompt_tokens  = len(final_output.prompt_token_ids)
        output_tokens  = len(final_output.outputs[0].token_ids)

        return GenerateResponse(
            text          = text,
            model         = self._model,
            backend       = "vllm",
            request_id    = req_id,
            finish_reason = finish_reason,
            usage         = {
                "prompt_tokens":     prompt_tokens,
                "completion_tokens": output_tokens,
                "total_tokens":      prompt_tokens + output_tokens,
            },
            latency_ms    = (time.perf_counter() - t0) * 1000,
        )

    async def agenerate_stream(self, request: GenerateRequest) -> AsyncIterator[str]:
        from vllm import SamplingParams
        prompt = request.flat_prompt()
        req_id = request.request_id or str(uuid.uuid4())
        params = SamplingParams(
            max_tokens  = request.max_tokens,
            temperature = request.temperature,
            top_p       = request.top_p,
            top_k       = request.top_k,
            stop        = request.stop or [],
        )
        prev_text = ""
        async for output in self._engine.generate(prompt, params, request_id=req_id):
            new_text = output.outputs[0].text
            delta    = new_text[len(prev_text):]
            prev_text = new_text
            if delta:
                yield delta

    async def aembed(self, request: EmbedRequest) -> EmbedResponse:
        raise NotImplementedError(
            "VLLMBackend does not support embeddings. "
            "Use a dedicated embedding model or OpenAICompatBackend."
        )

    async def ahealth(self) -> HealthStatus:
        try:
            # vLLM engine is healthy if it's running
            is_healthy = not self._engine.errored
            return HealthStatus(healthy=is_healthy, backend="vllm", model=self._model)
        except Exception as exc:
            return HealthStatus(healthy=False, backend="vllm", model=self._model, message=str(exc))

    # ── capabilities ─────────────────────────────────────────────────────

    def capabilities(self) -> ComputeCapability:
        return ComputeCapability(
            backend             = "vllm",
            models              = [self._model],
            max_context_length  = self._max_ctx,
            supports_streaming  = True,
            supports_embeddings = False,
            node_id             = self._node_id,
        )
