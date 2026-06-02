"""
LiteLLMAgent — any model via LiteLLM's unified interface.

LiteLLM proxies to OpenAI, Anthropic, Cohere, Together AI, Groq, Mistral,
Ollama, vLLM, and 100+ other providers through a single API.

Install:  pip install litellm

Usage
-----
    from ravnest.agents.litellm_agent import LiteLLMAgent
    from ravnest.agents.base import AgentRequest, Message

    # OpenAI
    agent = LiteLLMAgent(model="gpt-4o-mini")

    # Anthropic
    agent = LiteLLMAgent(model="claude-3-haiku-20240307")

    # Local Ollama
    agent = LiteLLMAgent(model="ollama/llama3.2", api_base="http://localhost:11434")

    resp = agent.run(AgentRequest(
        messages=[Message("user", "What is 2 + 2?")],
    ))
    print(resp.text)

Tool use
--------
Pass OpenAI-style tool schemas in ``AgentRequest.tools``.  The agent will
run a ReAct loop, executing tool calls via the optional ``tool_executor``
callback you supply at construction time.

    def my_executor(name: str, args: dict) -> str:
        if name == "add":
            return str(args["a"] + args["b"])
        return "unknown tool"

    agent = LiteLLMAgent(model="gpt-4o-mini", tool_executor=my_executor)
"""

from __future__ import annotations

import json
import socket
import time
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from .base import (
    AgentBackend, AgentCapability, AgentHealthStatus, AgentRequest,
    AgentResponse, Message, ToolCall, ToolResult,
)

_LITELLM_ERR = "litellm is not installed. Run: pip install litellm"


class LiteLLMAgent(AgentBackend):
    """
    AgentBackend powered by LiteLLM.

    Args:
        model:           LiteLLM model string, e.g. "gpt-4o", "claude-3-haiku-20240307",
                         "ollama/llama3.2", "together_ai/meta-llama/Llama-3-8b-chat-hf".
        api_base:        Override the provider's base URL (useful for local servers).
        api_key:         API key (falls back to the relevant env variable).
        default_tools:   Tool schemas (OpenAI format) always injected into every call.
        tool_executor:   Callable(name, args) → str  that handles tool calls.
                         If None, tool calls are returned to the caller unfulfilled.
        node_id:         Registry node_id override.
        timeout:         Per-LLM-call timeout in seconds.
        extra_kwargs:    Passed verbatim to every litellm.acompletion() call.
    """

    def __init__(
        self,
        model:          str,
        api_base:       Optional[str]                        = None,
        api_key:        Optional[str]                        = None,
        default_tools:  List[Dict[str, Any]]                 = None,
        tool_executor:  Optional[Callable[[str, dict], str]] = None,
        node_id:        Optional[str]                        = None,
        timeout:        float                                = 60.0,
        extra_kwargs:   Dict[str, Any]                       = None,
    ):
        try:
            import litellm  # noqa: F401
        except ImportError:
            raise ImportError(_LITELLM_ERR)

        self._model         = model
        self._api_base      = api_base
        self._api_key       = api_key
        self._default_tools = default_tools or []
        self._tool_executor = tool_executor
        self._node_id       = node_id or f"litellm_{socket.gethostname()}"
        self._timeout       = timeout
        self._extra         = extra_kwargs or {}

    # ── async interface ───────────────────────────────────────────────────

    async def arun(self, request: AgentRequest) -> AgentResponse:
        import litellm

        t0       = time.perf_counter()
        model    = request.model or self._model
        messages = self._build_messages(request)
        tools    = (self._default_tools + request.tools) or None
        steps    = 0
        total_usage: Dict[str, int] = {"prompt_tokens": 0,
                                       "completion_tokens": 0, "total_tokens": 0}

        for _ in range(request.max_steps):
            steps += 1
            kwargs = dict(
                model       = model,
                messages    = messages,
                max_tokens  = request.max_tokens,
                temperature = request.temperature,
                timeout     = self._timeout,
                **self._extra,
            )
            if tools:
                kwargs["tools"] = tools
            if self._api_base:
                kwargs["api_base"] = self._api_base
            if self._api_key:
                kwargs["api_key"] = self._api_key

            resp = await litellm.acompletion(**kwargs)
            _accum_usage(total_usage, resp.usage)

            choice  = resp.choices[0]
            message = choice.message

            # ── tool calls ────────────────────────────────────────────────
            if choice.finish_reason == "tool_calls" and message.tool_calls:
                tool_calls = _parse_tool_calls(message.tool_calls)

                # Append assistant turn with tool calls
                messages.append({
                    "role":       "assistant",
                    "content":    message.content or "",
                    "tool_calls": [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.name, "arguments": tc.raw}}
                        for tc in tool_calls
                    ],
                })

                if self._tool_executor:
                    # Execute tools and continue
                    for tc in tool_calls:
                        try:
                            result = self._tool_executor(tc.name, tc.args)
                            is_error = False
                        except Exception as exc:
                            result   = f"Error: {exc}"
                            is_error = True
                        messages.append({
                            "role":         "tool",
                            "tool_call_id": tc.id,
                            "content":      result,
                        })
                    continue  # next ReAct step

                # No executor — return tool calls to caller
                return AgentResponse(
                    text          = message.content or "",
                    agent         = "litellm",
                    model         = model,
                    request_id    = request.request_id,
                    finish_reason = "tool_calls",
                    tool_calls    = tool_calls,
                    steps         = steps,
                    usage         = total_usage,
                    latency_ms    = (time.perf_counter() - t0) * 1000,
                )

            # ── final answer ──────────────────────────────────────────────
            text = (message.content or "").strip()
            return AgentResponse(
                text          = text,
                agent         = "litellm",
                model         = model,
                request_id    = request.request_id,
                finish_reason = choice.finish_reason or "stop",
                steps         = steps,
                usage         = total_usage,
                latency_ms    = (time.perf_counter() - t0) * 1000,
            )

        # Reached max_steps
        return AgentResponse(
            text          = "",
            agent         = "litellm",
            model         = model,
            request_id    = request.request_id,
            finish_reason = "max_steps",
            steps         = steps,
            usage         = total_usage,
            latency_ms    = (time.perf_counter() - t0) * 1000,
        )

    async def astream(self, request: AgentRequest) -> AsyncIterator[str]:
        import litellm

        model    = request.model or self._model
        messages = self._build_messages(request)
        kwargs   = dict(
            model       = model,
            messages    = messages,
            max_tokens  = request.max_tokens,
            temperature = request.temperature,
            stream      = True,
            timeout     = self._timeout,
            **self._extra,
        )
        if self._api_base:
            kwargs["api_base"] = self._api_base
        if self._api_key:
            kwargs["api_key"]  = self._api_key

        resp = await litellm.acompletion(**kwargs)
        async for chunk in resp:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    async def ahealth(self) -> AgentHealthStatus:
        try:
            import litellm
            # Use a minimal completion to verify the model is reachable
            resp = await litellm.acompletion(
                model      = self._model,
                messages   = [{"role": "user", "content": "ping"}],
                max_tokens = 1,
                timeout    = 5.0,
                **({"api_base": self._api_base} if self._api_base else {}),
                **({"api_key":  self._api_key}  if self._api_key  else {}),
            )
            return AgentHealthStatus(
                healthy = True,
                agent   = "litellm",
                model   = self._model,
                message = f"Model {self._model} reachable",
            )
        except Exception as exc:
            return AgentHealthStatus(
                healthy = False,
                agent   = "litellm",
                model   = self._model,
                message = str(exc),
            )

    # ── capabilities ─────────────────────────────────────────────────────

    def capabilities(self) -> AgentCapability:
        return AgentCapability(
            agent_type         = "litellm",
            models             = [self._model],
            tools              = [t["function"]["name"]
                                  for t in self._default_tools
                                  if "function" in t],
            supports_streaming = True,
            node_id            = self._node_id,
            extra              = {
                "api_base": self._api_base or "",
                "address":  f"{socket.gethostname()}:0",
            },
        )

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _build_messages(request: AgentRequest) -> List[Dict[str, Any]]:
        """Convert AgentRequest messages + tool results into LiteLLM format."""
        msgs: List[Dict[str, Any]] = []
        for m in request.messages:
            msgs.append({"role": m.role, "content": m.content})
        # Append any tool results from a previous step
        for tr in request.tool_results:
            msgs.append({
                "role":         "tool",
                "tool_call_id": tr.tool_call_id,
                "content":      tr.content,
            })
        return msgs


# ─────────────────────────────────────────────────────────────────────────────
# Internal utilities
# ─────────────────────────────────────────────────────────────────────────────

def _parse_tool_calls(raw_calls) -> List[ToolCall]:
    result = []
    for tc in raw_calls:
        fn   = tc.function
        args_str = fn.arguments or "{}"
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            args = {}
        result.append(ToolCall(id=tc.id, name=fn.name, args=args, raw=args_str))
    return result


def _accum_usage(total: Dict[str, int], usage) -> None:
    if usage is None:
        return
    total["prompt_tokens"]     += getattr(usage, "prompt_tokens",     0) or 0
    total["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
    total["total_tokens"]      += getattr(usage, "total_tokens",      0) or 0
