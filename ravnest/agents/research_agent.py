"""
ResearchAgent — web-search-capable ReAct agent.

The agent runs a multi-step ReAct (Reason + Act) loop:

  1. The LLM decides whether it needs to search the web.
  2. If yes, it emits a ``web_search`` tool call.
  3. The agent executes the search and feeds results back.
  4. The LLM synthesises a final answer.

Search providers (auto-detected, first available wins):
  - DuckDuckGo  (``pip install duckduckgo-search``)
  - SerpAPI     (set env var ``SERPAPI_KEY`` + ``pip install google-search-results``)
  - Brave       (set env var ``BRAVE_SEARCH_KEY`` + ``pip install httpx``)

The underlying LLM is any ``ComputeBackend`` (ravnest, vLLM, Ollama, OpenAI-compat…)
or a LiteLLM model string.

Usage
-----
    from ravnest.agents.research_agent import ResearchAgent
    from ravnest.agents.base import AgentRequest, Message

    agent = ResearchAgent(model="gpt-4o-mini")    # LiteLLM path
    resp  = agent.run(AgentRequest(
        messages = [Message("user", "Who won the 2024 Nobel Prize in Physics?")],
        max_steps = 5,
    ))
    print(resp.text)
"""

from __future__ import annotations

import json
import os
import socket
import time
from typing import AsyncIterator, List, Optional

from .base import (
    AgentBackend, AgentCapability, AgentHealthStatus, AgentRequest,
    AgentResponse, Message, ToolCall,
)

# ── Tool schema ───────────────────────────────────────────────────────────────

_WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name":        "web_search",
        "description": "Search the web for up-to-date information. "
                       "Use when you need current facts, news, or data.",
        "parameters": {
            "type":       "object",
            "properties": {
                "query": {
                    "type":        "string",
                    "description": "The search query string.",
                },
                "num_results": {
                    "type":        "integer",
                    "description": "Number of results to return (default 5, max 10).",
                    "default":     5,
                },
            },
            "required": ["query"],
        },
    },
}

_SYSTEM_PROMPT = (
    "You are a helpful research assistant with access to web search.\n"
    "When you need current information, call the web_search tool.\n"
    "After gathering enough information, provide a clear, well-sourced answer.\n"
    "Always cite your sources by mentioning the URLs you found."
)


class ResearchAgent(AgentBackend):
    """
    Web-search-capable ReAct agent.

    Args:
        model:        LiteLLM model string OR a ComputeBackend instance.
                      When a string is supplied, LiteLLM is used.
        api_base:     Override API base URL (LiteLLM path only).
        api_key:      API key override (LiteLLM path only).
        max_results:  Default number of search results per query.
        node_id:      Registry node_id override.
        system_prompt: Override the default system prompt.
    """

    def __init__(
        self,
        model:         str | object,
        api_base:      Optional[str] = None,
        api_key:       Optional[str] = None,
        max_results:   int           = 5,
        node_id:       Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        self._model_spec   = model  # str → LiteLLM; ComputeBackend → direct
        self._api_base     = api_base
        self._api_key      = api_key
        self._max_results  = max_results
        self._node_id      = node_id or f"research_{socket.gethostname()}"
        self._system_prompt = system_prompt or _SYSTEM_PROMPT

        # Determine model label for metadata
        if isinstance(model, str):
            self._model_label = model
        else:
            self._model_label = getattr(model, "_model", "compute_backend")

    # ── async interface ───────────────────────────────────────────────────

    async def arun(self, request: AgentRequest) -> AgentResponse:
        t0 = time.perf_counter()

        # Inject system prompt if not already present
        messages = list(request.messages)
        if not messages or messages[0].role != "system":
            messages.insert(0, Message("system", self._system_prompt))

        # Build LiteLLM-style message list
        llm_msgs = [{"role": m.role, "content": m.content} for m in messages]
        tools    = [_WEB_SEARCH_TOOL]
        steps    = 0
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        model = request.model or (self._model_spec
                                  if isinstance(self._model_spec, str)
                                  else self._model_label)

        for _ in range(request.max_steps):
            steps += 1
            llm_resp = await self._llm_call(
                messages    = llm_msgs,
                tools       = tools,
                model       = model,
                max_tokens  = request.max_tokens,
                temperature = request.temperature,
            )
            _accum_usage(total_usage, llm_resp.get("usage"))

            finish_reason = llm_resp.get("finish_reason", "stop")
            content       = llm_resp.get("content", "") or ""
            raw_tool_calls = llm_resp.get("tool_calls") or []

            # ── tool execution ────────────────────────────────────────────
            if finish_reason == "tool_calls" and raw_tool_calls:
                # Append assistant turn
                llm_msgs.append({
                    "role":       "assistant",
                    "content":    content,
                    "tool_calls": raw_tool_calls,
                })
                for tc_raw in raw_tool_calls:
                    fn       = tc_raw["function"]
                    args_str = fn.get("arguments", "{}")
                    try:
                        args = json.loads(args_str)
                    except json.JSONDecodeError:
                        args = {}
                    tc = ToolCall(
                        id   = tc_raw["id"],
                        name = fn["name"],
                        args = args,
                        raw  = args_str,
                    )

                    if tc.name == "web_search":
                        query       = tc.args.get("query", "")
                        num_results = tc.args.get("num_results", self._max_results)
                        result      = await self._search(query, num_results)
                    else:
                        result = f"Unknown tool: {tc.name}"

                    llm_msgs.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "content":      result,
                    })
                continue  # next step

            # ── final answer ──────────────────────────────────────────────
            return AgentResponse(
                text          = content.strip(),
                agent         = "research",
                model         = model,
                request_id    = request.request_id,
                finish_reason = finish_reason,
                steps         = steps,
                usage         = total_usage,
                latency_ms    = (time.perf_counter() - t0) * 1000,
                metadata      = {"searched": steps > 1},
            )

        return AgentResponse(
            text          = "",
            agent         = "research",
            model         = model,
            request_id    = request.request_id,
            finish_reason = "max_steps",
            steps         = steps,
            usage         = total_usage,
            latency_ms    = (time.perf_counter() - t0) * 1000,
        )

    async def astream(self, request: AgentRequest) -> AsyncIterator[str]:
        """Non-streaming fallback — yields the full answer as one chunk."""
        resp = await self.arun(request)
        if resp.text:
            yield resp.text

    async def ahealth(self) -> AgentHealthStatus:
        try:
            provider = await _detect_search_provider()
            model    = (self._model_spec if isinstance(self._model_spec, str)
                        else self._model_label)
            return AgentHealthStatus(
                healthy = True,
                agent   = "research",
                model   = model,
                message = f"Search provider: {provider}",
            )
        except Exception as exc:
            return AgentHealthStatus(
                healthy = False,
                agent   = "research",
                model   = self._model_label,
                message = str(exc),
            )

    def capabilities(self) -> AgentCapability:
        model = (self._model_spec if isinstance(self._model_spec, str)
                 else self._model_label)
        return AgentCapability(
            agent_type         = "research",
            models             = [model],
            tools              = ["web_search"],
            supports_streaming = False,
            node_id            = self._node_id,
            extra              = {"address": f"{socket.gethostname()}:0"},
        )

    # ── private helpers ───────────────────────────────────────────────────

    async def _llm_call(self, messages, tools, model, max_tokens, temperature) -> dict:
        """Call LiteLLM (or the ComputeBackend) and return a normalised dict."""
        if isinstance(self._model_spec, str):
            # LiteLLM path
            import litellm
            kwargs: dict = dict(
                model       = model,
                messages    = messages,
                max_tokens  = max_tokens,
                temperature = temperature,
                tools       = tools,
            )
            if self._api_base:
                kwargs["api_base"] = self._api_base
            if self._api_key:
                kwargs["api_key"]  = self._api_key
            resp    = await litellm.acompletion(**kwargs)
            choice  = resp.choices[0]
            message = choice.message
            return {
                "finish_reason": choice.finish_reason,
                "content":       message.content,
                "tool_calls":    [
                    {
                        "id":   tc.id,
                        "type": "function",
                        "function": {
                            "name":      tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in (message.tool_calls or [])
                ],
                "usage": resp.usage,
            }

        # ComputeBackend path — no native tool calling, use prompt injection
        from ravnest.compute.base import GenerateRequest, Message as CMsg
        prompt_msgs = [CMsg(role=m["role"], content=m["content"])
                       for m in messages if "content" in m]
        req  = GenerateRequest(messages=prompt_msgs, max_tokens=max_tokens,
                               temperature=temperature)
        resp = await self._model_spec.agenerate(req)
        return {
            "finish_reason": resp.finish_reason,
            "content":       resp.text,
            "tool_calls":    [],
            "usage":         resp.usage,
        }

    async def _search(self, query: str, num_results: int) -> str:
        """Execute a web search; returns a formatted result string."""
        results = await _web_search(query, num_results)
        if not results:
            return f"No results found for: {query}"
        lines = [f"Search results for '{query}':\n"]
        for i, r in enumerate(results, 1):
            title   = r.get("title",   "No title")
            url     = r.get("url",     r.get("href", ""))
            snippet = r.get("body",    r.get("snippet", r.get("description", "")))
            lines.append(f"{i}. {title}\n   {url}\n   {snippet}\n")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Search provider implementations
# ─────────────────────────────────────────────────────────────────────────────

async def _detect_search_provider() -> str:
    """Return the name of the first available search provider."""
    try:
        from duckduckgo_search import AsyncDDGS  # noqa: F401
        return "duckduckgo"
    except ImportError:
        pass
    if os.getenv("SERPAPI_KEY"):
        try:
            from serpapi import GoogleSearch  # noqa: F401
            return "serpapi"
        except ImportError:
            pass
    if os.getenv("BRAVE_SEARCH_KEY"):
        return "brave"
    raise RuntimeError(
        "No search provider available. Install one of:\n"
        "  pip install duckduckgo-search          (free, no key)\n"
        "  pip install google-search-results       (SerpAPI, needs SERPAPI_KEY)\n"
        "  pip install httpx + set BRAVE_SEARCH_KEY (Brave Search API)"
    )


async def _web_search(query: str, num_results: int = 5) -> List[dict]:
    """Try each search provider in order and return result dicts."""
    # ── DuckDuckGo (preferred — free, no API key) ────────────────────────
    try:
        from duckduckgo_search import AsyncDDGS
        async with AsyncDDGS() as ddgs:
            results = await ddgs.atext(query, max_results=num_results)
        return list(results) if results else []
    except ImportError:
        pass
    except Exception:
        pass  # DuckDuckGo failed, try next

    # ── SerpAPI ──────────────────────────────────────────────────────────
    key = os.getenv("SERPAPI_KEY")
    if key:
        try:
            from serpapi import GoogleSearch
            import asyncio
            loop    = asyncio.get_event_loop()
            search  = GoogleSearch({"q": query, "num": num_results, "api_key": key})
            results = await loop.run_in_executor(None, search.get_dict)
            organic = results.get("organic_results", [])
            return [
                {"title": r.get("title"), "url": r.get("link"), "snippet": r.get("snippet")}
                for r in organic[:num_results]
            ]
        except ImportError:
            pass
        except Exception:
            pass

    # ── Brave Search ─────────────────────────────────────────────────────
    brave_key = os.getenv("BRAVE_SEARCH_KEY")
    if brave_key:
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": query, "count": num_results},
                    headers={"Accept": "application/json",
                             "X-Subscription-Token": brave_key},
                    timeout=10.0,
                )
                resp.raise_for_status()
                data = resp.json()
                return [
                    {"title": r.get("title"), "url": r.get("url"),
                     "snippet": r.get("description", "")}
                    for r in data.get("web", {}).get("results", [])[:num_results]
                ]
        except Exception:
            pass

    return []


def _accum_usage(total: dict, usage) -> None:
    if usage is None:
        return
    total["prompt_tokens"]     += getattr(usage, "prompt_tokens",     0) or 0
    total["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
    total["total_tokens"]      += getattr(usage, "total_tokens",      0) or 0
