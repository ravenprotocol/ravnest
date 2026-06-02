"""
ravnest.agents — Agent node framework.

Every node that participates as an "agent" in the Ravnest network implements
``AgentBackend``.  The router discovers agent nodes from the registry and
dispatches ``AgentRequest`` objects to the best available one.

Quick-start
-----------
>>> from ravnest.agents import (
...     AgentRequest, AgentResponse, Message,
...     LiteLLMAgent, ResearchAgent, SQLAgent,
...     AgentRouter,
... )
>>>
>>> # Single agent
>>> agent = LiteLLMAgent(model="gpt-4o-mini")
>>> resp  = agent.run(AgentRequest(messages=[Message("user", "Hi!")]))
>>> print(resp.text)
>>>
>>> # Routed (requires a running registry + registered agent nodes)
>>> router = AgentRouter(registry_address="localhost:50099")
>>> resp   = router.run(AgentRequest(
...     messages = [Message("user", "Search the web for…")],
...     extra    = {"agent_type": "research"},
... ))

Agent types
-----------
- LiteLLMAgent   — any model via LiteLLM (OpenAI, Anthropic, Ollama, …)
- ResearchAgent  — web-search ReAct agent
- SQLAgent       — natural-language-to-SQL agent

Router strategies
-----------------
- LoadBasedAgentStrategy   — route to least-loaded node (default)
- RoundRobinAgentStrategy  — distribute evenly
- AgentTypeStrategy        — prefer exact agent_type match, then load-based
"""

from .base import (
    AgentBackend,
    AgentCapability,
    AgentHealthStatus,
    AgentRequest,
    AgentResponse,
    Message,
    ToolCall,
    ToolResult,
)

from .litellm_agent  import LiteLLMAgent
from .research_agent import ResearchAgent
from .sql_agent      import SQLAgent

from .router import (
    AgentRouter,
    AgentRoutingStrategy,
    AgentTypeStrategy,
    LoadBasedAgentStrategy,
    RoundRobinAgentStrategy,
)

__all__ = [
    # ── data classes ──────────────────────────────────────────────────────
    "Message",
    "ToolCall",
    "ToolResult",
    "AgentRequest",
    "AgentResponse",
    "AgentHealthStatus",
    "AgentCapability",
    # ── abstract base ─────────────────────────────────────────────────────
    "AgentBackend",
    # ── concrete agents ───────────────────────────────────────────────────
    "LiteLLMAgent",
    "ResearchAgent",
    "SQLAgent",
    # ── router + strategies ───────────────────────────────────────────────
    "AgentRouter",
    "AgentRoutingStrategy",
    "LoadBasedAgentStrategy",
    "RoundRobinAgentStrategy",
    "AgentTypeStrategy",
]
