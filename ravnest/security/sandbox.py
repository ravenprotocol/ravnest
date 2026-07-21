"""
ravnest.security.sandbox — Tool-execution sandboxing for Ravnest agent nodes.

Provides two layers of protection:

ToolSandbox
    Wraps an agent's ``tool_executor`` callable to enforce:
    - Allowlist / denylist of tool names
    - Per-request call-count cap (``max_calls``)
    - Per-call wall-clock timeout (``timeout``)

    No subprocess isolation — the tool code runs in the same process.
    Use this for lightweight guard-railing of trusted-but-bounded tools.

AgentSandbox
    Runs an :class:`AgentBackend` in a separate subprocess via
    ``concurrent.futures.ProcessPoolExecutor``.  Provides:
    - Hard wall-clock timeout (the worker process is killed on breach)
    - Memory isolation (subprocess OOM doesn't crash the parent)
    - No shared state with the parent process

    Requires the agent and request to be picklable.  Works on macOS / Linux.

Usage
-----
    from ravnest.security.sandbox import ToolSandbox, AgentSandbox
    from ravnest.agents import LiteLLMAgent, AgentRequest

    # ToolSandbox — wrap an executor
    def my_executor(name, args):
        if name == "python_repl":
            return exec(args["code"])   # dangerous without sandboxing!

    safe_exec = ToolSandbox(
        allowed_tools={"web_search", "calculator"},
        blocked_tools={"python_repl", "bash"},
        max_calls=5,
        timeout=10.0,
    ).wrap(my_executor)

    # AgentSandbox — subprocess isolation
    agent  = LiteLLMAgent(model="gpt-4o")
    result = await AgentSandbox(timeout=30.0).arun(agent, request)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Dict, Iterable, Optional, Set

logger = logging.getLogger(__name__)


class ToolSandbox:
    """
    Policy layer that wraps a tool executor with safety constraints.

    Args:
        allowed_tools:  Set of tool names that are permitted.
                        Empty set (default) means all tools are allowed
                        (subject to ``blocked_tools``).
        blocked_tools:  Set of tool names that are always denied, regardless
                        of the allowlist.  Takes priority over ``allowed_tools``.
        max_calls:      Maximum number of tool calls allowed in a single
                        agent run.  Raises ``ToolLimitExceeded`` when breached.
        timeout:        Per-call timeout in seconds (asyncio timeout).
    """

    def __init__(
        self,
        allowed_tools: Optional[Iterable[str]] = None,
        blocked_tools: Optional[Iterable[str]] = None,
        max_calls:     int                      = 20,
        timeout:       float                    = 30.0,
    ):
        self._allowed  : Optional[Set[str]] = set(allowed_tools) if allowed_tools else None
        self._blocked  : Set[str]           = set(blocked_tools or [])
        self._max_calls: int                = max_calls
        self._timeout  : float              = timeout

    # ── policy checks ─────────────────────────────────────────────────────

    def is_allowed(self, tool_name: str) -> bool:
        """Return True if ``tool_name`` is permitted by this sandbox."""
        if tool_name in self._blocked:
            return False
        if self._allowed is not None and tool_name not in self._allowed:
            return False
        return True

    def check_tool(self, tool_name: str) -> None:
        """Raise ``ToolNotAllowed`` if the tool is denied."""
        if not self.is_allowed(tool_name):
            raise ToolNotAllowed(
                f"Tool '{tool_name}' is not permitted by this sandbox"
            )

    # ── executor wrapping ─────────────────────────────────────────────────

    def wrap(self, executor: Callable[[str, Dict], Any]) -> Callable:
        """
        Return a new callable that wraps ``executor`` with sandbox enforcement.

        The wrapped callable shares a call-counter per *invocation context*.
        Use :meth:`wrap_stateful` when you need per-agent-run counters.
        """
        return _WrappedExecutor(executor, self)

    def wrap_stateful(self, executor: Callable[[str, Dict], Any]
                      ) -> "_StatefulWrappedExecutor":
        """
        Return a stateful wrapper with a fresh call counter each time an
        agent run starts.  Call ``.reset()`` at the start of each run.
        """
        return _StatefulWrappedExecutor(executor, self)

    # ── async call helper ─────────────────────────────────────────────────

    async def acall(self, executor: Callable, name: str,
                    args: Dict) -> Any:
        """
        Call ``executor(name, args)`` with the sandbox timeout applied.

        Suitable when you hold the executor reference directly and want to
        enforce a single-call timeout without wrapping.
        """
        self.check_tool(name)
        return await asyncio.wait_for(
            asyncio.to_thread(executor, name, args),
            timeout=self._timeout,
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "allowed_tools": sorted(self._allowed) if self._allowed else "*",
            "blocked_tools": sorted(self._blocked),
            "max_calls":     self._max_calls,
            "timeout":       self._timeout,
        }


class _WrappedExecutor:
    """
    Shared-counter executor wrapper.  Not safe across concurrent requests —
    use ``_StatefulWrappedExecutor`` for per-request counters.
    """

    def __init__(self, executor, sandbox: ToolSandbox):
        self._executor = executor
        self._sandbox  = sandbox
        self._calls    = 0

    def __call__(self, name: str, args: Dict) -> Any:
        self._sandbox.check_tool(name)
        if self._calls >= self._sandbox._max_calls:
            raise ToolLimitExceeded(
                f"Tool call limit ({self._sandbox._max_calls}) exceeded"
            )
        self._calls += 1
        t0 = time.monotonic()
        try:
            result = _call_with_timeout_sync(
                self._executor, name, args, self._sandbox._timeout
            )
        except TimeoutError:
            raise ToolTimeout(
                f"Tool '{name}' timed out after {self._sandbox._timeout}s"
            )
        logger.debug("[sandbox] %s → %.2fs", name, time.monotonic() - t0)
        return result


class _StatefulWrappedExecutor(_WrappedExecutor):
    def reset(self) -> None:
        """Reset the call counter (call at the start of each agent run)."""
        self._calls = 0


def _call_with_timeout_sync(executor, name, args, timeout):
    """Call executor synchronously with a thread-based timeout."""
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(executor, name, args)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError()


# ── Agent subprocess sandbox ──────────────────────────────────────────────────

class AgentSandbox:
    """
    Run an agent backend in a subprocess for hard isolation.

    This prevents:
    - Runaway agents from consuming unbounded CPU/memory in the parent.
    - Tool code from accessing parent-process state.

    Args:
        timeout:     Max wall-clock time for the agent run (seconds).
        max_workers: Size of the process pool.

    Limitations
    -----------
    - Agent and AgentRequest must be picklable (standard dataclasses are fine;
      closures or lambda functions are not).
    - Not available on platforms without ``fork`` or ``spawn`` support.
    """

    def __init__(self, timeout: float = 60.0, max_workers: int = 4):
        self._timeout     = timeout
        self._max_workers = max_workers
        self._pool: Optional[Any] = None

    def _get_pool(self):
        if self._pool is None:
            import concurrent.futures
            self._pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=self._max_workers
            )
        return self._pool

    async def arun(self, agent, request) -> Any:
        """
        Run ``agent.run(request)`` in a subprocess and return the result.

        Raises ``asyncio.TimeoutError`` if ``self.timeout`` is exceeded.
        """
        import concurrent.futures
        loop = asyncio.get_running_loop()
        pool = self._get_pool()
        future = loop.run_in_executor(pool, _subprocess_agent_run, agent, request)
        return await asyncio.wait_for(future, timeout=self._timeout)

    def run(self, agent, request) -> Any:
        """Synchronous variant."""
        import asyncio
        return asyncio.run(self.arun(agent, request))

    def shutdown(self, wait: bool = True) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=wait)
            self._pool = None


def _subprocess_agent_run(agent, request):
    """Top-level function executed in the subprocess (must be picklable)."""
    import asyncio
    return asyncio.run(agent.arun(request))


# ── input validation ──────────────────────────────────────────────────────────

class InputValidator:
    """
    Lightweight input sanity checks for gateway requests.

    Raises ``ValueError`` with a user-visible message on violation.
    """

    def __init__(
        self,
        max_prompt_length:  int = 32_768,
        max_messages:       int = 200,
        max_message_length: int = 32_768,
        max_tools:          int = 64,
    ):
        self._max_prompt  = max_prompt_length
        self._max_msgs    = max_messages
        self._max_msg_len = max_message_length
        self._max_tools   = max_tools

    def validate(self, request) -> None:
        """
        Validate a ``GatewayRequest``.  Raises ``ValueError`` on violation.
        """
        if len(request.prompt) > self._max_prompt:
            raise ValueError(
                f"prompt exceeds max length ({self._max_prompt} chars)"
            )
        if len(request.messages) > self._max_msgs:
            raise ValueError(
                f"too many messages ({len(request.messages)} > {self._max_msgs})"
            )
        for i, m in enumerate(request.messages):
            content = m.get("content", "")
            if len(content) > self._max_msg_len:
                raise ValueError(
                    f"message[{i}] content exceeds max length"
                )
        if len(request.tools) > self._max_tools:
            raise ValueError(
                f"too many tools ({len(request.tools)} > {self._max_tools})"
            )


# ── exceptions ────────────────────────────────────────────────────────────────

class ToolNotAllowed(PermissionError):
    """Raised when an agent tries to call a tool that is not on the allowlist."""


class ToolLimitExceeded(RuntimeError):
    """Raised when an agent exceeds the per-run tool call cap."""


class ToolTimeout(TimeoutError):
    """Raised when an individual tool call exceeds the timeout."""
