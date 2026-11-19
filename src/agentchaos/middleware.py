"""Tool-call fault injection middleware."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command


@dataclass(frozen=True, slots=True)
class FaultPolicy:
    """A deterministic tool failure."""

    tool_name: str
    error_message: str


class ChaosMiddleware(AgentMiddleware):
    """Fail calls matching an optional fault policy before tool execution."""

    def __init__(self, policy: FaultPolicy | None = None) -> None:
        self.policy = policy

    def _failure(self, request: ToolCallRequest) -> ToolMessage | None:
        if self.policy is None or request.tool_call["name"] != self.policy.tool_name:
            return None
        return ToolMessage(
            content=self.policy.error_message,
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
            status="error",
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        if failure := self._failure(request):
            return failure
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[
            [ToolCallRequest], Awaitable[ToolMessage | Command[Any]]
        ],
    ) -> ToolMessage | Command[Any]:
        if failure := self._failure(request):
            return failure
        return await handler(request)
