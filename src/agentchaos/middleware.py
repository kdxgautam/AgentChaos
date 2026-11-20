"""Tool-call fault injection middleware."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from agentchaos.tracing import TraceCollector, TraceEventType


@dataclass(frozen=True, slots=True)
class FaultPolicy:
    """A deterministic tool failure."""

    tool_name: str
    error_message: str


class ChaosMiddleware(AgentMiddleware):
    """Fail calls matching an optional fault policy before tool execution."""

    def __init__(
        self,
        policy: FaultPolicy | None = None,
        collector: TraceCollector | None = None,
    ) -> None:
        self.policy = policy
        self.collector = collector

    def _trace(
        self,
        request: ToolCallRequest,
        event: TraceEventType,
        exception: Exception | None = None,
    ) -> None:
        if self.collector is not None:
            self.collector._record(
                event=event,
                tool_name=request.tool_call["name"],
                call_id=request.tool_call["id"],
                exception_type=type(exception).__name__ if exception else None,
            )

    def _trace_result(
        self,
        request: ToolCallRequest,
        result: ToolMessage | Command[Any],
    ) -> None:
        event = (
            "returned_tool_error"
            if isinstance(result, ToolMessage) and result.status == "error"
            else "success"
        )
        self._trace(request, event)

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
        self._trace(request, "start")
        if failure := self._failure(request):
            self._trace(request, "injected_failure")
            return failure
        try:
            result = handler(request)
        except Exception as exception:
            self._trace(request, "raised_exception", exception)
            raise
        self._trace_result(request, result)
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[
            [ToolCallRequest], Awaitable[ToolMessage | Command[Any]]
        ],
    ) -> ToolMessage | Command[Any]:
        self._trace(request, "start")
        if failure := self._failure(request):
            self._trace(request, "injected_failure")
            return failure
        try:
            result = await handler(request)
        except Exception as exception:
            self._trace(request, "raised_exception", exception)
            raise
        self._trace_result(request, result)
        return result
