"""Tool-call fault injection middleware."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from threading import Lock
from typing import Any, Literal

from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from agentchaos.tracing import TraceCollector, TraceEventType


@dataclass(frozen=True, slots=True)
class FaultPolicy:
    """A deterministic tool failure."""

    tool_name: str
    error_message: str
    stage: Literal["before", "after"] = "before"
    on_call: int | None = None

    def __post_init__(self) -> None:
        if self.stage not in {"before", "after"}:
            raise ValueError("stage must be 'before' or 'after'")
        if self.on_call is not None and (
            isinstance(self.on_call, bool)
            or not isinstance(self.on_call, int)
            or self.on_call < 1
        ):
            raise ValueError("on_call must be a positive integer or None")


class ChaosMiddleware(AgentMiddleware):
    """Fail matching tool calls before execution or replace successful results."""

    def __init__(
        self,
        policy: FaultPolicy | None = None,
        collector: TraceCollector | None = None,
    ) -> None:
        self.policy = policy
        self.collector = collector
        self._matching_calls = 0
        self._counter_lock = Lock()

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

    def _should_inject(self, request: ToolCallRequest) -> bool:
        if self.policy is None or request.tool_call["name"] != self.policy.tool_name:
            return False
        with self._counter_lock:
            self._matching_calls += 1
            attempt = self._matching_calls
        return self.policy.on_call is None or attempt == self.policy.on_call

    def _failure(self, request: ToolCallRequest) -> ToolMessage:
        assert self.policy is not None
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
        inject = self._should_inject(request)
        if inject and self.policy is not None and self.policy.stage == "before":
            self._trace(request, "injected_failure")
            return self._failure(request)
        try:
            result = handler(request)
        except Exception as exception:
            self._trace(request, "raised_exception", exception)
            raise
        self._trace_result(request, result)
        if (
            inject
            and self.policy is not None
            and self.policy.stage == "after"
            and isinstance(result, ToolMessage)
            and result.status == "success"
        ):
            self._trace(request, "response_replaced")
            return self._failure(request)
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[
            [ToolCallRequest], Awaitable[ToolMessage | Command[Any]]
        ],
    ) -> ToolMessage | Command[Any]:
        self._trace(request, "start")
        inject = self._should_inject(request)
        if inject and self.policy is not None and self.policy.stage == "before":
            self._trace(request, "injected_failure")
            return self._failure(request)
        try:
            result = await handler(request)
        except Exception as exception:
            self._trace(request, "raised_exception", exception)
            raise
        self._trace_result(request, result)
        if (
            inject
            and self.policy is not None
            and self.policy.stage == "after"
            and isinstance(result, ToolMessage)
            and result.status == "success"
        ):
            self._trace(request, "response_replaced")
            return self._failure(request)
        return result
