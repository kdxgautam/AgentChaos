import pytest
from langchain.agents.middleware import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from agentchaos import ChaosMiddleware, FaultPolicy, TraceCollector


def request(call_id: str, tool_name: str = "get_order_status") -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={
            "name": tool_name,
            "args": {"order_id": "ORD-101"},
            "id": call_id,
            "type": "tool_call",
        },
        tool=None,
        state={},
        runtime=None,  # type: ignore[arg-type]
    )


def response(req: ToolCallRequest, *, error: bool = False) -> ToolMessage:
    return ToolMessage(
        content="original error" if error else "original success",
        name=req.tool_call["name"],
        tool_call_id=req.tool_call["id"],
        status="error" if error else "success",
    )


def events(collector: TraceCollector) -> list[tuple[str, str]]:
    return [(event.event, event.call_id) for event in collector.events]


def test_fault_policy_defaults_to_before_and_validates_stage() -> None:
    assert FaultPolicy("tool", "failure").stage == "before"

    with pytest.raises(ValueError, match="stage must be 'before' or 'after'"):
        FaultPolicy("tool", "failure", stage="invalid")  # type: ignore[arg-type]


def test_sync_after_fault_replaces_success_after_one_execution() -> None:
    collector = TraceCollector()
    middleware = ChaosMiddleware(
        FaultPolicy("get_order_status", "injected after execution", stage="after"),
        collector,
    )
    side_effects: list[str] = []

    def handler(req: ToolCallRequest) -> ToolMessage:
        side_effects.append(req.tool_call["id"])
        return response(req)

    result = middleware.wrap_tool_call(request("after-success"), handler)

    assert side_effects == ["after-success"]
    assert isinstance(result, ToolMessage)
    assert result.content == "injected after execution"
    assert result.status == "error"
    assert result.tool_call_id == "after-success"
    assert events(collector) == [
        ("start", "after-success"),
        ("success", "after-success"),
        ("response_replaced", "after-success"),
    ]


def test_sync_after_fault_preserves_tool_errors_commands_and_exceptions() -> None:
    collector = TraceCollector()
    middleware = ChaosMiddleware(
        FaultPolicy("get_order_status", "injected", stage="after"), collector
    )
    error_request = request("error")
    original_error = response(error_request, error=True)
    command = Command(update={"messages": []})
    exception = RuntimeError("boom")
    calls: list[str] = []

    def raise_original(req: ToolCallRequest) -> ToolMessage:
        calls.append(req.tool_call["id"])
        raise exception

    returned_error = middleware.wrap_tool_call(
        error_request, lambda req: calls.append(req.tool_call["id"]) or original_error
    )
    returned_command = middleware.wrap_tool_call(
        request("command"), lambda req: calls.append(req.tool_call["id"]) or command
    )
    with pytest.raises(RuntimeError) as raised:
        middleware.wrap_tool_call(request("exception"), raise_original)

    assert returned_error is original_error
    assert returned_command is command
    assert raised.value is exception
    assert calls == ["error", "command", "exception"]
    assert events(collector) == [
        ("start", "error"),
        ("returned_tool_error", "error"),
        ("start", "command"),
        ("success", "command"),
        ("start", "exception"),
        ("raised_exception", "exception"),
    ]
    assert collector.events[-1].exception_type == "RuntimeError"


def test_sync_after_fault_does_not_affect_nonmatching_tool() -> None:
    collector = TraceCollector()
    middleware = ChaosMiddleware(
        FaultPolicy("get_order_status", "injected", stage="after"), collector
    )
    original = response(request("other", "cancel_order"))

    result = middleware.wrap_tool_call(
        request("other", "cancel_order"), lambda _: original
    )

    assert result is original
    assert events(collector) == [("start", "other"), ("success", "other")]


@pytest.mark.asyncio
async def test_async_after_fault_has_sync_parity() -> None:
    collector = TraceCollector()
    middleware = ChaosMiddleware(
        FaultPolicy("get_order_status", "async injected", stage="after"), collector
    )
    calls: list[str] = []
    exception = LookupError("boom")

    async def succeed(req: ToolCallRequest) -> ToolMessage:
        calls.append(req.tool_call["id"])
        return response(req)

    async def return_error(req: ToolCallRequest) -> ToolMessage:
        calls.append(req.tool_call["id"])
        return response(req, error=True)

    async def return_command(req: ToolCallRequest) -> Command:
        calls.append(req.tool_call["id"])
        return Command(update={"messages": []})

    async def raise_original(req: ToolCallRequest) -> ToolMessage:
        calls.append(req.tool_call["id"])
        raise exception

    replaced = await middleware.awrap_tool_call(request("success"), succeed)
    preserved = await middleware.awrap_tool_call(request("error"), return_error)
    command = await middleware.awrap_tool_call(request("command"), return_command)
    nonmatching = await middleware.awrap_tool_call(
        request("other", "cancel_order"), succeed
    )
    with pytest.raises(LookupError) as raised:
        await middleware.awrap_tool_call(request("exception"), raise_original)

    assert isinstance(replaced, ToolMessage)
    assert replaced.status == "error"
    assert replaced.content == "async injected"
    assert replaced.tool_call_id == "success"
    assert preserved.status == "error"
    assert preserved.content == "original error"
    assert isinstance(command, Command)
    assert isinstance(nonmatching, ToolMessage)
    assert nonmatching.status == "success"
    assert raised.value is exception
    assert calls == ["success", "error", "command", "other", "exception"]
    assert events(collector) == [
        ("start", "success"),
        ("success", "success"),
        ("response_replaced", "success"),
        ("start", "error"),
        ("returned_tool_error", "error"),
        ("start", "command"),
        ("success", "command"),
        ("start", "other"),
        ("success", "other"),
        ("start", "exception"),
        ("raised_exception", "exception"),
    ]
