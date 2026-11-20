from datetime import UTC

import pytest
from langchain.agents.middleware import ToolCallRequest
from langchain_core.messages import ToolMessage

from agentchaos import ChaosMiddleware, FaultPolicy, TraceCollector


def request(call_id: str) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={
            "name": "get_order_status",
            "args": {"order_id": "ORD-101"},
            "id": call_id,
            "type": "tool_call",
        },
        tool=None,
        state={},
        runtime=None,  # type: ignore[arg-type]
    )


def response(request: ToolCallRequest, *, error: bool = False) -> ToolMessage:
    return ToolMessage(
        content="tool error" if error else "ok",
        name=request.tool_call["name"],
        tool_call_id=request.tool_call["id"],
        status="error" if error else "success",
    )


def assert_common_fields(collector: TraceCollector) -> None:
    assert all(event.tool_name == "get_order_status" for event in collector.events)
    assert all(event.timestamp.tzinfo is UTC for event in collector.events)


def test_sync_trace_outcomes_and_exception_propagation() -> None:
    collector = TraceCollector()
    middleware = ChaosMiddleware(collector=collector)

    middleware.wrap_tool_call(request("success"), response)
    middleware.wrap_tool_call(request("tool-error"), lambda req: response(req, error=True))

    injected = ChaosMiddleware(
        policy=FaultPolicy("get_order_status", "injected"),
        collector=collector,
    )
    injected.wrap_tool_call(
        request("injected"),
        lambda _: pytest.fail("injected call reached the original handler"),
    )

    exception = ValueError("boom")

    def raise_exception(_: ToolCallRequest) -> ToolMessage:
        raise exception

    with pytest.raises(ValueError) as raised:
        middleware.wrap_tool_call(request("exception"), raise_exception)

    assert raised.value is exception
    assert [
        (event.event, event.call_id, event.exception_type)
        for event in collector.events
    ] == [
        ("start", "success", None),
        ("success", "success", None),
        ("start", "tool-error", None),
        ("returned_tool_error", "tool-error", None),
        ("start", "injected", None),
        ("injected_failure", "injected", None),
        ("start", "exception", None),
        ("raised_exception", "exception", "ValueError"),
    ]
    assert_common_fields(collector)


@pytest.mark.asyncio
async def test_async_trace_outcomes_and_exception_propagation() -> None:
    collector = TraceCollector()
    middleware = ChaosMiddleware(collector=collector)

    async def success(req: ToolCallRequest) -> ToolMessage:
        return response(req)

    async def tool_error(req: ToolCallRequest) -> ToolMessage:
        return response(req, error=True)

    async def should_not_run(_: ToolCallRequest) -> ToolMessage:
        pytest.fail("injected call reached the original handler")

    exception = LookupError("boom")

    async def raise_exception(_: ToolCallRequest) -> ToolMessage:
        raise exception

    await middleware.awrap_tool_call(request("success"), success)
    await middleware.awrap_tool_call(request("tool-error"), tool_error)
    injected = ChaosMiddleware(
        policy=FaultPolicy("get_order_status", "injected"),
        collector=collector,
    )
    await injected.awrap_tool_call(request("injected"), should_not_run)

    with pytest.raises(LookupError) as raised:
        await middleware.awrap_tool_call(request("exception"), raise_exception)

    assert raised.value is exception
    assert [
        (event.event, event.call_id, event.exception_type)
        for event in collector.events
    ] == [
        ("start", "success", None),
        ("success", "success", None),
        ("start", "tool-error", None),
        ("returned_tool_error", "tool-error", None),
        ("start", "injected", None),
        ("injected_failure", "injected", None),
        ("start", "exception", None),
        ("raised_exception", "exception", "LookupError"),
    ]
    assert_common_fields(collector)


def test_collectors_are_independent() -> None:
    first = TraceCollector()
    second = TraceCollector()

    ChaosMiddleware(collector=first).wrap_tool_call(request("first"), response)

    assert [event.call_id for event in first.events] == ["first", "first"]
    assert second.events == ()

    ChaosMiddleware(collector=second).wrap_tool_call(request("second"), response)

    assert [event.call_id for event in first.events] == ["first", "first"]
    assert [event.call_id for event in second.events] == ["second", "second"]
