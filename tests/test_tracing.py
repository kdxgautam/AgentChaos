import asyncio
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
    shared_call_id = "same-call-id"

    ChaosMiddleware(collector=first).wrap_tool_call(
        request(shared_call_id), response
    )
    ChaosMiddleware(collector=second).wrap_tool_call(
        request(shared_call_id), lambda req: response(req, error=True)
    )

    assert [(event.event, event.call_id) for event in first.events] == [
        ("start", shared_call_id),
        ("success", shared_call_id),
    ]
    assert [(event.event, event.call_id) for event in second.events] == [
        ("start", shared_call_id),
        ("returned_tool_error", shared_call_id),
    ]


@pytest.mark.asyncio
async def test_overlapping_async_calls_keep_events_correlated() -> None:
    collector = TraceCollector()
    middleware = ChaosMiddleware(collector=collector)
    success_entered = asyncio.Event()
    exception_entered = asyncio.Event()
    release = asyncio.Event()
    exception = RuntimeError("boom")

    async def succeed(req: ToolCallRequest) -> ToolMessage:
        success_entered.set()
        await exception_entered.wait()
        await release.wait()
        return response(req)

    async def fail(_: ToolCallRequest) -> ToolMessage:
        exception_entered.set()
        await success_entered.wait()
        await release.wait()
        raise exception

    success_task = asyncio.create_task(
        middleware.awrap_tool_call(request("success-call"), succeed)
    )
    exception_task = asyncio.create_task(
        middleware.awrap_tool_call(request("exception-call"), fail)
    )

    await success_entered.wait()
    await exception_entered.wait()
    assert not success_task.done()
    assert not exception_task.done()
    assert {(event.event, event.call_id) for event in collector.events} == {
        ("start", "success-call"),
        ("start", "exception-call"),
    }

    release.set()
    success_result = await success_task
    with pytest.raises(RuntimeError) as raised:
        await exception_task

    assert success_result.status == "success"
    assert raised.value is exception
    events_by_call = {
        call_id: [event for event in collector.events if event.call_id == call_id]
        for call_id in ("success-call", "exception-call")
    }
    assert [event.event for event in events_by_call["success-call"]] == [
        "start",
        "success",
    ]
    assert [event.event for event in events_by_call["exception-call"]] == [
        "start",
        "raised_exception",
    ]
    assert events_by_call["exception-call"][-1].exception_type == "RuntimeError"
