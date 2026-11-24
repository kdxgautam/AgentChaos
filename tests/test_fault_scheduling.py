import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from langchain.agents.middleware import ToolCallRequest
from langchain_core.messages import ToolMessage

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


def success(req: ToolCallRequest) -> ToolMessage:
    return ToolMessage(
        content="ok",
        name=req.tool_call["name"],
        tool_call_id=req.tool_call["id"],
        status="success",
    )


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5, "1"])
def test_on_call_rejects_invalid_values(invalid: object) -> None:
    with pytest.raises(ValueError, match="on_call must be a positive integer or None"):
        FaultPolicy("tool", "failure", on_call=invalid)  # type: ignore[arg-type]


def test_sync_before_fault_injects_only_on_scheduled_matching_call() -> None:
    collector = TraceCollector()
    middleware = ChaosMiddleware(
        FaultPolicy("get_order_status", "scheduled", on_call=2), collector
    )
    executed: list[str] = []

    def handler(req: ToolCallRequest) -> ToolMessage:
        executed.append(req.tool_call["id"])
        return success(req)

    first = middleware.wrap_tool_call(request("first"), handler)
    nonmatching = middleware.wrap_tool_call(request("other", "cancel_order"), handler)
    second = middleware.wrap_tool_call(request("second"), handler)
    third = middleware.wrap_tool_call(request("third"), handler)

    assert first.status == "success"
    assert nonmatching.status == "success"
    assert second.status == "error"
    assert second.tool_call_id == "second"
    assert third.status == "success"
    assert executed == ["first", "other", "third"]
    assert [(event.event, event.call_id) for event in collector.events] == [
        ("start", "first"),
        ("success", "first"),
        ("start", "other"),
        ("success", "other"),
        ("start", "second"),
        ("injected_failure", "second"),
        ("start", "third"),
        ("success", "third"),
    ]


def test_counters_are_isolated_between_middleware_instances() -> None:
    policy = FaultPolicy("get_order_status", "first only", on_call=1)
    first = ChaosMiddleware(policy)
    second = ChaosMiddleware(policy)

    first_result = first.wrap_tool_call(request("shared-id"), success)
    second_result = second.wrap_tool_call(request("shared-id"), success)

    assert first_result.status == "error"
    assert second_result.status == "error"


def test_none_injects_on_every_matching_call() -> None:
    middleware = ChaosMiddleware(FaultPolicy("get_order_status", "always"))

    results = [
        middleware.wrap_tool_call(request(call_id), success)
        for call_id in ("first", "second")
    ]

    assert [result.status for result in results] == ["error", "error"]


def test_sync_after_fault_replaces_only_scheduled_response() -> None:
    middleware = ChaosMiddleware(
        FaultPolicy("get_order_status", "scheduled", stage="after", on_call=2)
    )
    executed: list[str] = []

    def handler(req: ToolCallRequest) -> ToolMessage:
        executed.append(req.tool_call["id"])
        return success(req)

    results = [
        middleware.wrap_tool_call(request(call_id), handler)
        for call_id in ("first", "second", "third")
    ]

    assert [result.status for result in results] == ["success", "error", "success"]
    assert results[1].tool_call_id == "second"
    assert executed == ["first", "second", "third"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["before", "after"])
async def test_async_scheduling_matches_sync_behavior(stage: str) -> None:
    middleware = ChaosMiddleware(
        FaultPolicy(
            "get_order_status",
            "scheduled",
            stage=stage,  # type: ignore[arg-type]
            on_call=2,
        )
    )
    executed: list[str] = []

    async def handler(req: ToolCallRequest) -> ToolMessage:
        executed.append(req.tool_call["id"])
        return success(req)

    results = [
        await middleware.awrap_tool_call(request(call_id), handler)
        for call_id in ("first", "second", "third")
    ]

    assert [result.status for result in results] == ["success", "error", "success"]
    assert executed == (
        ["first", "third"] if stage == "before" else ["first", "second", "third"]
    )


@pytest.mark.asyncio
async def test_concurrent_calls_claim_distinct_attempt_numbers() -> None:
    middleware = ChaosMiddleware(
        FaultPolicy("get_order_status", "scheduled", stage="after", on_call=2)
    )
    first_entered = asyncio.Event()
    release_first = asyncio.Event()

    async def handler(req: ToolCallRequest) -> ToolMessage:
        if req.tool_call["id"] == "first":
            first_entered.set()
            await release_first.wait()
        return success(req)

    first_task = asyncio.create_task(
        middleware.awrap_tool_call(request("first"), handler)
    )
    await first_entered.wait()
    second_result = await middleware.awrap_tool_call(request("second"), handler)
    release_first.set()
    first_result = await first_task

    assert first_result.status == "success"
    assert second_result.status == "error"
    assert second_result.tool_call_id == "second"


def test_concurrent_sync_calls_claim_distinct_attempt_numbers() -> None:
    middleware = ChaosMiddleware(
        FaultPolicy("get_order_status", "first only", on_call=1)
    )
    start = Barrier(3)

    def invoke(call_id: str) -> ToolMessage:
        start.wait()
        result = middleware.wrap_tool_call(request(call_id), success)
        assert isinstance(result, ToolMessage)
        return result

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(invoke, call_id) for call_id in ("one", "two")]
        start.wait()
        results = [future.result() for future in futures]

    assert sorted(result.status for result in results) == ["error", "success"]
