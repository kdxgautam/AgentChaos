from dataclasses import FrozenInstanceError

import pytest
from langchain_core.messages import ToolMessage

from agentchaos import FaultPolicy
from examples.order_agent import (
    agent,
    chaos_agent,
    executed_order_ids,
    nonmatching_policy_agent,
)


@pytest.fixture(autouse=True)
def clear_executed_orders() -> None:
    executed_order_ids.clear()


def get_tool_message(result: dict) -> ToolMessage:
    return next(
        message for message in result["messages"] if isinstance(message, ToolMessage)
    )


def test_sync_normal_mode_executes_tool() -> None:
    result = agent.invoke({"messages": [{"role": "user", "content": "Where is ORD-101?"}]})
    tool_message = get_tool_message(result)

    assert executed_order_ids == ["ORD-101"]
    assert tool_message.status == "success"
    assert tool_message.tool_call_id == "order-status-call"
    assert tool_message.content == (
        "Order ORD-101 is shipped. Expected delivery: 2026-11-20."
    )
    assert result["messages"][-1].content == f"Tool result received: {tool_message.content}"


def test_sync_chaos_mode_skips_tool() -> None:
    result = chaos_agent.invoke(
        {"messages": [{"role": "user", "content": "Where is ORD-101?"}]}
    )
    tool_message = get_tool_message(result)

    assert executed_order_ids == []
    assert tool_message.status == "error"
    assert tool_message.tool_call_id == "order-status-call"
    assert tool_message.content == "Simulated order service failure."
    assert result["messages"][-1].content == "Tool failed: Simulated order service failure."


def test_sync_nonmatching_policy_executes_tool() -> None:
    result = nonmatching_policy_agent.invoke(
        {"messages": [{"role": "user", "content": "Where is ORD-101?"}]}
    )
    tool_message = get_tool_message(result)

    assert executed_order_ids == ["ORD-101"]
    assert tool_message.status == "success"
    assert tool_message.tool_call_id == "order-status-call"
    assert result["messages"][-1].content == f"Tool result received: {tool_message.content}"


@pytest.mark.asyncio
async def test_async_normal_mode_executes_tool() -> None:
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Where is ORD-101?"}]}
    )
    tool_message = get_tool_message(result)

    assert executed_order_ids == ["ORD-101"]
    assert tool_message.status == "success"
    assert tool_message.tool_call_id == "order-status-call"
    assert tool_message.content == (
        "Order ORD-101 is shipped. Expected delivery: 2026-11-20."
    )
    assert result["messages"][-1].content == f"Tool result received: {tool_message.content}"


@pytest.mark.asyncio
async def test_async_chaos_mode_skips_tool() -> None:
    result = await chaos_agent.ainvoke(
        {"messages": [{"role": "user", "content": "Where is ORD-101?"}]}
    )
    tool_message = get_tool_message(result)

    assert executed_order_ids == []
    assert tool_message.status == "error"
    assert tool_message.tool_call_id == "order-status-call"
    assert tool_message.content == "Simulated order service failure."
    assert result["messages"][-1].content == "Tool failed: Simulated order service failure."


@pytest.mark.asyncio
async def test_async_nonmatching_policy_executes_tool() -> None:
    result = await nonmatching_policy_agent.ainvoke(
        {"messages": [{"role": "user", "content": "Where is ORD-101?"}]}
    )
    tool_message = get_tool_message(result)

    assert executed_order_ids == ["ORD-101"]
    assert tool_message.status == "success"
    assert tool_message.tool_call_id == "order-status-call"
    assert result["messages"][-1].content == f"Tool result received: {tool_message.content}"


def test_fault_policy_is_immutable() -> None:
    policy = FaultPolicy(tool_name="get_order_status", error_message="failure")

    with pytest.raises(FrozenInstanceError):
        policy.error_message = "changed"  # type: ignore[misc]
