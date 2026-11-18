import pytest
from langchain_core.messages import AIMessage, ToolMessage

from examples.order_agent import agent


@pytest.mark.asyncio
async def test_agent_executes_order_status_tool() -> None:
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Where is ORD-101?"}]}
    )
    messages = result["messages"]

    tool_call_message = next(
        message for message in messages if isinstance(message, AIMessage) and message.tool_calls
    )
    assert tool_call_message.tool_calls[0]["name"] == "get_order_status"
    assert tool_call_message.tool_calls[0]["args"] == {"order_id": "ORD-101"}

    tool_message = next(message for message in messages if isinstance(message, ToolMessage))
    assert tool_message.name == "get_order_status"
    assert tool_message.tool_call_id == "order-status-call"
    assert tool_message.content == (
        "Order ORD-101 is shipped. Expected delivery: 2026-11-20."
    )
    assert messages[-1].content == f"Tool result received: {tool_message.content}"
