"""A deterministic LangChain agent that executes a real Python tool."""

from collections.abc import Callable, Sequence
from typing import Any, Self

from agentchaos import ChaosMiddleware, FaultPolicy
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool, StructuredTool


executed_order_ids: list[str] = []


def get_order_status(order_id: str) -> str:
    """Return the status of an order."""
    executed_order_ids.append(order_id)
    return f"Order {order_id} is shipped. Expected delivery: 2026-11-20."


async def aget_order_status(order_id: str) -> str:
    """Return the status of an order asynchronously."""
    return get_order_status(order_id)


order_status_tool = StructuredTool.from_function(
    func=get_order_status,
    coroutine=aget_order_status,
)
# StructuredTool exposes the function's typed schema and sync/async implementations
# so the agent can validate tool-call arguments and execute the function reliably.


class DeterministicOrderModel(BaseChatModel):
    """Emit one fixed tool call, then answer from the executed tool result."""

    @property
    def _llm_type(self) -> str:
        return "deterministic-order-model"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Self:
        return self

    def _next_message(self, messages: list[BaseMessage]) -> AIMessage:
        tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
        if not tool_messages:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_order_status",
                        "args": {"order_id": "ORD-101"},
                        "id": "order-status-call",
                        "type": "tool_call",
                    }
                ],
            )

        tool_message = tool_messages[-1]
        if tool_message.status == "error":
            return AIMessage(content=f"Tool failed: {tool_message.content}")
        return AIMessage(content=f"Tool result received: {tool_message.content}")

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next_message(messages))])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next_message(messages))])


def create_order_agent(middleware: ChaosMiddleware):
    """Create a fresh deterministic order agent."""
    return create_agent(
        model=DeterministicOrderModel(),
        tools=[order_status_tool],
        middleware=[middleware],
    )


agent = create_order_agent(ChaosMiddleware())
chaos_agent = create_order_agent(
    ChaosMiddleware(
        FaultPolicy(
            tool_name="get_order_status",
            error_message="Simulated order service failure.",
        )
    )
)
nonmatching_policy_agent = create_order_agent(
    ChaosMiddleware(
        FaultPolicy(
            tool_name="cancel_order",
            error_message="Simulated cancellation service failure.",
        )
    )
)


if __name__ == "__main__":
    result = agent.invoke({"messages": [{"role": "user", "content": "Where is my order?"}]})
    print(result["messages"][-1].content)
