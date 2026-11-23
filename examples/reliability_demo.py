"""End-to-end deterministic AgentChaos reliability demonstration."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Self

from agentchaos import (
    ChaosMiddleware,
    EvaluationResult,
    FaultPolicy,
    ScenarioEvaluator,
    ScenarioResult,
    ScenarioRunner,
)
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool, StructuredTool


AGENT_INPUT = {"messages": [{"role": "user", "content": "Where is ORD-101?"}]}
PRIMARY_FAULT = FaultPolicy(
    tool_name="get_order_status",
    error_message="Primary order service unavailable.",
)


@dataclass(slots=True)
class ToolExecutions:
    primary: list[str] = field(default_factory=list)
    backup: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class AgentDemonstration:
    scenario: ScenarioResult
    evaluation: EvaluationResult
    executions: ToolExecutions


@dataclass(frozen=True, slots=True)
class ReliabilityDemonstration:
    fault_policy: FaultPolicy
    v1: AgentDemonstration
    v2: AgentDemonstration


def create_order_tools(executions: ToolExecutions) -> list[StructuredTool]:
    def get_order_status(order_id: str) -> str:
        """Return order status from the primary service."""
        executions.primary.append(order_id)
        return f"Primary confirms order {order_id} is shipped."

    async def aget_order_status(order_id: str) -> str:
        return get_order_status(order_id)

    def get_order_status_backup(order_id: str) -> str:
        """Return order status from the independent backup service."""
        executions.backup.append(order_id)
        return f"Backup confirms order {order_id} is delayed."

    async def aget_order_status_backup(order_id: str) -> str:
        return get_order_status_backup(order_id)

    return [
        StructuredTool.from_function(
            func=get_order_status,
            coroutine=aget_order_status,
            name="get_order_status",
        ),
        StructuredTool.from_function(
            func=get_order_status_backup,
            coroutine=aget_order_status_backup,
            name="get_order_status_backup",
        ),
    ]


class DeterministicReliabilityModel(BaseChatModel):
    """Use the primary tool, with version-specific failure behavior."""

    recover_with_backup: ClassVar[bool] = False
    model_name: ClassVar[str] = "reliability-v1"

    @property
    def _llm_type(self) -> str:
        return self.model_name

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
                        "id": "primary-status-call",
                        "type": "tool_call",
                    }
                ],
            )

        latest = tool_messages[-1]
        if latest.name == "get_order_status" and latest.status == "error":
            if self.recover_with_backup:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_order_status_backup",
                            "args": {"order_id": "ORD-101"},
                            "id": "backup-status-call",
                            "type": "tool_call",
                        }
                    ],
                )
            return AIMessage(content="Order ORD-101 is shipped.")

        source = "backup" if latest.name == "get_order_status_backup" else "primary"
        return AIMessage(content=f"VERIFIED {source}: {latest.content}")

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


class V1OrderModel(DeterministicReliabilityModel):
    pass


class V2OrderModel(DeterministicReliabilityModel):
    recover_with_backup = True
    model_name = "reliability-v2"


def create_v1_agent(middleware: ChaosMiddleware, executions: ToolExecutions):
    return create_agent(
        model=V1OrderModel(),
        tools=create_order_tools(executions),
        middleware=[middleware],
    )


def create_v2_agent(middleware: ChaosMiddleware, executions: ToolExecutions):
    return create_agent(
        model=V2OrderModel(),
        tools=create_order_tools(executions),
        middleware=[middleware],
    )


def uses_verified_backup(scenario: ScenarioResult) -> bool:
    return scenario.faulted.output["messages"][-1].content.startswith("VERIFIED backup:")


def run_demo() -> ReliabilityDemonstration:
    v1_executions = ToolExecutions()
    v2_executions = ToolExecutions()
    evaluator = ScenarioEvaluator({"uses verified backup": uses_verified_backup})

    v1_scenario = ScenarioRunner(
        lambda middleware: create_v1_agent(middleware, v1_executions)
    ).run(AGENT_INPUT, PRIMARY_FAULT)
    v2_scenario = ScenarioRunner(
        lambda middleware: create_v2_agent(middleware, v2_executions)
    ).run(AGENT_INPUT, PRIMARY_FAULT)

    return ReliabilityDemonstration(
        fault_policy=PRIMARY_FAULT,
        v1=AgentDemonstration(
            scenario=v1_scenario,
            evaluation=evaluator.evaluate(v1_scenario),
            executions=v1_executions,
        ),
        v2=AgentDemonstration(
            scenario=v2_scenario,
            evaluation=evaluator.evaluate(v2_scenario),
            executions=v2_executions,
        ),
    )


if __name__ == "__main__":
    demonstration = run_demo()
    print(f"V1: {demonstration.v1.evaluation.status}")
    print(f"V2: {demonstration.v2.evaluation.status}")
