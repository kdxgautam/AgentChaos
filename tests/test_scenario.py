import pytest
from langchain.agents.middleware import ToolCallRequest
from langchain_core.messages import ToolMessage

from agentchaos import ChaosMiddleware, FaultPolicy, ScenarioRunner
from examples.order_agent import create_order_agent, executed_order_ids


AGENT_INPUT = {"messages": [{"role": "user", "content": "Where is ORD-101?"}]}


@pytest.fixture(autouse=True)
def clear_executed_orders() -> None:
    executed_order_ids.clear()


def terminal_events(result) -> list[str]:
    return [event.event for event in result.trace_events]


def test_baseline_and_matching_policy() -> None:
    result = ScenarioRunner(create_order_agent).run(
        AGENT_INPUT,
        FaultPolicy("get_order_status", "Simulated order service failure."),
    )

    assert result.baseline.outcome == "success"
    assert result.baseline.exception is None
    assert result.baseline.fault_triggered is False
    assert terminal_events(result.baseline) == ["start", "success"]
    assert result.baseline.output["messages"][-1].content.startswith(
        "Tool result received:"
    )

    assert result.faulted.outcome == "success"
    assert result.faulted.exception is None
    assert result.faulted.fault_triggered is True
    assert terminal_events(result.faulted) == ["start", "injected_failure"]
    assert result.faulted.output["messages"][-1].content == (
        "Tool failed: Simulated order service failure."
    )
    assert executed_order_ids == ["ORD-101"]


def test_nonmatching_policy_executes_both_runs() -> None:
    result = ScenarioRunner(create_order_agent).run(
        AGENT_INPUT,
        FaultPolicy("cancel_order", "Simulated cancellation service failure."),
    )

    assert result.baseline.outcome == "success"
    assert result.faulted.outcome == "success"
    assert result.baseline.fault_triggered is False
    assert result.faulted.fault_triggered is False
    assert terminal_events(result.baseline) == ["start", "success"]
    assert terminal_events(result.faulted) == ["start", "success"]
    assert executed_order_ids == ["ORD-101", "ORD-101"]


def test_agent_exceptions_are_returned_with_their_traces() -> None:
    exceptions: list[RuntimeError] = []

    class FailingAgent:
        def __init__(self, middleware: ChaosMiddleware) -> None:
            self.middleware = middleware

        def invoke(self, _: object) -> None:
            exception = RuntimeError("order service crashed")
            exceptions.append(exception)
            request = ToolCallRequest(
                tool_call={
                    "name": "get_order_status",
                    "args": {"order_id": "ORD-101"},
                    "id": "order-status-call",
                    "type": "tool_call",
                },
                tool=None,
                state={},
                runtime=None,  # type: ignore[arg-type]
            )

            def fail(_: ToolCallRequest) -> ToolMessage:
                raise exception

            self.middleware.wrap_tool_call(request, fail)

    result = ScenarioRunner(FailingAgent).run(
        AGENT_INPUT,
        FaultPolicy("cancel_order", "unrelated failure"),
    )

    assert result.baseline.outcome == "exception"
    assert result.baseline.output is None
    assert result.baseline.exception is exceptions[0]
    assert result.baseline.fault_triggered is False
    assert terminal_events(result.baseline) == ["start", "raised_exception"]
    assert result.baseline.trace_events[-1].exception_type == "RuntimeError"

    assert result.faulted.outcome == "exception"
    assert result.faulted.output is None
    assert result.faulted.exception is exceptions[1]
    assert result.faulted.fault_triggered is False
    assert terminal_events(result.faulted) == ["start", "raised_exception"]
    assert result.faulted.trace_events[-1].exception_type == "RuntimeError"


def test_runs_use_fresh_agents_and_collectors() -> None:
    agents = []
    collectors = []

    def tracked_factory(middleware: ChaosMiddleware):
        agent = create_order_agent(middleware)
        agents.append(agent)
        collectors.append(middleware.collector)
        return agent

    runner = ScenarioRunner(tracked_factory)
    first = runner.run(AGENT_INPUT, FaultPolicy("cancel_order", "unrelated"))
    first_events = first.baseline.trace_events
    runner.run(AGENT_INPUT, FaultPolicy("cancel_order", "unrelated"))

    assert len(agents) == 4
    assert len({id(agent) for agent in agents}) == 4
    assert len({id(collector) for collector in collectors}) == 4
    assert first.baseline.trace_events is not first.faulted.trace_events
    assert first.baseline.trace_events == first_events
    assert terminal_events(first.baseline) == ["start", "success"]
    assert terminal_events(first.faulted) == ["start", "success"]
