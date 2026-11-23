import asyncio

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


@pytest.mark.asyncio
async def test_arun_with_matching_policy() -> None:
    result = await ScenarioRunner(create_order_agent).arun(
        AGENT_INPUT,
        FaultPolicy("get_order_status", "Simulated order service failure."),
    )

    assert result.baseline.outcome == "success"
    assert result.baseline.fault_triggered is False
    assert terminal_events(result.baseline) == ["start", "success"]
    assert result.faulted.outcome == "success"
    assert result.faulted.fault_triggered is True
    assert terminal_events(result.faulted) == ["start", "injected_failure"]
    assert result.faulted.output["messages"][-1].content == (
        "Tool failed: Simulated order service failure."
    )
    assert executed_order_ids == ["ORD-101"]


@pytest.mark.asyncio
async def test_arun_with_nonmatching_policy() -> None:
    result = await ScenarioRunner(create_order_agent).arun(
        AGENT_INPUT,
        FaultPolicy("cancel_order", "unrelated failure"),
    )

    assert result.baseline.outcome == "success"
    assert result.faulted.outcome == "success"
    assert result.baseline.fault_triggered is False
    assert result.faulted.fault_triggered is False
    assert terminal_events(result.baseline) == ["start", "success"]
    assert terminal_events(result.faulted) == ["start", "success"]
    assert executed_order_ids == ["ORD-101", "ORD-101"]


@pytest.mark.asyncio
async def test_arun_calls_ainvoke_and_captures_exceptions() -> None:
    agents = []
    exceptions: list[RuntimeError] = []

    class AsyncFailingAgent:
        def __init__(self, middleware: ChaosMiddleware) -> None:
            self.middleware = middleware
            self.ainvoke_calls = 0
            agents.append(self)

        def invoke(self, _: object) -> None:
            pytest.fail("arun called invoke instead of ainvoke")

        async def ainvoke(self, _: object) -> None:
            self.ainvoke_calls += 1
            exception = RuntimeError("async order service crashed")
            exceptions.append(exception)
            request = ToolCallRequest(
                tool_call={
                    "name": "get_order_status",
                    "args": {"order_id": "ORD-101"},
                    "id": "async-order-status-call",
                    "type": "tool_call",
                },
                tool=None,
                state={},
                runtime=None,  # type: ignore[arg-type]
            )

            async def fail(_: ToolCallRequest) -> ToolMessage:
                raise exception

            await self.middleware.awrap_tool_call(request, fail)

    result = await ScenarioRunner(AsyncFailingAgent).arun(
        AGENT_INPUT,
        FaultPolicy("cancel_order", "unrelated failure"),
    )

    assert len(agents) == 2
    assert [agent.ainvoke_calls for agent in agents] == [1, 1]
    assert result.baseline.outcome == "exception"
    assert result.baseline.exception is exceptions[0]
    assert terminal_events(result.baseline) == ["start", "raised_exception"]
    assert result.faulted.outcome == "exception"
    assert result.faulted.exception is exceptions[1]
    assert terminal_events(result.faulted) == ["start", "raised_exception"]


@pytest.mark.asyncio
async def test_concurrent_arun_calls_keep_scenarios_isolated() -> None:
    agents = []
    collectors = []
    baseline_handlers_entered = 0
    both_baselines_entered = asyncio.Event()

    class OverlappingAgent:
        def __init__(self, middleware: ChaosMiddleware) -> None:
            self.middleware = middleware
            agents.append(self)
            collectors.append(middleware.collector)

        async def ainvoke(self, _: object) -> ToolMessage:
            request = ToolCallRequest(
                tool_call={
                    "name": "get_order_status",
                    "args": {"order_id": "ORD-101"},
                    "id": "shared-call-id",
                    "type": "tool_call",
                },
                tool=None,
                state={},
                runtime=None,  # type: ignore[arg-type]
            )

            async def execute(req: ToolCallRequest) -> ToolMessage:
                nonlocal baseline_handlers_entered
                baseline_handlers_entered += 1
                if baseline_handlers_entered == 2:
                    both_baselines_entered.set()
                await both_baselines_entered.wait()
                return ToolMessage(
                    content="ok",
                    name=req.tool_call["name"],
                    tool_call_id=req.tool_call["id"],
                    status="success",
                )

            return await self.middleware.awrap_tool_call(request, execute)

    runner = ScenarioRunner(OverlappingAgent)
    policy = FaultPolicy("get_order_status", "injected")
    first, second = await asyncio.gather(
        runner.arun(AGENT_INPUT, policy),
        runner.arun(AGENT_INPUT, policy),
    )

    assert len(agents) == 4
    assert len({id(agent) for agent in agents}) == 4
    assert len({id(collector) for collector in collectors}) == 4
    for result in (first, second):
        assert terminal_events(result.baseline) == ["start", "success"]
        assert terminal_events(result.faulted) == ["start", "injected_failure"]
        assert result.baseline.fault_triggered is False
        assert result.faulted.fault_triggered is True
        assert {
            event.call_id
            for event in result.baseline.trace_events + result.faulted.trace_events
        } == {"shared-call-id"}


@pytest.mark.asyncio
async def test_arun_propagates_cancellation() -> None:
    handler_started = asyncio.Event()
    never_release = asyncio.Event()
    collectors = []

    class BlockingAgent:
        def __init__(self, middleware: ChaosMiddleware) -> None:
            self.middleware = middleware
            collectors.append(middleware.collector)

        async def ainvoke(self, _: object) -> ToolMessage:
            request = ToolCallRequest(
                tool_call={
                    "name": "get_order_status",
                    "args": {"order_id": "ORD-101"},
                    "id": "cancelled-call",
                    "type": "tool_call",
                },
                tool=None,
                state={},
                runtime=None,  # type: ignore[arg-type]
            )

            async def block(_: ToolCallRequest) -> ToolMessage:
                handler_started.set()
                await never_release.wait()
                raise AssertionError("unreachable")

            return await self.middleware.awrap_tool_call(request, block)

    task = asyncio.create_task(
        ScenarioRunner(BlockingAgent).arun(
            AGENT_INPUT,
            FaultPolicy("cancel_order", "unrelated failure"),
        )
    )
    await handler_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(collectors) == 1
    assert [event.event for event in collectors[0].events] == ["start"]
