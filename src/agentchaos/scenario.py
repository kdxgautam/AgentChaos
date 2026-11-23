"""Baseline-versus-fault scenario execution."""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

from agentchaos.middleware import ChaosMiddleware, FaultPolicy
from agentchaos.tracing import TraceCollector, TraceEvent


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """The outcome of one agent execution."""

    outcome: Literal["success", "exception"]
    output: Any | None
    exception: Exception | None
    trace_events: tuple[TraceEvent, ...]
    fault_triggered: bool


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    """The isolated baseline and faulted executions."""

    baseline: ExecutionResult
    faulted: ExecutionResult


class ScenarioRunner:
    """Run fresh baseline and faulted agents for one input."""

    def __init__(self, agent_factory: Callable[[ChaosMiddleware], Any]) -> None:
        self.agent_factory = agent_factory

    def run(self, agent_input: Any, policy: FaultPolicy) -> ScenarioResult:
        return ScenarioResult(
            baseline=self._execute(agent_input, policy=None),
            faulted=self._execute(agent_input, policy=policy),
        )

    async def arun(self, agent_input: Any, policy: FaultPolicy) -> ScenarioResult:
        return ScenarioResult(
            baseline=await self._aexecute(agent_input, policy=None),
            faulted=await self._aexecute(agent_input, policy=policy),
        )

    def _execute(
        self,
        agent_input: Any,
        policy: FaultPolicy | None,
    ) -> ExecutionResult:
        collector = TraceCollector()
        middleware = ChaosMiddleware(policy=policy, collector=collector)

        try:
            agent = self.agent_factory(middleware)
            output = agent.invoke(deepcopy(agent_input))
        except Exception as exception:
            return self._result(collector, "exception", exception=exception)
        return self._result(collector, "success", output=output)

    async def _aexecute(
        self,
        agent_input: Any,
        policy: FaultPolicy | None,
    ) -> ExecutionResult:
        collector = TraceCollector()
        middleware = ChaosMiddleware(policy=policy, collector=collector)

        try:
            agent = self.agent_factory(middleware)
            output = await agent.ainvoke(deepcopy(agent_input))
        except Exception as exception:
            return self._result(collector, "exception", exception=exception)
        return self._result(collector, "success", output=output)

    @staticmethod
    def _result(
        collector: TraceCollector,
        outcome: Literal["success", "exception"],
        *,
        output: Any | None = None,
        exception: Exception | None = None,
    ) -> ExecutionResult:
        events = collector.events
        return ExecutionResult(
            outcome=outcome,
            output=output,
            exception=exception,
            trace_events=events,
            fault_triggered=any(
                event.event == "injected_failure" for event in events
            ),
        )
