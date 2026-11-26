"""Deterministic LangGraph supervisor and parallel repository workers."""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
import operator
from pathlib import PurePosixPath
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agentchaos.repository import RepositoryAnalysisError, RepositoryAnalyzer


WorkerRole = Literal["security", "reliability", "architecture", "performance"]
WorkerOutcome = Literal["success", "expected_failure"]
FindingUncertainty = Literal["low", "medium", "high"]


@dataclass(frozen=True, slots=True)
class CandidateFinding:
    """An unverified, source-referenced specialist observation."""

    claim: str
    source_path: str
    start_line: int
    end_line: int
    evidence: str
    uncertainty: FindingUncertainty

    def __post_init__(self) -> None:
        if not isinstance(self.claim, str) or not self.claim.strip():
            raise ValueError("claim must be non-empty")
        if not isinstance(self.source_path, str):
            raise ValueError("source_path must be repository-relative")
        path = PurePosixPath(self.source_path)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise ValueError("source_path must be repository-relative")
        if (
            not isinstance(self.start_line, int)
            or not isinstance(self.end_line, int)
            or self.start_line < 1
            or self.end_line < self.start_line
        ):
            raise ValueError("source line range is invalid")
        if not isinstance(self.evidence, str) or not self.evidence.strip():
            raise ValueError("evidence must be non-empty")
        if not isinstance(self.uncertainty, str) or self.uncertainty not in {
            "low",
            "medium",
            "high",
        }:
            raise ValueError("uncertainty must be low, medium, or high")


@dataclass(frozen=True, slots=True)
class WorkerOutput:
    """Structured data returned by a successful specialist worker."""

    observations: tuple[str, ...] = ()
    candidate_findings: tuple[CandidateFinding, ...] = ()

    def __post_init__(self) -> None:
        observations = tuple(self.observations)
        findings = tuple(self.candidate_findings)
        if any(not isinstance(item, str) for item in observations):
            raise ValueError("observations must contain only strings")
        if any(not isinstance(item, CandidateFinding) for item in findings):
            raise ValueError("candidate_findings must contain only CandidateFinding values")
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "candidate_findings", findings)


@dataclass(frozen=True, slots=True)
class AuditTask:
    """One stable supervisor assignment."""

    task_id: str
    role: WorkerRole
    order: int

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id must be non-empty")
        if self.order < 0:
            raise ValueError("order must not be negative")


@dataclass(frozen=True, slots=True)
class WorkerResult:
    """The structured outcome from one repository worker."""

    task_id: str
    role: WorkerRole
    outcome: WorkerOutcome
    observations: tuple[str, ...]
    error_type: str | None = None
    error_message: str | None = None
    candidate_findings: tuple[CandidateFinding, ...] = ()

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id must be non-empty")
        observations = tuple(self.observations)
        if any(not isinstance(item, str) for item in observations):
            raise ValueError("observations must contain only strings")
        candidate_findings = tuple(self.candidate_findings)
        if any(not isinstance(item, CandidateFinding) for item in candidate_findings):
            raise ValueError("candidate_findings must contain only CandidateFinding values")
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "candidate_findings", candidate_findings)


class ExpectedWorkerFailure(Exception):
    """A worker-specific failure that should not discard sibling results."""

    def __init__(self, message: str, *, observations: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.observations = tuple(observations)


WorkerHandler = Callable[
    [AuditTask, RepositoryAnalyzer], Awaitable[Sequence[str] | WorkerOutput]
]


AUDIT_TASKS = (
    AuditTask("repository-security", "security", 0),
    AuditTask("repository-reliability", "reliability", 1),
    AuditTask("repository-architecture", "architecture", 2),
    AuditTask("repository-performance", "performance", 3),
)


async def _security_worker(
    _: AuditTask, repository: RepositoryAnalyzer
) -> tuple[str, ...]:
    return (
        f"python_files={len(repository.list_files())}",
        f"imports={len(repository.get_imports())}",
    )


async def _reliability_worker(
    _: AuditTask, repository: RepositoryAnalyzer
) -> tuple[str, ...]:
    return (f"analysis_issues={len(repository.issues)}",)


async def _architecture_worker(
    _: AuditTask, repository: RepositoryAnalyzer
) -> tuple[str, ...]:
    return (
        f"imports={len(repository.get_imports())}",
        f"decorators={len(repository.get_decorators())}",
    )


async def _performance_worker(
    _: AuditTask, repository: RepositoryAnalyzer
) -> tuple[str, ...]:
    readable_files = 0
    source_bytes = 0
    for path in repository.list_files():
        try:
            source = repository.read_source(path)
        except RepositoryAnalysisError:
            continue
        readable_files += 1
        source_bytes += len(source.encode("utf-8"))
    return (
        f"readable_files={readable_files}",
        f"source_bytes={source_bytes}",
    )


DEFAULT_WORKERS: Mapping[WorkerRole, WorkerHandler] = {
    "security": _security_worker,
    "reliability": _reliability_worker,
    "architecture": _architecture_worker,
    "performance": _performance_worker,
}


class _AuditGraphState(TypedDict):
    repository: RepositoryAnalyzer
    tasks: tuple[AuditTask, ...]
    worker_results: Annotated[list[WorkerResult], operator.add]
    results: tuple[WorkerResult, ...]


class _WorkerState(TypedDict):
    repository: RepositoryAnalyzer
    task: AuditTask


def _supervise(_: _AuditGraphState) -> dict[str, tuple[AuditTask, ...]]:
    return {"tasks": AUDIT_TASKS}


def _dispatch(state: _AuditGraphState) -> list[Send]:
    return [
        Send("worker", {"repository": state["repository"], "task": task})
        for task in state["tasks"]
    ]


def _aggregate(state: _AuditGraphState) -> dict[str, tuple[WorkerResult, ...]]:
    task_order = {task.task_id: task.order for task in state["tasks"]}
    return {
        "results": tuple(
            sorted(
                state["worker_results"],
                key=lambda result: task_order[result.task_id],
            )
        )
    }


class RepositoryAuditOrchestrator:
    """Run deterministic repository workers concurrently through LangGraph."""

    def __init__(
        self,
        workers: Mapping[WorkerRole, WorkerHandler] | None = None,
    ) -> None:
        configured_workers = dict(DEFAULT_WORKERS)
        if workers is not None:
            configured_workers.update(workers)

        async def run_worker(state: _WorkerState) -> dict[str, list[WorkerResult]]:
            task = state["task"]
            handler = configured_workers[task.role]
            try:
                output = await handler(task, state["repository"])
            except ExpectedWorkerFailure as exception:
                result = WorkerResult(
                    task_id=task.task_id,
                    role=task.role,
                    outcome="expected_failure",
                    observations=exception.observations,
                    error_type=type(exception).__name__,
                    error_message=str(exception),
                )
            else:
                if isinstance(output, WorkerOutput):
                    observations = output.observations
                    candidate_findings = output.candidate_findings
                else:
                    observations = tuple(output)
                    candidate_findings = ()
                result = WorkerResult(
                    task_id=task.task_id,
                    role=task.role,
                    outcome="success",
                    observations=observations,
                    candidate_findings=candidate_findings,
                )
            return {"worker_results": [result]}

        graph = StateGraph(_AuditGraphState)
        graph.add_node("supervisor", _supervise)
        graph.add_node("worker", run_worker)
        graph.add_node("aggregate", _aggregate)
        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges("supervisor", _dispatch, ["worker"])
        graph.add_edge("worker", "aggregate")
        graph.add_edge("aggregate", END)
        self.graph = graph.compile()

    async def arun(
        self, repository: RepositoryAnalyzer
    ) -> tuple[WorkerResult, ...]:
        state = await self.graph.ainvoke(
            {
                "repository": repository,
                "tasks": (),
                "worker_results": [],
                "results": (),
            }
        )
        return state["results"]
