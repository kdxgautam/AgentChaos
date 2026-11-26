import asyncio
from pathlib import Path

import pytest

from agentchaos import (
    AUDIT_TASKS,
    ExpectedWorkerFailure,
    RepositoryAnalyzer,
    RepositoryAuditOrchestrator,
)


@pytest.fixture
def repository(tmp_path: Path) -> RepositoryAnalyzer:
    (tmp_path / "app.py").write_text(
        """import os

@decorator
class App:
    def run(self):
        return os.getcwd()
""",
        encoding="utf-8",
    )
    return RepositoryAnalyzer(tmp_path)


@pytest.mark.asyncio
async def test_supervisor_delegates_all_roles_with_isolated_task_state(
    repository: RepositoryAnalyzer,
) -> None:
    seen_tasks: list[tuple[int, str, str, int]] = []

    async def inspect(task, received_repository):
        assert received_repository is repository
        seen_tasks.append((id(task), task.task_id, task.role, task.order))
        return (f"handled={task.task_id}",)

    orchestrator = RepositoryAuditOrchestrator(
        {task.role: inspect for task in AUDIT_TASKS}
    )

    results = await orchestrator.arun(repository)

    assert sorted(
        ((task_id, role, order) for _, task_id, role, order in seen_tasks),
        key=lambda item: item[2],
    ) == [(task.task_id, task.role, task.order) for task in AUDIT_TASKS]
    assert len({identity for identity, *_ in seen_tasks}) == 4
    assert [result.task_id for result in results] == [
        task.task_id for task in AUDIT_TASKS
    ]
    assert all(result.outcome == "success" for result in results)
    assert [result.observations for result in results] == [
        (f"handled={task.task_id}",) for task in AUDIT_TASKS
    ]


@pytest.mark.asyncio
async def test_workers_overlap_and_aggregate_by_task_order(
    repository: RepositoryAnalyzer,
) -> None:
    started: list[str] = []
    completed: list[str] = []
    all_started = asyncio.Event()
    release = {task.role: asyncio.Event() for task in AUDIT_TASKS}
    finished = {task.role: asyncio.Event() for task in AUDIT_TASKS}

    async def overlap(task, _):
        started.append(task.role)
        if len(started) == len(AUDIT_TASKS):
            all_started.set()
        await release[task.role].wait()
        completed.append(task.role)
        finished[task.role].set()
        return (f"completed={task.role}",)

    orchestrator = RepositoryAuditOrchestrator(
        {task.role: overlap for task in AUDIT_TASKS}
    )
    run = asyncio.create_task(orchestrator.arun(repository))

    await asyncio.wait_for(all_started.wait(), timeout=2)
    assert not run.done()
    completion_order = [task.role for task in reversed(AUDIT_TASKS)]
    for role in completion_order:
        release[role].set()
        await asyncio.wait_for(finished[role].wait(), timeout=2)

    results = await run
    assert completed == completion_order
    assert [result.role for result in results] == [task.role for task in AUDIT_TASKS]


@pytest.mark.asyncio
async def test_expected_worker_failure_keeps_successful_siblings(
    repository: RepositoryAnalyzer,
) -> None:
    async def inspect(task, _):
        if task.role == "reliability":
            raise ExpectedWorkerFailure(
                "repository could not be inspected",
                observations=("partial=true",),
            )
        return (f"role={task.role}",)

    results = await RepositoryAuditOrchestrator(
        {task.role: inspect for task in AUDIT_TASKS}
    ).arun(repository)

    assert [result.outcome for result in results] == [
        "success",
        "expected_failure",
        "success",
        "success",
    ]
    failed = results[1]
    assert failed.observations == ("partial=true",)
    assert failed.error_type == "ExpectedWorkerFailure"
    assert failed.error_message == "repository could not be inspected"


@pytest.mark.asyncio
async def test_default_workers_use_repository_intelligence(
    repository: RepositoryAnalyzer,
) -> None:
    results = await RepositoryAuditOrchestrator().arun(repository)

    assert [result.role for result in results] == [task.role for task in AUDIT_TASKS]
    assert all(result.outcome == "success" for result in results)
    assert results[0].observations == ("python_files=1", "imports=1")
    assert results[1].observations == ("analysis_issues=0",)
    assert results[2].observations == ("imports=1", "decorators=1")
    assert results[3].observations[0] == "readable_files=1"


@pytest.mark.asyncio
async def test_unexpected_programming_errors_propagate(
    repository: RepositoryAnalyzer,
) -> None:
    async def broken(task, _):
        if task.role == "security":
            raise AssertionError("worker bug")
        return ()

    with pytest.raises(AssertionError, match="worker bug"):
        await RepositoryAuditOrchestrator(
            {task.role: broken for task in AUDIT_TASKS}
        ).arun(repository)


@pytest.mark.asyncio
async def test_cancellation_propagates_to_parallel_workers(
    repository: RepositoryAnalyzer,
) -> None:
    started = 0
    all_started = asyncio.Event()
    never_release = asyncio.Event()

    async def wait_forever(_, __):
        nonlocal started
        started += 1
        if started == len(AUDIT_TASKS):
            all_started.set()
        await never_release.wait()
        return ()

    orchestrator = RepositoryAuditOrchestrator(
        {task.role: wait_forever for task in AUDIT_TASKS}
    )
    run = asyncio.create_task(orchestrator.arun(repository))
    await asyncio.wait_for(all_started.wait(), timeout=2)

    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
