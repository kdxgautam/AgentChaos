from examples.reliability_demo import run_demo


def final_message(execution) -> str:
    return execution.output["messages"][-1].content


def event_pairs(execution) -> list[tuple[str, str]]:
    return [(event.tool_name, event.event) for event in execution.trace_events]


def test_same_fault_exposes_v1_and_v2_reliability_difference() -> None:
    demonstration = run_demo()

    assert demonstration.fault_policy.tool_name == "get_order_status"
    assert demonstration.v1.scenario.faulted.fault_triggered is True
    assert demonstration.v2.scenario.faulted.fault_triggered is True

    assert final_message(demonstration.v1.scenario.baseline).startswith(
        "VERIFIED primary:"
    )
    assert final_message(demonstration.v2.scenario.baseline).startswith(
        "VERIFIED primary:"
    )
    assert final_message(demonstration.v1.scenario.faulted) == (
        "Order ORD-101 is shipped."
    )
    assert final_message(demonstration.v2.scenario.faulted).startswith(
        "VERIFIED backup: Backup confirms order ORD-101 is delayed."
    )

    assert demonstration.v1.evaluation.status == "FAIL"
    assert demonstration.v2.evaluation.status == "PASS"
    assert demonstration.v1.evaluation.checks[0].status == "FAIL"
    assert demonstration.v2.evaluation.checks[0].status == "PASS"


def test_tool_executions_and_traces_match_recovery_paths() -> None:
    demonstration = run_demo()

    assert demonstration.v1.executions.primary == ["ORD-101"]
    assert demonstration.v1.executions.backup == []
    assert demonstration.v2.executions.primary == ["ORD-101"]
    assert demonstration.v2.executions.backup == ["ORD-101"]

    assert event_pairs(demonstration.v1.scenario.faulted) == [
        ("get_order_status", "start"),
        ("get_order_status", "injected_failure"),
    ]
    assert event_pairs(demonstration.v2.scenario.faulted) == [
        ("get_order_status", "start"),
        ("get_order_status", "injected_failure"),
        ("get_order_status_backup", "start"),
        ("get_order_status_backup", "success"),
    ]

    primary_service_attempts = len(demonstration.v1.executions.primary) + len(
        demonstration.v2.executions.primary
    )
    assert primary_service_attempts == 2
    assert primary_service_attempts <= 2


def test_demonstration_runs_keep_tool_state_isolated() -> None:
    first = run_demo()
    second = run_demo()

    assert first.v1.executions is not second.v1.executions
    assert first.v2.executions is not second.v2.executions
    assert first.v1.executions.primary == ["ORD-101"]
    assert second.v1.executions.primary == ["ORD-101"]
    assert first.v2.executions.backup == ["ORD-101"]
    assert second.v2.executions.backup == ["ORD-101"]
