from datetime import UTC, datetime
import json

from agentchaos import (
    AuditReport,
    EvaluationResult,
    ExecutionResult,
    ScenarioEvaluator,
    ScenarioResult,
    TraceEvent,
    evaluation_to_audit_finding,
)


TIMESTAMP = datetime(2026, 11, 26, 10, 0, tzinfo=UTC)


def scenario(*, fault_triggered: bool = True) -> ScenarioResult:
    return ScenarioResult(
        baseline=ExecutionResult(
            outcome="success",
            output={"mode": "baseline"},
            exception=None,
            trace_events=(
                TraceEvent(
                    event="success",
                    tool_name="get_order_status",
                    call_id="baseline-call",
                    timestamp=TIMESTAMP,
                ),
            ),
            fault_triggered=False,
        ),
        faulted=ExecutionResult(
            outcome="success",
            output={"mode": "faulted"},
            exception=None,
            trace_events=(
                TraceEvent(
                    event="response_replaced",
                    tool_name="get_order_status",
                    call_id="faulted-call",
                    timestamp=TIMESTAMP,
                ),
            ),
            fault_triggered=fault_triggered,
        ),
    )


def adapt(evaluation: EvaluationResult):
    return evaluation_to_audit_finding(
        evaluation,
        check_id="AC-RECOVERY-001",
        category="recovery",
        severity="HIGH",
    )


def evidence_by_observation(finding, observation: str):
    return next(item for item in finding.evidence if item.observation == observation)


def test_pass_evaluation_preserves_checks_fault_state_and_trace_references() -> None:
    evaluation = ScenarioEvaluator(
        {
            "baseline succeeded": lambda result: result.baseline.outcome == "success",
            "faulted result accepted": lambda _: True,
        }
    ).evaluate(scenario())

    finding = adapt(evaluation)

    assert finding.finding_id == "AC-RECOVERY-001"
    assert finding.check_name == "AC-RECOVERY-001"
    assert finding.category == "recovery"
    assert finding.status == "PASS"
    assert finding.severity == "HIGH"
    assert evidence_by_observation(
        finding, "Scenario fault-trigger state"
    ).actual == {"baseline": False, "faulted": True}
    assert evidence_by_observation(
        finding, "Evaluation check: baseline succeeded"
    ).actual == {
        "name": "baseline succeeded",
        "status": "PASS",
        "exception": None,
    }
    assert [
        item.trace_reference
        for item in finding.evidence
        if item.trace_reference is not None
    ] == ["baseline-call", "faulted-call"]
    assert all(item.expected is None for item in finding.evidence)


def test_fail_evaluation_status_is_preserved_without_reexecution() -> None:
    calls = 0

    def fail(_: ScenarioResult) -> bool:
        nonlocal calls
        calls += 1
        return False

    evaluation = ScenarioEvaluator({"duplicate prevented": fail}).evaluate(scenario())
    assert calls == 1

    finding = adapt(evaluation)

    assert calls == 1
    assert finding.status == "FAIL"
    assert evidence_by_observation(
        finding, "Evaluation check: duplicate prevented"
    ).actual["status"] == "FAIL"


def test_inconclusive_evaluation_with_no_checks_is_preserved() -> None:
    finding = adapt(ScenarioEvaluator().evaluate(scenario()))

    assert finding.status == "INCONCLUSIVE"
    assert "0 check outcome(s)" in finding.explanation


def test_check_exception_details_are_preserved() -> None:
    exception = RuntimeError("check crashed")

    def crash(_: ScenarioResult) -> bool:
        raise exception

    finding = adapt(ScenarioEvaluator({"unsafe check": crash}).evaluate(scenario()))
    check = evidence_by_observation(finding, "Evaluation check: unsafe check")

    assert finding.status == "INCONCLUSIVE"
    assert check.actual == {
        "name": "unsafe check",
        "status": "ERROR",
        "exception": {"type": "RuntimeError", "message": "check crashed"},
    }


def test_untriggered_fault_remains_inconclusive_and_visible() -> None:
    evaluation = ScenarioEvaluator({"check passed": lambda _: True}).evaluate(
        scenario(fault_triggered=False)
    )

    finding = adapt(evaluation)

    assert finding.status == "INCONCLUSIVE"
    assert evidence_by_observation(
        finding, "Scenario fault-trigger state"
    ).actual == {"baseline": False, "faulted": False}


def test_adapted_finding_serializes_through_audit_report() -> None:
    finding = adapt(ScenarioEvaluator({"passes": lambda _: True}).evaluate(scenario()))
    report = AuditReport(
        audit_id="AUDIT-ADAPTER-001",
        name="Adapter audit",
        started_at=TIMESTAMP,
        completed_at=TIMESTAMP,
        findings=(finding,),
    )

    serialized = json.loads(report.to_json())
    serialized_finding = serialized["findings"][0]
    assert serialized_finding["status"] == "PASS"
    assert serialized_finding["severity"] == "HIGH"
    assert serialized_finding["evidence"][0]["expected"] is None
    assert serialized_finding["evidence"][-1]["trace_reference"] == "faulted-call"
