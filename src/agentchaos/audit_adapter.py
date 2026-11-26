"""Convert completed scenario evaluations into audit findings."""

from agentchaos.audit import AuditFinding, AuditSeverity, Evidence
from agentchaos.evaluator import CheckOutcome, EvaluationResult
from agentchaos.scenario import ExecutionResult


def _check_evidence(check: CheckOutcome) -> Evidence:
    exception = (
        {
            "type": type(check.exception).__name__,
            "message": str(check.exception),
        }
        if check.exception is not None
        else None
    )
    return Evidence(
        observation=f"Evaluation check: {check.name}",
        expected=None,
        actual={
            "name": check.name,
            "status": check.status,
            "exception": exception,
        },
    )


def _trace_evidence(
    execution: ExecutionResult,
    phase: str,
) -> tuple[Evidence, ...]:
    return tuple(
        Evidence(
            observation=f"{phase} trace event",
            expected=None,
            actual={
                "phase": phase,
                "event": event.event,
                "tool_name": event.tool_name,
                "call_id": event.call_id,
                "timestamp": event.timestamp.isoformat(),
                "exception_type": event.exception_type,
            },
            trace_reference=event.call_id,
        )
        for event in execution.trace_events
    )


def evaluation_to_audit_finding(
    evaluation: EvaluationResult,
    *,
    check_id: str,
    category: str,
    severity: AuditSeverity,
) -> AuditFinding:
    """Adapt an existing evaluation without executing or interpreting it again."""
    scenario = evaluation.scenario
    evidence = (
        Evidence(
            observation="Scenario fault-trigger state",
            expected=None,
            actual={
                "baseline": scenario.baseline.fault_triggered,
                "faulted": scenario.faulted.fault_triggered,
            },
        ),
        *(_check_evidence(check) for check in evaluation.checks),
        *_trace_evidence(scenario.baseline, "baseline"),
        *_trace_evidence(scenario.faulted, "faulted"),
    )
    return AuditFinding(
        finding_id=check_id,
        check_name=check_id,
        category=category,
        status=evaluation.status,
        severity=severity,
        explanation=(
            f"Scenario evaluation returned {evaluation.status} with "
            f"{len(evaluation.checks)} check outcome(s)."
        ),
        evidence=evidence,
    )
