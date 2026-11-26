from datetime import UTC, datetime

import pytest
from langchain.agents.middleware import ToolCallRequest
from langchain_core.messages import ToolMessage

from agentchaos import AuditRunner, AuditScenario, ChaosMiddleware, FaultPolicy


TIMESTAMP = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)
AGENT_INPUT = {"order_id": "ORD-101"}


class ExpectedAgentError(RuntimeError):
    pass


class FaultAwareAgent:
    def __init__(self, middleware: ChaosMiddleware) -> None:
        self.middleware = middleware

    def invoke(self, agent_input: dict) -> ToolMessage:
        request = ToolCallRequest(
            tool_call={
                "name": "get_order_status",
                "args": agent_input,
                "id": "subject-call",
                "type": "tool_call",
            },
            tool=None,
            state={},
            runtime=None,  # type: ignore[arg-type]
        )

        def execute(req: ToolCallRequest) -> ToolMessage:
            return ToolMessage(
                content="shipped",
                name=req.tool_call["name"],
                tool_call_id=req.tool_call["id"],
                status="success",
            )

        result = self.middleware.wrap_tool_call(request, execute)
        assert isinstance(result, ToolMessage)
        if result.status == "error":
            raise ExpectedAgentError("agent rejected the failed tool result")
        return result


def subject_scenario(check_id: str, check) -> AuditScenario:
    return AuditScenario(
        check_id=check_id,
        category="exception-recovery",
        severity="HIGH",
        agent_factory=FaultAwareAgent,
        agent_input=AGENT_INPUT,
        policy=FaultPolicy("get_order_status", "injected failure"),
        evaluation_checks={"subject exception check": check},
    )


def test_expected_subject_exception_can_pass() -> None:
    configured = subject_scenario(
        "EXPECTED-EXCEPTION",
        lambda result: (
            result.faulted.exception_provenance == "subject"
            and isinstance(result.faulted.exception, ExpectedAgentError)
        ),
    )

    finding = AuditRunner((configured,), clock=lambda: TIMESTAMP).run(
        audit_id="AUDIT-EXPECTED",
        name="Expected exception audit",
    ).findings[0]

    assert finding.status == "PASS"
    assert finding.evidence[0].actual == {"baseline": False, "faulted": True}


def test_verified_subject_exception_violation_can_fail() -> None:
    configured = subject_scenario(
        "VERIFIED-VIOLATION",
        lambda result: result.faulted.exception is None,
    )

    finding = AuditRunner((configured,), clock=lambda: TIMESTAMP).run(
        audit_id="AUDIT-VIOLATION",
        name="Violation audit",
    ).findings[0]

    assert finding.status == "FAIL"


def test_known_harness_failure_is_inconclusive_with_diagnostics() -> None:
    checks_run = 0

    def broken_factory(_: ChaosMiddleware):
        raise OSError("fixture setup failed")

    def check(_: object) -> bool:
        nonlocal checks_run
        checks_run += 1
        return False

    configured = AuditScenario(
        check_id="HARNESS-FAILURE",
        category="infrastructure",
        severity="MEDIUM",
        agent_factory=broken_factory,
        agent_input=AGENT_INPUT,
        policy=FaultPolicy("get_order_status", "injected failure"),
        evaluation_checks={"must not run": check},
    )

    finding = AuditRunner((configured,), clock=lambda: TIMESTAMP).run(
        audit_id="AUDIT-HARNESS",
        name="Harness audit",
    ).findings[0]

    assert checks_run == 0
    assert finding.status == "INCONCLUSIVE"
    diagnostics = [
        item.actual
        for item in finding.evidence
        if item.observation.endswith("harness failure")
    ]
    assert diagnostics == [
        {
            "phase": "baseline",
            "provenance": "harness",
            "type": "OSError",
            "message": "fixture setup failed",
        },
        {
            "phase": "faulted",
            "provenance": "harness",
            "type": "OSError",
            "message": "fixture setup failed",
        },
    ]


def test_mixed_audit_preserves_each_exception_classification() -> None:
    def broken_factory(_: ChaosMiddleware):
        raise RuntimeError("setup unavailable")

    harness = AuditScenario(
        check_id="HARNESS",
        category="infrastructure",
        severity="LOW",
        agent_factory=broken_factory,
        agent_input=AGENT_INPUT,
        policy=FaultPolicy("get_order_status", "failure"),
        evaluation_checks={"not reached": lambda _: False},
    )
    scenarios = (
        subject_scenario(
            "EXPECTED",
            lambda result: isinstance(result.faulted.exception, ExpectedAgentError),
        ),
        subject_scenario("VIOLATION", lambda result: result.faulted.exception is None),
        harness,
    )

    report = AuditRunner(scenarios, clock=lambda: TIMESTAMP).run(
        audit_id="AUDIT-MIXED",
        name="Mixed audit",
    )

    assert [finding.finding_id for finding in report.findings] == [
        "EXPECTED",
        "VIOLATION",
        "HARNESS",
    ]
    assert [finding.status for finding in report.findings] == [
        "PASS",
        "FAIL",
        "INCONCLUSIVE",
    ]


def test_base_exceptions_are_not_captured_as_subject_or_harness_failures() -> None:
    class StopAgent:
        def invoke(self, _: dict) -> None:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        AuditRunner(
            (
                AuditScenario(
                    check_id="STOP",
                    category="framework",
                    severity="INFO",
                    agent_factory=lambda _: StopAgent(),
                    agent_input={},
                    policy=FaultPolicy("unused", "unused"),
                    evaluation_checks={},
                ),
            ),
            clock=lambda: TIMESTAMP,
        ).run(audit_id="AUDIT-STOP", name="Stop audit")
