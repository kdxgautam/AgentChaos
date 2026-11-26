from datetime import UTC, datetime
import json

import pytest

from agentchaos import AuditRunner, AuditScenario, FaultPolicy
from examples.order_agent import create_order_agent, executed_order_ids


AGENT_INPUT = {"messages": [{"role": "user", "content": "Where is ORD-101?"}]}
TIMESTAMP = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def clear_executions() -> None:
    executed_order_ids.clear()


def scenario(
    check_id: str,
    *,
    policy_tool: str = "get_order_status",
    check=lambda _: True,
) -> AuditScenario:
    return AuditScenario(
        check_id=check_id,
        category="tool-reliability",
        severity="HIGH",
        agent_factory=create_order_agent,
        agent_input=AGENT_INPUT,
        policy=FaultPolicy(policy_tool, "simulated failure"),
        evaluation_checks={"configured check": check},
    )


def test_runner_preserves_order_and_evaluation_semantics() -> None:
    runner = AuditRunner(
        (
            scenario("CHECK-PASS"),
            scenario("CHECK-FAIL", check=lambda _: False),
            scenario("CHECK-INCONCLUSIVE", policy_tool="unrelated_tool"),
        ),
        clock=lambda: TIMESTAMP,
    )

    report = runner.run(
        audit_id="AUDIT-001",
        name="Sequential reliability audit",
        metadata={"environment": "test"},
    )

    assert [finding.finding_id for finding in report.findings] == [
        "CHECK-PASS",
        "CHECK-FAIL",
        "CHECK-INCONCLUSIVE",
    ]
    assert [finding.status for finding in report.findings] == [
        "PASS",
        "FAIL",
        "INCONCLUSIVE",
    ]
    assert [finding.category for finding in report.findings] == [
        "tool-reliability",
        "tool-reliability",
        "tool-reliability",
    ]
    assert [finding.severity for finding in report.findings] == [
        "HIGH",
        "HIGH",
        "HIGH",
    ]
    assert executed_order_ids == ["ORD-101"] * 4


def test_each_scenario_uses_fresh_agents_and_copied_inputs() -> None:
    factory_calls: list[tuple[str, object]] = []
    original_input = {"scenario": "isolated"}

    class Agent:
        def __init__(self, marker: object) -> None:
            self.marker = marker

        def invoke(self, agent_input: dict) -> dict:
            agent_input["mutated"] = True
            return {"marker": self.marker}

    def factory(middleware):
        marker = object()
        factory_calls.append(("created", marker))
        return Agent(marker)

    configured = AuditScenario(
        check_id="CHECK-ISOLATION",
        category="isolation",
        severity="MEDIUM",
        agent_factory=factory,
        agent_input=original_input,
        policy=FaultPolicy("unused", "unused"),
        evaluation_checks={"input stayed isolated": lambda _: True},
    )

    report = AuditRunner((configured,), clock=lambda: TIMESTAMP).run(
        audit_id="AUDIT-ISOLATION",
        name="Isolation audit",
    )

    assert len(factory_calls) == 2
    assert factory_calls[0][1] is not factory_calls[1][1]
    assert original_input == {"scenario": "isolated"}
    assert report.findings[0].status == "INCONCLUSIVE"


def test_setup_errors_are_inconclusive_and_skip_reliability_checks() -> None:
    check_calls = 0

    def broken_factory(_):
        raise RuntimeError("dependency setup failed")

    def should_not_run(_):
        nonlocal check_calls
        check_calls += 1
        return False

    configured = AuditScenario(
        check_id="CHECK-SETUP",
        category="infrastructure",
        severity="CRITICAL",
        agent_factory=broken_factory,
        agent_input={},
        policy=FaultPolicy("tool", "failure"),
        evaluation_checks={"would fail": should_not_run},
    )

    finding = AuditRunner((configured,), clock=lambda: TIMESTAMP).run(
        audit_id="AUDIT-SETUP",
        name="Setup audit",
    ).findings[0]

    assert check_calls == 0
    assert finding.status == "INCONCLUSIVE"
    assert "reliability was not assessed" in finding.explanation
    errors = [
        item
        for item in finding.evidence
        if item.observation.endswith("harness failure")
    ]
    assert [item.actual for item in errors] == [
        {
            "phase": "baseline",
            "provenance": "harness",
            "type": "RuntimeError",
            "message": "dependency setup failed",
        },
        {
            "phase": "faulted",
            "provenance": "harness",
            "type": "RuntimeError",
            "message": "dependency setup failed",
        },
    ]


def test_check_exceptions_remain_inconclusive() -> None:
    def crash(_):
        raise LookupError("check failed")

    finding = AuditRunner(
        (scenario("CHECK-ERROR", check=crash),), clock=lambda: TIMESTAMP
    ).run(audit_id="AUDIT-ERROR", name="Check error audit").findings[0]

    assert finding.status == "INCONCLUSIVE"
    check = next(
        item for item in finding.evidence if item.observation == "Evaluation check: configured check"
    )
    assert check.actual["exception"] == {
        "type": "LookupError",
        "message": "check failed",
    }


def test_report_json_is_deterministic() -> None:
    report = AuditRunner((scenario("CHECK-JSON"),), clock=lambda: TIMESTAMP).run(
        audit_id="AUDIT-JSON",
        name="JSON audit",
        metadata={"z": 2, "a": 1},
    )

    first = report.to_json()
    second = report.to_json()

    assert first == second
    serialized = json.loads(first)
    assert serialized["audit_id"] == "AUDIT-JSON"
    assert serialized["findings"][0]["status"] == "PASS"
    assert serialized["metadata"] == {"a": 1, "z": 2}
