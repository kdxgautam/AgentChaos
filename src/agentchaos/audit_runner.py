"""Sequential execution of explicitly configured audit scenarios."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from agentchaos.audit import (
    AuditFinding,
    AuditReport,
    AuditSeverity,
    Evidence,
    JSONValue,
)
from agentchaos.audit_adapter import evaluation_to_audit_finding
from agentchaos.evaluator import ScenarioCheck, ScenarioEvaluator
from agentchaos.middleware import ChaosMiddleware, FaultPolicy
from agentchaos.scenario import ScenarioResult, ScenarioRunner


@dataclass(frozen=True, slots=True)
class AuditScenario:
    """One explicitly configured scenario and its audit labels."""

    check_id: str
    category: str
    severity: AuditSeverity
    agent_factory: Callable[[ChaosMiddleware], Any]
    agent_input: Any
    policy: FaultPolicy
    evaluation_checks: Mapping[str, ScenarioCheck]

    def __post_init__(self) -> None:
        object.__setattr__(self, "evaluation_checks", dict(self.evaluation_checks))


def _now() -> datetime:
    return datetime.now(UTC)


def _harness_errors(result: ScenarioResult) -> tuple[Evidence, ...]:
    evidence: list[Evidence] = []
    for phase, execution in (
        ("baseline", result.baseline),
        ("faulted", result.faulted),
    ):
        if execution.exception_provenance != "harness":
            continue
        exception = execution.exception
        evidence.append(
            Evidence(
                observation=f"{phase} harness failure",
                expected=None,
                actual={
                    "phase": phase,
                    "provenance": "harness",
                    "type": type(exception).__name__ if exception else None,
                    "message": str(exception) if exception else None,
                },
            )
        )
    return tuple(evidence)


class AuditRunner:
    """Run configured scenarios sequentially and collect one audit report."""

    def __init__(
        self,
        scenarios: Sequence[AuditScenario],
        *,
        clock: Callable[[], datetime] = _now,
    ) -> None:
        self.scenarios = tuple(scenarios)
        self.clock = clock

    def run(
        self,
        *,
        audit_id: str,
        name: str,
        metadata: Mapping[str, JSONValue] | None = None,
    ) -> AuditReport:
        started_at = self.clock()
        findings = tuple(self._run_scenario(scenario) for scenario in self.scenarios)
        return AuditReport(
            audit_id=audit_id,
            name=name,
            started_at=started_at,
            completed_at=self.clock(),
            findings=findings,
            metadata=dict(metadata or {}),
        )

    @staticmethod
    def _run_scenario(scenario: AuditScenario) -> AuditFinding:
        result = ScenarioRunner(scenario.agent_factory).run(
            scenario.agent_input,
            scenario.policy,
        )
        harness_errors = _harness_errors(result)
        evaluator = (
            ScenarioEvaluator()
            if harness_errors
            else ScenarioEvaluator(scenario.evaluation_checks)
        )
        evaluation = evaluator.evaluate(result)
        finding = evaluation_to_audit_finding(
            evaluation,
            check_id=scenario.check_id,
            category=scenario.category,
            severity=scenario.severity,
        )
        if not harness_errors:
            return finding
        return AuditFinding(
            finding_id=finding.finding_id,
            check_name=finding.check_name,
            category=finding.category,
            status="INCONCLUSIVE",
            severity=finding.severity,
            explanation="AgentChaos harness setup did not complete; reliability was not assessed.",
            evidence=(*finding.evidence, *harness_errors),
        )
