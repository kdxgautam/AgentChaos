"""Deterministic evaluation of scenario results."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

from agentchaos.scenario import ScenarioResult


EvaluationStatus = Literal["PASS", "FAIL", "INCONCLUSIVE"]
CheckStatus = Literal["PASS", "FAIL", "ERROR"]
ScenarioCheck = Callable[[ScenarioResult], bool]


@dataclass(frozen=True, slots=True)
class CheckOutcome:
    """The outcome of one named scenario check."""

    name: str
    status: CheckStatus
    exception: Exception | None = None


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """The overall evaluation and its individual check outcomes."""

    status: EvaluationStatus
    checks: tuple[CheckOutcome, ...]
    scenario: ScenarioResult

    @property
    def errors(self) -> tuple[CheckOutcome, ...]:
        return tuple(check for check in self.checks if check.status == "ERROR")


class ScenarioEvaluator:
    """Evaluate a scenario with deterministic named checks."""

    def __init__(self, checks: Mapping[str, ScenarioCheck] | None = None) -> None:
        self.checks = tuple((checks or {}).items())

    def evaluate(self, scenario: ScenarioResult) -> EvaluationResult:
        outcomes = tuple(
            self._run_check(name, check, scenario) for name, check in self.checks
        )

        if any(outcome.status == "FAIL" for outcome in outcomes):
            status: EvaluationStatus = "FAIL"
        elif (
            not outcomes
            or not scenario.faulted.fault_triggered
            or any(outcome.status == "ERROR" for outcome in outcomes)
        ):
            status = "INCONCLUSIVE"
        else:
            status = "PASS"

        return EvaluationResult(status=status, checks=outcomes, scenario=scenario)

    @staticmethod
    def _run_check(
        name: str,
        check: ScenarioCheck,
        scenario: ScenarioResult,
    ) -> CheckOutcome:
        try:
            passed = check(scenario)
        except Exception as exception:
            return CheckOutcome(name=name, status="ERROR", exception=exception)
        return CheckOutcome(name=name, status="PASS" if passed else "FAIL")
