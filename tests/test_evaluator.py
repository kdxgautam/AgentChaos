from agentchaos import (
    EvaluationResult,
    ExecutionResult,
    ScenarioEvaluator,
    ScenarioResult,
)


def scenario_result(*, fault_triggered: bool = True) -> ScenarioResult:
    return ScenarioResult(
        baseline=ExecutionResult(
            outcome="success",
            output={"mode": "baseline"},
            exception=None,
            trace_events=(),
            fault_triggered=False,
        ),
        faulted=ExecutionResult(
            outcome="success",
            output={"mode": "faulted"},
            exception=None,
            trace_events=(),
            fault_triggered=fault_triggered,
        ),
    )


def test_pass_requires_triggered_fault_and_all_checks_to_pass() -> None:
    scenario = scenario_result()
    evaluation = ScenarioEvaluator(
        {
            "baseline succeeded": lambda result: result.baseline.outcome == "success",
            "faulted output retained": lambda result: result.faulted.output
            == {"mode": "faulted"},
        }
    ).evaluate(scenario)

    assert evaluation.status == "PASS"
    assert [(check.name, check.status) for check in evaluation.checks] == [
        ("baseline succeeded", "PASS"),
        ("faulted output retained", "PASS"),
    ]
    assert evaluation.errors == ()
    assert evaluation.scenario is scenario


def test_failed_check_means_fail() -> None:
    evaluation = ScenarioEvaluator(
        {
            "passes": lambda _: True,
            "fails": lambda _: False,
        }
    ).evaluate(scenario_result())

    assert evaluation.status == "FAIL"
    assert [check.status for check in evaluation.checks] == ["PASS", "FAIL"]


def test_missing_fault_is_inconclusive_when_checks_pass() -> None:
    evaluation = ScenarioEvaluator({"passes": lambda _: True}).evaluate(
        scenario_result(fault_triggered=False)
    )

    assert evaluation.status == "INCONCLUSIVE"
    assert evaluation.checks[0].status == "PASS"


def test_no_checks_is_inconclusive() -> None:
    evaluation = ScenarioEvaluator().evaluate(scenario_result())

    assert evaluation.status == "INCONCLUSIVE"
    assert evaluation.checks == ()


def test_check_exception_is_captured_as_inconclusive() -> None:
    exception = RuntimeError("check crashed")

    def crash(_: ScenarioResult) -> bool:
        raise exception

    evaluation = ScenarioEvaluator({"crashes": crash}).evaluate(scenario_result())

    assert evaluation.status == "INCONCLUSIVE"
    assert evaluation.checks[0].status == "ERROR"
    assert evaluation.checks[0].exception is exception
    assert evaluation.errors == (evaluation.checks[0],)


def test_known_failure_is_not_hidden_by_check_errors() -> None:
    calls = []
    exception = ValueError("unexpected")

    def fail(_: ScenarioResult) -> bool:
        calls.append("fail")
        return False

    def crash(_: ScenarioResult) -> bool:
        calls.append("crash")
        raise exception

    def pass_after_error(_: ScenarioResult) -> bool:
        calls.append("pass")
        return True

    evaluation = ScenarioEvaluator(
        {
            "known failure": fail,
            "unexpected error": crash,
            "still runs": pass_after_error,
        }
    ).evaluate(scenario_result(fault_triggered=False))

    assert calls == ["fail", "crash", "pass"]
    assert evaluation.status == "FAIL"
    assert [check.status for check in evaluation.checks] == [
        "FAIL",
        "ERROR",
        "PASS",
    ]
    assert evaluation.errors[0].exception is exception


def test_evaluation_result_preserves_scenario_data() -> None:
    scenario = scenario_result()
    evaluation: EvaluationResult = ScenarioEvaluator(
        {"reads original": lambda result: result is scenario}
    ).evaluate(scenario)

    assert evaluation.status == "PASS"
    assert evaluation.scenario is scenario
    assert evaluation.scenario.baseline.output == {"mode": "baseline"}
    assert evaluation.scenario.faulted.output == {"mode": "faulted"}
