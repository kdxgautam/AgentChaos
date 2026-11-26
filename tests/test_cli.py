import json
from pathlib import Path

import pytest

from agentchaos.cli import _exit_code, load_scenarios, main
from examples.order_agent import executed_order_ids


def configuration(status: str = "PASS") -> str:
    policy_tool = "get_order_status" if status != "INCONCLUSIVE" else "other_tool"
    check_result = "False" if status == "FAIL" else "True"
    return f'''\
from agentchaos import AuditScenario, FaultPolicy
from examples.order_agent import create_order_agent

SCENARIOS = (
    AuditScenario(
        check_id="CLI-{status}",
        category="cli-test",
        severity="LOW",
        agent_factory=create_order_agent,
        agent_input={{"messages": [{{"role": "user", "content": "Where is ORD-101?"}}]}},
        policy=FaultPolicy("{policy_tool}", "failure"),
        evaluation_checks={{"configured check": lambda _: {check_result}}},
    ),
)
'''


@pytest.fixture(autouse=True)
def clear_executions() -> None:
    executed_order_ids.clear()


def write_configuration(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "audits.py"
    path.write_text(content, encoding="utf-8")
    return path


def test_loads_trusted_python_configuration(tmp_path: Path) -> None:
    path = write_configuration(tmp_path, configuration())

    scenarios = load_scenarios(path)

    assert len(scenarios) == 1
    assert scenarios[0].check_id == "CLI-PASS"


@pytest.mark.parametrize(
    ("status", "expected_exit"),
    [("PASS", 0), ("FAIL", 1), ("INCONCLUSIVE", 2)],
)
def test_cli_exit_codes_summary_and_json_output(
    tmp_path: Path,
    capsys,
    status: str,
    expected_exit: int,
) -> None:
    path = write_configuration(tmp_path, configuration(status))
    output = tmp_path / "report.json"

    exit_code = main(["audit", str(path), "--output", str(output)])

    assert exit_code == expected_exit
    assert capsys.readouterr().out == (
        "Audit: "
        f"PASS={int(status == 'PASS')} "
        f"FAIL={int(status == 'FAIL')} "
        f"INCONCLUSIVE={int(status == 'INCONCLUSIVE')} "
        "NOT_RUN=0\n"
    )
    serialized = json.loads(output.read_text(encoding="utf-8"))
    assert serialized["findings"][0]["status"] == status
    assert serialized["findings"][0]["finding_id"] == f"CLI-{status}"
    assert output.read_text(encoding="utf-8").endswith("\n")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("VALUE = 1", "SCENARIOS must be a list or tuple"),
        ("SCENARIOS = []", "SCENARIOS must not be empty"),
        ("SCENARIOS = ['invalid']", "only AuditScenario"),
        ("this is not python!", "configuration raised SyntaxError"),
    ],
)
def test_malformed_configuration_returns_two(
    tmp_path: Path,
    capsys,
    content: str,
    message: str,
) -> None:
    path = write_configuration(tmp_path, content)

    assert main(["audit", str(path), "--output", str(tmp_path / "report.json")]) == 2
    assert message in capsys.readouterr().err


def test_missing_configuration_and_output_errors_return_two(
    tmp_path: Path,
    capsys,
) -> None:
    missing = tmp_path / "missing.py"
    assert main(["audit", str(missing), "--output", str(tmp_path / "report.json")]) == 2
    assert "configuration file not found" in capsys.readouterr().err

    path = write_configuration(tmp_path, configuration())
    assert main(["audit", str(path), "--output", str(tmp_path)]) == 2
    assert "cannot write" in capsys.readouterr().err


def test_not_run_without_failure_uses_exit_two() -> None:
    assert _exit_code(
        {"PASS": 0, "FAIL": 0, "INCONCLUSIVE": 0, "NOT_RUN": 1}
    ) == 2
    assert _exit_code(
        {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 1, "NOT_RUN": 1}
    ) == 1
