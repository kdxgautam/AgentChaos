"""Local command-line entry point for AgentChaos audits."""

import argparse
from collections import Counter
import importlib.util
from pathlib import Path
import sys
from typing import Sequence

from agentchaos.audit import AuditReport, AuditStatus
from agentchaos.audit_runner import AuditRunner, AuditScenario


class ConfigurationError(ValueError):
    """A trusted Python audit file does not export valid scenarios."""


def load_scenarios(configuration: Path) -> tuple[AuditScenario, ...]:
    path = configuration.resolve()
    if not path.is_file():
        raise ConfigurationError(f"configuration file not found: {configuration}")

    module_name = "_agentchaos_local_audit_config"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ConfigurationError(f"cannot load configuration: {configuration}")
    module = importlib.util.module_from_spec(spec)
    parent = str(path.parent)
    added_path = parent not in sys.path
    if added_path:
        sys.path.insert(0, parent)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exception:
        raise ConfigurationError(
            f"configuration raised {type(exception).__name__}: {exception}"
        ) from exception
    finally:
        sys.modules.pop(module_name, None)
        if added_path:
            sys.path.remove(parent)

    configured = getattr(module, "SCENARIOS", None)
    if not isinstance(configured, (list, tuple)):
        raise ConfigurationError("SCENARIOS must be a list or tuple of AuditScenario")
    scenarios = tuple(configured)
    if not scenarios:
        raise ConfigurationError("SCENARIOS must not be empty")
    if any(not isinstance(item, AuditScenario) for item in scenarios):
        raise ConfigurationError("SCENARIOS must contain only AuditScenario objects")
    return scenarios


def _counts(report: AuditReport) -> dict[AuditStatus, int]:
    observed = Counter(finding.status for finding in report.findings)
    return {
        "PASS": observed["PASS"],
        "FAIL": observed["FAIL"],
        "INCONCLUSIVE": observed["INCONCLUSIVE"],
        "NOT_RUN": observed["NOT_RUN"],
    }


def _exit_code(counts: dict[AuditStatus, int]) -> int:
    if counts["FAIL"]:
        return 1
    if counts["INCONCLUSIVE"] or counts["NOT_RUN"]:
        return 2
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentchaos")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="run a trusted local audit file")
    audit.add_argument("configuration", type=Path)
    audit.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        scenarios = load_scenarios(args.configuration)
        report = AuditRunner(scenarios).run(
            audit_id=f"local-audit:{args.configuration.stem}",
            name=f"Local audit: {args.configuration.name}",
            metadata={"configuration": args.configuration.name},
        )
    except (ConfigurationError, ValueError) as exception:
        print(f"agentchaos: {exception}", file=sys.stderr)
        return 2

    try:
        args.output.write_text(report.to_json() + "\n", encoding="utf-8")
    except OSError as exception:
        print(f"agentchaos: cannot write {args.output}: {exception}", file=sys.stderr)
        return 2

    counts = _counts(report)
    print(
        "Audit: "
        f"PASS={counts['PASS']} "
        f"FAIL={counts['FAIL']} "
        f"INCONCLUSIVE={counts['INCONCLUSIVE']} "
        f"NOT_RUN={counts['NOT_RUN']}"
    )
    return _exit_code(counts)


if __name__ == "__main__":
    raise SystemExit(main())
