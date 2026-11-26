from pathlib import Path

import pytest

from agentchaos import (
    InvalidRepositoryPath,
    OversizedSourceError,
    RepositoryAnalyzer,
)


@pytest.fixture
def python_repository(tmp_path: Path) -> Path:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "module.py").write_text(
        """import os
from helpers import helper as h

@decorate
class Service:
    @staticmethod
    def run(self):
        h()
        os.path.join("a", "b")
""",
        encoding="utf-8",
    )
    (tmp_path / "helpers.py").write_text(
        """def helper():
    return "ok"

async def async_helper():
    return "ok"
""",
        encoding="utf-8",
    )
    (tmp_path / "broken.py").write_text("def broken(:\n", encoding="utf-8")
    excluded = tmp_path / ".venv"
    excluded.mkdir()
    (excluded / "ignored.py").write_text("def ignored(): pass\n", encoding="utf-8")
    return tmp_path


def test_builds_deterministic_symbol_and_source_index(
    python_repository: Path,
) -> None:
    analyzer = RepositoryAnalyzer(python_repository)

    assert analyzer.list_files() == (
        "broken.py",
        "helpers.py",
        "pkg/module.py",
    )
    assert analyzer.read_source("helpers.py").startswith('def helper():\n    return "ok"\n')
    service = analyzer.find_symbol("Service")[0]
    method = analyzer.find_symbol("Service.run")[0]
    assert (service.kind, service.location.path, service.location.line) == (
        "class",
        "pkg/module.py",
        5,
    )
    assert (method.name, method.kind, method.location.line, method.location.column) == (
        "run",
        "function",
        7,
        5,
    )
    assert analyzer.find_symbol("missing") == ()
    assert analyzer.find_symbol("async_helper")[0].kind == "async_function"


def test_indexes_imports_calls_and_decorators_as_syntax(
    python_repository: Path,
) -> None:
    analyzer = RepositoryAnalyzer(python_repository)

    imports = analyzer.get_imports("pkg/module.py")
    assert [(item.module, item.name, item.alias, item.location.line) for item in imports] == [
        ("os", None, None, 1),
        ("helpers", "helper", "h", 2),
    ]
    assert [
        (call.callee, call.caller, call.location.line)
        for call in analyzer.find_callers("h")
    ] == [("h", "Service.run", 8)]
    assert [call.callee for call in analyzer.find_callers("join")] == ["os.path.join"]
    assert analyzer.find_callers("helper") == ()
    assert [
        (item.symbol, item.expression, item.location.line)
        for item in analyzer.get_decorators()
    ] == [
        ("Service", "decorate", 4),
        ("Service.run", "staticmethod", 6),
    ]
    assert analyzer.get_decorators("run")[0].symbol == "Service.run"


def test_records_syntax_errors_without_executing_source(
    python_repository: Path,
) -> None:
    (python_repository / "dangerous.py").write_text(
        'raise RuntimeError("must not execute")\n', encoding="utf-8"
    )

    analyzer = RepositoryAnalyzer(python_repository)

    syntax_issue = next(issue for issue in analyzer.issues if issue.path == "broken.py")
    assert syntax_issue.kind == "syntax_error"
    assert syntax_issue.line == 1
    assert "dangerous.py" in analyzer.list_files()
    assert analyzer.read_source("dangerous.py").startswith("raise RuntimeError")


def test_handles_oversized_files_and_invalid_paths(tmp_path: Path) -> None:
    (tmp_path / "large.py").write_text("x = 1\n" * 20, encoding="utf-8")
    analyzer = RepositoryAnalyzer(tmp_path, max_file_size=20)

    assert analyzer.list_files() == ("large.py",)
    assert analyzer.issues[0].kind == "oversized"
    with pytest.raises(OversizedSourceError):
        analyzer.read_source("large.py")
    with pytest.raises(InvalidRepositoryPath):
        analyzer.read_source("../large.py")
    with pytest.raises(InvalidRepositoryPath):
        analyzer.read_source(tmp_path / "large.py")
    with pytest.raises(InvalidRepositoryPath):
        analyzer.get_imports("missing.py")


def test_excludes_custom_directories(python_repository: Path) -> None:
    generated = python_repository / "generated"
    generated.mkdir()
    (generated / "code.py").write_text("def generated(): pass\n", encoding="utf-8")

    analyzer = RepositoryAnalyzer(
        python_repository, excluded_directories={"generated"}
    )

    assert "generated/code.py" not in analyzer.list_files()


def test_rejects_unsafe_symlinks(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.py"
    outside.write_text("def outside(): pass\n", encoding="utf-8")
    link = tmp_path / "linked.py"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symbolic links are unavailable")

    analyzer = RepositoryAnalyzer(tmp_path)

    assert "linked.py" not in analyzer.list_files()
    assert [(issue.path, issue.kind) for issue in analyzer.issues] == [
        ("linked.py", "unsafe_symlink")
    ]


def test_repository_indexes_are_isolated(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "one.py").write_text("def shared(): pass\n", encoding="utf-8")
    (second / "two.py").write_text("def other(): pass\n", encoding="utf-8")

    first_index = RepositoryAnalyzer(first)
    second_index = RepositoryAnalyzer(second)

    assert first_index.list_files() == ("one.py",)
    assert second_index.list_files() == ("two.py",)
    assert first_index.find_symbol("shared")[0].location.path == "one.py"
    assert second_index.find_symbol("shared") == ()


def test_rejects_invalid_roots(tmp_path: Path) -> None:
    file_path = tmp_path / "file.py"
    file_path.write_text("pass\n", encoding="utf-8")

    with pytest.raises(InvalidRepositoryPath):
        RepositoryAnalyzer(tmp_path / "missing")
    with pytest.raises(InvalidRepositoryPath):
        RepositoryAnalyzer(file_path)
    with pytest.raises(ValueError, match="positive"):
        RepositoryAnalyzer(tmp_path, max_file_size=0)
