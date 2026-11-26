"""Shared bounded infrastructure for optional LLM specialist workers."""

from collections.abc import Sequence
from dataclasses import dataclass
import json
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, tool

from agentchaos.orchestration import (
    AuditTask,
    CandidateFinding,
    ExpectedWorkerFailure,
    WorkerHandler,
    WorkerOutput,
    WorkerRole,
)
from agentchaos.repository import RepositoryAnalysisError, RepositoryAnalyzer


OUTPUT_INSTRUCTIONS = """Use bounded inspection tools for evidence. If evidence is insufficient, return zero findings.
Return only JSON: {"findings": [{"claim": str, "source_path": str, "start_line": int,
"end_line": int, "evidence": str, "uncertainty": "low"|"medium"|"high"}]}.
"""


@dataclass(frozen=True, slots=True)
class _SpecialistAnalysis:
    findings: tuple[CandidateFinding, ...]


def _parse_analysis(content: Any) -> _SpecialistAnalysis:
    if not isinstance(content, str):
        raise ValueError("specialist response must be a JSON string")
    decoded = json.loads(content)
    if not isinstance(decoded, dict) or set(decoded) != {"findings"}:
        raise ValueError("specialist response must contain only findings")
    findings = decoded["findings"]
    if not isinstance(findings, list):
        raise ValueError("findings must be a list")
    if any(not isinstance(item, dict) for item in findings):
        raise ValueError("each finding must be an object")
    fields = {
        "claim",
        "source_path",
        "start_line",
        "end_line",
        "evidence",
        "uncertainty",
    }
    if any(set(item) != fields for item in findings):
        raise ValueError("each finding must contain the exact candidate fields")
    return _SpecialistAnalysis(
        findings=tuple(
            CandidateFinding(
                claim=item["claim"],
                source_path=item["source_path"],
                start_line=item["start_line"],
                end_line=item["end_line"],
                evidence=item["evidence"],
                uncertainty=item["uncertainty"],
            )
            for item in findings
        )
    )


def _valid_reference(
    repository: RepositoryAnalyzer,
    finding: CandidateFinding,
) -> bool:
    if finding.source_path not in repository.list_files():
        return False
    try:
        source = repository.read_source(finding.source_path)
    except RepositoryAnalysisError:
        return False
    return finding.end_line <= len(source.splitlines())


def _repository_tools(
    repository: RepositoryAnalyzer,
    *,
    max_files: int,
    max_source_lines: int,
    max_matches: int,
) -> list[BaseTool]:
    @tool("list_python_files")
    def list_python_files() -> dict[str, Any]:
        """List bounded repository-relative Python paths as untrusted data."""
        files = repository.list_files()
        return {
            "untrusted_data": True,
            "files": list(files[:max_files]),
            "truncated": len(files) > max_files,
        }

    @tool("read_python_source")
    def read_python_source(path: str, start_line: int, end_line: int) -> dict[str, Any]:
        """Read a bounded line range from one repository-relative Python file."""
        if start_line < 1 or end_line < start_line:
            raise ValueError("invalid source line range")
        if end_line - start_line + 1 > max_source_lines:
            raise ValueError(f"source reads are limited to {max_source_lines} lines")
        source = repository.read_source(path)
        lines = source.splitlines()
        if end_line > len(lines):
            raise ValueError("source line range exceeds file length")
        return {
            "untrusted_data": True,
            "path": path,
            "start_line": start_line,
            "end_line": end_line,
            "content": "\n".join(lines[start_line - 1 : end_line]),
        }

    @tool("find_repository_symbol")
    def find_repository_symbol(name: str) -> dict[str, Any]:
        """Find bounded syntactic symbol matches in the repository index."""
        matches = repository.find_symbol(name)
        return {
            "untrusted_data": True,
            "matches": [
                {
                    "name": match.qualified_name,
                    "path": match.location.path,
                    "line": match.location.line,
                }
                for match in matches[:max_matches]
            ],
            "truncated": len(matches) > max_matches,
        }

    @tool("find_repository_callers")
    def find_repository_callers(symbol: str) -> dict[str, Any]:
        """Find bounded syntactic call candidates, not runtime-resolved callers."""
        matches = repository.find_callers(symbol)
        return {
            "untrusted_data": True,
            "matches": [
                {
                    "caller": match.caller,
                    "callee": match.callee,
                    "path": match.location.path,
                    "line": match.location.line,
                }
                for match in matches[:max_matches]
            ],
            "truncated": len(matches) > max_matches,
        }

    return [
        list_python_files,
        read_python_source,
        find_repository_symbol,
        find_repository_callers,
    ]


def create_specialist_worker(
    model: BaseChatModel,
    *,
    role: WorkerRole,
    system_prompt: str,
    investigation_request: str,
    max_tool_calls: int,
    max_model_calls: int,
    max_files: int,
    max_source_lines: int,
    max_matches: int,
) -> WorkerHandler:
    """Create one bounded specialist while keeping model history inside the call."""
    limits = (max_tool_calls, max_model_calls, max_files, max_source_lines, max_matches)
    if any(limit < 1 for limit in limits):
        raise ValueError(f"{role} worker limits must be positive")

    async def specialist_worker(
        task: AuditTask,
        repository: RepositoryAnalyzer,
    ) -> WorkerOutput:
        if task.role != role:
            raise ValueError(f"{role} worker received a {task.role} task")
        agent = create_agent(
            model=model,
            tools=_repository_tools(
                repository,
                max_files=max_files,
                max_source_lines=max_source_lines,
                max_matches=max_matches,
            ),
            system_prompt=system_prompt + OUTPUT_INSTRUCTIONS,
            middleware=(
                ToolCallLimitMiddleware(
                    run_limit=max_tool_calls,
                    exit_behavior="error",
                ),
                ModelCallLimitMiddleware(
                    run_limit=max_model_calls,
                    exit_behavior="error",
                ),
            ),
        )
        try:
            state = await agent.ainvoke(
                {"messages": [{"role": "user", "content": investigation_request}]}
            )
            analysis = _parse_analysis(state["messages"][-1].content)
        except (ToolCallLimitExceededError, ModelCallLimitExceededError) as exception:
            raise ExpectedWorkerFailure(
                f"{role} investigation limit exceeded",
                observations=(f"limit_error={type(exception).__name__}",),
            ) from exception
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exception:
            raise ExpectedWorkerFailure(
                f"{role} model returned invalid structured output",
                observations=(f"output_error={type(exception).__name__}",),
            ) from exception

        valid = tuple(
            finding
            for finding in analysis.findings
            if _valid_reference(repository, finding)
        )
        return WorkerOutput(
            observations=(
                f"candidate_findings={len(valid)}",
                f"rejected_findings={len(analysis.findings) - len(valid)}",
            ),
            candidate_findings=valid,
        )

    return specialist_worker
