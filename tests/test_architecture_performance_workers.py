import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Self

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from pydantic import Field

from agentchaos import (
    AUDIT_TASKS,
    RepositoryAnalyzer,
    RepositoryAuditOrchestrator,
    create_architecture_worker,
    create_performance_worker,
)
from agentchaos.orchestration import WorkerHandler


class ScriptedSpecialistModel(BaseChatModel):
    responses: list[AIMessage]
    index: int = 0
    seen_messages: list[list[BaseMessage]] = Field(default_factory=list)
    bound_tools: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-specialist-model"

    def bind_tools(
        self,
        tools: list[dict[str, Any] | type | BaseTool],
        **kwargs: Any,
    ) -> Self:
        self.bound_tools = [
            tool.name if isinstance(tool, BaseTool) else str(tool) for tool in tools
        ]
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen_messages.append(list(messages))
        response = self.responses[self.index]
        self.index += 1
        return ChatResult(generations=[ChatGeneration(message=response)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


WorkerFactory = Callable[..., WorkerHandler]


@pytest.fixture
def specialist_repository(tmp_path: Path) -> RepositoryAnalyzer:
    (tmp_path / "graph.py").write_text(
        """from typing import Annotated, TypedDict
import operator

class State(TypedDict):
    events: Annotated[list[str], operator.add]

def route(state):
    return \"worker\"
""",
        encoding="utf-8",
    )
    (tmp_path / "service.py").write_text(
        """import requests

async def fetch_status(url):
    return requests.get(url)
""",
        encoding="utf-8",
    )
    return RepositoryAnalyzer(tmp_path)


def tool_call(path: str, end_line: int, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "read_python_source",
                "args": {"path": path, "start_line": 1, "end_line": end_line},
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


def final_response(findings: list[dict[str, Any]]) -> AIMessage:
    return AIMessage(content=json.dumps({"findings": findings}))


SPECIALISTS = (
    (
        "architecture",
        2,
        create_architecture_worker,
        "graph.py",
        8,
        {
            "claim": "The additive reducer may retain duplicate events across loop iterations.",
            "source_path": "graph.py",
            "start_line": 4,
            "end_line": 5,
            "evidence": "State.events uses operator.add as its reducer.",
            "uncertainty": "high",
        },
        "state schemas",
    ),
    (
        "performance",
        3,
        create_performance_worker,
        "service.py",
        4,
        {
            "claim": "A synchronous HTTP request may block the event loop.",
            "source_path": "service.py",
            "start_line": 3,
            "end_line": 4,
            "evidence": "The async function directly invokes requests.get without a timeout.",
            "uncertainty": "low",
        },
        "blocking calls",
    ),
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("role", "task_index", "factory", "path", "end_line", "finding", "prompt_term"),
    SPECIALISTS,
)
async def test_specialist_calls_tool_and_returns_typed_finding(
    specialist_repository: RepositoryAnalyzer,
    role: str,
    task_index: int,
    factory: WorkerFactory,
    path: str,
    end_line: int,
    finding: dict[str, Any],
    prompt_term: str,
) -> None:
    model = ScriptedSpecialistModel(
        responses=[tool_call(path, end_line, f"{role}-read"), final_response([finding])]
    )

    output = await factory(model)(AUDIT_TASKS[task_index], specialist_repository)

    assert output.candidate_findings[0].claim == finding["claim"]
    assert output.candidate_findings[0].uncertainty == finding["uncertainty"]
    assert "read_python_source" in model.bound_tools
    assert any(
        isinstance(message, ToolMessage)
        for messages in model.seen_messages
        for message in messages
    )
    assert prompt_term in str(model.seen_messages[0][0].content)
    assert "untrusted data" in str(model.seen_messages[0][0].content)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("role", "task_index", "factory", "_path", "_end_line", "finding", "_term"),
    SPECIALISTS,
)
async def test_specialist_allows_zero_findings_and_rejects_invalid_citations(
    specialist_repository: RepositoryAnalyzer,
    role: str,
    task_index: int,
    factory: WorkerFactory,
    _path: str,
    _end_line: int,
    finding: dict[str, Any],
    _term: str,
) -> None:
    benign = await factory(ScriptedSpecialistModel(responses=[final_response([])]))(
        AUDIT_TASKS[task_index], specialist_repository
    )
    invalid = finding | {"source_path": "missing.py"}
    rejected = await factory(
        ScriptedSpecialistModel(responses=[final_response([invalid])])
    )(AUDIT_TASKS[task_index], specialist_repository)

    assert benign.candidate_findings == ()
    assert benign.observations == ("candidate_findings=0", "rejected_findings=0")
    assert rejected.candidate_findings == ()
    assert rejected.observations == ("candidate_findings=0", "rejected_findings=1")


@pytest.mark.asyncio
async def test_architecture_and_performance_workers_integrate_with_supervisor(
    specialist_repository: RepositoryAnalyzer,
) -> None:
    architecture_finding = SPECIALISTS[0][5]
    performance_finding = SPECIALISTS[1][5]
    results = await RepositoryAuditOrchestrator(
        {
            "architecture": create_architecture_worker(
                ScriptedSpecialistModel(
                    responses=[final_response([architecture_finding])]
                )
            ),
            "performance": create_performance_worker(
                ScriptedSpecialistModel(
                    responses=[final_response([performance_finding])]
                )
            ),
        }
    ).arun(specialist_repository)

    assert [result.role for result in results] == [
        "security",
        "reliability",
        "architecture",
        "performance",
    ]
    assert [result.outcome for result in results] == ["success"] * 4
    assert results[0].candidate_findings == results[1].candidate_findings == ()
    assert results[2].candidate_findings[0].source_path == "graph.py"
    assert results[3].candidate_findings[0].source_path == "service.py"
