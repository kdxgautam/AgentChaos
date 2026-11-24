"""In-memory tool-call tracing."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal


TraceEventType = Literal[
    "start",
    "success",
    "injected_failure",
    "response_replaced",
    "returned_tool_error",
    "raised_exception",
]


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """One point in a tool call's execution."""

    event: TraceEventType
    tool_name: str
    call_id: str
    timestamp: datetime
    exception_type: str | None = None


class TraceCollector:
    """Collect tool-call trace events in memory."""

    def __init__(self) -> None:
        self._events: list[TraceEvent] = []

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        return tuple(self._events)

    def _record(
        self,
        event: TraceEventType,
        tool_name: str,
        call_id: str,
        exception_type: str | None = None,
    ) -> None:
        self._events.append(
            TraceEvent(
                event=event,
                tool_name=tool_name,
                call_id=call_id,
                timestamp=datetime.now(UTC),
                exception_type=exception_type,
            )
        )
