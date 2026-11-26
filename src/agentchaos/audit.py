"""Typed, deterministic audit data models."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
from typing import Any, Literal, TypeAlias, cast


JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
AuditStatus: TypeAlias = Literal["PASS", "FAIL", "INCONCLUSIVE", "NOT_RUN"]
AuditSeverity: TypeAlias = Literal["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"]

_STATUSES = {"PASS", "FAIL", "INCONCLUSIVE", "NOT_RUN"}
_SEVERITIES = {"INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"}


def _required(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _canonical_json(value: Any, field_name: str) -> JSONValue:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exception:
        raise ValueError(f"{field_name} must be JSON-serializable") from exception
    return cast(JSONValue, json.loads(encoded))


def _timestamp(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{field_name} must be a timezone-aware datetime")
    return value.astimezone(UTC)


def _timestamp_json(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class Evidence:
    """One observed result and the expectation it was compared with."""

    observation: str
    expected: JSONValue
    actual: JSONValue
    trace_reference: str | None = None
    artifact_reference: str | None = None

    def __post_init__(self) -> None:
        _required(self.observation, "observation")
        if self.trace_reference is not None:
            _required(self.trace_reference, "trace_reference")
        if self.artifact_reference is not None:
            _required(self.artifact_reference, "artifact_reference")
        object.__setattr__(self, "expected", _canonical_json(self.expected, "expected"))
        object.__setattr__(self, "actual", _canonical_json(self.actual, "actual"))

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "observation": self.observation,
            "expected": self.expected,
            "actual": self.actual,
            "trace_reference": self.trace_reference,
            "artifact_reference": self.artifact_reference,
        }


@dataclass(frozen=True, slots=True)
class AuditFinding:
    """The deterministic outcome of one named audit check."""

    finding_id: str
    check_name: str
    category: str
    status: AuditStatus
    severity: AuditSeverity
    explanation: str
    evidence: tuple[Evidence, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("finding_id", "check_name", "category", "explanation"):
            _required(getattr(self, field_name), field_name)
        if not isinstance(self.status, str) or self.status not in _STATUSES:
            raise ValueError(f"invalid audit status: {self.status}")
        if not isinstance(self.severity, str) or self.severity not in _SEVERITIES:
            raise ValueError(f"invalid audit severity: {self.severity}")
        evidence = tuple(self.evidence)
        if any(not isinstance(item, Evidence) for item in evidence):
            raise ValueError("evidence must contain only Evidence values")
        object.__setattr__(self, "evidence", evidence)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "finding_id": self.finding_id,
            "check_name": self.check_name,
            "category": self.category,
            "status": self.status,
            "severity": self.severity,
            "explanation": self.explanation,
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass(frozen=True, slots=True)
class AuditReport:
    """Audit metadata, execution timestamps, and ordered findings."""

    audit_id: str
    name: str
    started_at: datetime
    completed_at: datetime
    findings: tuple[AuditFinding, ...] = ()
    metadata: dict[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _required(self.audit_id, "audit_id")
        _required(self.name, "name")
        started_at = _timestamp(self.started_at, "started_at")
        completed_at = _timestamp(self.completed_at, "completed_at")
        if completed_at < started_at:
            raise ValueError("completed_at must not be earlier than started_at")
        findings = tuple(self.findings)
        if any(not isinstance(item, AuditFinding) for item in findings):
            raise ValueError("findings must contain only AuditFinding values")
        metadata = _canonical_json(self.metadata, "metadata")
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a JSON object")
        object.__setattr__(self, "started_at", started_at)
        object.__setattr__(self, "completed_at", completed_at)
        object.__setattr__(self, "findings", findings)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "audit_id": self.audit_id,
            "name": self.name,
            "started_at": _timestamp_json(self.started_at),
            "completed_at": _timestamp_json(self.completed_at),
            "metadata": self.metadata,
            "findings": [item.to_dict() for item in self.findings],
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
