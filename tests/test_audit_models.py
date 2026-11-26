from datetime import UTC, datetime, timedelta, timezone
import json

import pytest

from agentchaos import AuditFinding, AuditReport, Evidence


def evidence() -> Evidence:
    return Evidence(
        observation="The recovered order count was measured.",
        expected={"orders": 1},
        actual={"orders": 2},
        trace_reference="trace:create-order-retry",
        artifact_reference="artifact:orders.sqlite",
    )


def finding(
    finding_id: str,
    *,
    status: str = "FAIL",
    severity: str = "HIGH",
) -> AuditFinding:
    return AuditFinding(
        finding_id=finding_id,
        check_name="duplicate order check",
        category="idempotency",
        status=status,  # type: ignore[arg-type]
        severity=severity,  # type: ignore[arg-type]
        explanation="Recovery created more orders than expected.",
        evidence=(evidence(),),
    )


def test_evidence_serializes_without_inventing_missing_values() -> None:
    observed = Evidence(
        observation="No trace reference was supplied.",
        expected=None,
        actual={"values": [1, True, None]},
    )

    assert observed.to_dict() == {
        "observation": "No trace reference was supplied.",
        "expected": None,
        "actual": {"values": [1, True, None]},
        "trace_reference": None,
        "artifact_reference": None,
    }
    json.dumps(observed.to_dict(), allow_nan=False)


def test_json_values_are_validated_and_copied() -> None:
    actual = {"orders": ["ORDER-001"]}
    observed = Evidence("Persisted orders", expected=[], actual=actual)
    actual["orders"].append("ORDER-002")

    assert observed.actual == {"orders": ["ORDER-001"]}
    with pytest.raises(ValueError, match="actual must be JSON-serializable"):
        Evidence("Invalid", expected=None, actual={"value": float("nan")})
    with pytest.raises(ValueError, match="expected must be JSON-serializable"):
        Evidence("Invalid", expected=object(), actual=None)  # type: ignore[arg-type]


def test_required_evidence_and_report_fields_are_validated() -> None:
    timestamp = datetime(2026, 11, 26, tzinfo=UTC)

    with pytest.raises(ValueError, match="observation"):
        Evidence(" ", expected=None, actual=None)
    with pytest.raises(ValueError, match="audit_id"):
        AuditReport("", "name", timestamp, timestamp)
    with pytest.raises(ValueError, match="name"):
        AuditReport("audit", " ", timestamp, timestamp)


@pytest.mark.parametrize("status", ["", "UNKNOWN", "pass"])
def test_invalid_status_is_rejected(status: str) -> None:
    with pytest.raises(ValueError, match="invalid audit status"):
        finding("AC-001", status=status)


@pytest.mark.parametrize("severity", ["", "URGENT", "high"])
def test_invalid_severity_is_rejected(severity: str) -> None:
    with pytest.raises(ValueError, match="invalid audit severity"):
        finding("AC-001", severity=severity)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("finding_id", ""),
        ("check_name", " "),
        ("category", ""),
        ("explanation", ""),
    ],
)
def test_required_finding_fields_are_validated(field_name: str, value: str) -> None:
    values = {
        "finding_id": "AC-001",
        "check_name": "check",
        "category": "recovery",
        "status": "PASS",
        "severity": "INFO",
        "explanation": "The check passed.",
    }
    values[field_name] = value

    with pytest.raises(ValueError, match=field_name):
        AuditFinding(**values)  # type: ignore[arg-type]


def test_report_serializes_multiple_ordered_findings_deterministically() -> None:
    started = datetime(2026, 11, 26, 10, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    report = AuditReport(
        audit_id="AUDIT-2026-001",
        name="Order recovery audit",
        started_at=started,
        completed_at=started + timedelta(seconds=3),
        findings=(
            finding("AC-001", status="PASS", severity="CRITICAL"),
            finding("AC-002", status="INCONCLUSIVE", severity="LOW"),
        ),
        metadata={"scenario": "process-crash", "attempt": 1},
    )

    serialized = report.to_dict()
    assert serialized["started_at"] == "2026-11-26T04:30:00Z"
    assert serialized["completed_at"] == "2026-11-26T04:30:03Z"
    assert [item["finding_id"] for item in serialized["findings"]] == [
        "AC-001",
        "AC-002",
    ]
    assert serialized["findings"][0]["status"] == "PASS"
    assert serialized["findings"][0]["severity"] == "CRITICAL"
    assert json.loads(report.to_json()) == serialized
    assert report.to_json() == report.to_json()


def test_report_validates_timestamps_and_finding_types() -> None:
    timestamp = datetime(2026, 11, 26, tzinfo=UTC)

    with pytest.raises(ValueError, match="timezone-aware"):
        AuditReport("audit", "name", timestamp.replace(tzinfo=None), timestamp)
    with pytest.raises(ValueError, match="completed_at"):
        AuditReport("audit", "name", timestamp, timestamp - timedelta(seconds=1))
    with pytest.raises(ValueError, match="AuditFinding"):
        AuditReport(
            "audit",
            "name",
            timestamp,
            timestamp,
            findings=("not a finding",),  # type: ignore[arg-type]
        )
