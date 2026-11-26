"""Trusted local audit configuration for the deterministic order agent."""

from agentchaos import AuditScenario, FaultPolicy
from order_agent import create_order_agent


def reports_injected_failure(result) -> bool:
    return (
        result.faulted.fault_triggered
        and result.faulted.output["messages"][-1].content
        == "Tool failed: Simulated order service failure."
    )


SCENARIOS = (
    AuditScenario(
        check_id="ORDER-TOOL-FAILURE-001",
        category="tool-reliability",
        severity="HIGH",
        agent_factory=create_order_agent,
        agent_input={
            "messages": [{"role": "user", "content": "Where is ORD-101?"}]
        },
        policy=FaultPolicy(
            tool_name="get_order_status",
            error_message="Simulated order service failure.",
        ),
        evaluation_checks={"agent reports tool failure": reports_injected_failure},
    ),
)
