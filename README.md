# AgentChaos

AgentChaos is a Python SDK for chaos engineering and reliability testing of AI agents.

Milestone 1 proves that a deterministic LangChain agent can issue a real tool call and that LangGraph executes the underlying Python function. The example requires no API keys or network requests at runtime.

## Run

```bash
uv sync
uv run python examples/order_agent.py
uv run pytest
```

The model's first response requests `get_order_status(order_id="ORD-101")`. LangGraph then runs the registered Python function and adds its return value as a `ToolMessage`. The deterministic model reads that message and produces the final response.

## Limitations

- Fault injection and AgentChaos middleware are intentionally not implemented in this milestone.
- The fake model covers one fixed tool-call path; it does not interpret arbitrary user prompts.
- The order data is hard-coded and the example has no persistence or external integrations.
- Tool calls and results are observable in agent message history; interception and alteration remain future work.
