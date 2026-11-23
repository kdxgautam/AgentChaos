# AgentChaos

AgentChaos is a Python SDK for chaos engineering and reliability testing of AI agents.

Milestone 1 proves that a deterministic LangChain agent can issue a real tool call and that public LangChain middleware can intercept it before execution. The reusable `ChaosMiddleware` accepts an immutable `FaultPolicy` and an optional in-memory `TraceCollector`. `ScenarioRunner` compares isolated baseline and faulted executions using an application-provided agent factory, and `ScenarioEvaluator` applies deterministic named checks to those results. The example requires no API keys or network requests at runtime.

## Run

```bash
uv sync
uv run python examples/order_agent.py
uv run python examples/reliability_demo.py
uv run pytest
```

The model's first response requests `get_order_status(order_id="ORD-101")`. With no policy—or a policy targeting another tool—the registered Python function runs normally. A matching policy short-circuits before execution and returns an error `ToolMessage`; the model reports that failure instead of success.

The Milestone 8 reliability demonstration compares an unsafe V1 agent that fabricates a status after the primary service fails with a V2 agent that calls an independent backup tool. The same deterministic evaluator marks V1 `FAIL` and V2 `PASS`.

## Limitations

- Each middleware instance supports one exact tool-name policy.
- A single scenario runs its baseline and faulted executions sequentially; separate async scenarios can run concurrently.
- ScenarioRunner isolates agents, collectors, traces, and copied inputs, but not mutable dependencies captured by an agent factory. Use separate factory closures or another per-scenario provider for isolated in-memory state.
- ScenarioEvaluator treats check exceptions as inconclusive unless another check has a known failure; it does not assign scores or infer correctness beyond configured checks.
- Trace events live only in their collector instance and are lost when it is discarded.
- The fake model covers one fixed tool-call path; it does not interpret arbitrary user prompts.
- The order data is hard-coded and the example has no persistence or external integrations.
- There are no retries, delays, timeouts, or generalized fault types.
