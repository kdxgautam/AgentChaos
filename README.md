# AgentChaos

AgentChaos is a Python SDK for chaos engineering and reliability testing of AI agents.

Milestone 1 proves that a deterministic LangChain agent can issue a real tool call and that public LangChain middleware can intercept it before execution. The reusable `ChaosMiddleware` accepts an immutable `FaultPolicy` and an optional in-memory `TraceCollector`. `ScenarioRunner` compares isolated baseline and faulted executions using an application-provided agent factory, and `ScenarioEvaluator` applies deterministic named checks to those results. The example requires no API keys or network requests at runtime.

## Run

```bash
uv sync
uv run python examples/order_agent.py
uv run python examples/reliability_demo.py
uv run python examples/idempotency_demo.py
uv run python examples/checkpoint_recovery_demo.py
uv run python examples/process_crash_recovery_demo.py
uv run agentchaos audit examples/audits.py --output report.json
uv run pytest
```

The local CLI executes `SCENARIOS` from an explicitly selected, trusted Python file, prints status counts, and writes the complete audit report as deterministic JSON. Python audit files execute arbitrary code and must not be loaded from untrusted sources.

The model's first response requests `get_order_status(order_id="ORD-101")`. With no policy—or a policy targeting another tool—the registered Python function runs normally. A matching policy defaults to `stage="before"`, which short-circuits execution and returns an error `ToolMessage`. With `stage="after"`, the tool executes once and a successful `ToolMessage` is replaced with the configured error. An optional `on_call` schedules the fault for one matching attempt, while `None` injects on every matching call. Returned tool errors, commands, and raised exceptions are preserved.

The Milestone 8 reliability demonstration compares an unsafe V1 agent that fabricates a status after the primary service fails with a V2 agent that calls an independent backup tool. The same deterministic evaluator marks V1 `FAIL` and V2 `PASS`.

The Milestone 9B.2 experiment replaces the first successful `create_order` response with an error, causing one deterministic retry with the same idempotency key. A naive in-memory service persists two orders and evaluates `FAIL`; an atomic idempotent service reuses the first order and evaluates `PASS`.

The checkpoint-recovery experiment saves a LangGraph checkpoint before `create_order`, raises a controlled exception after the external commit, and resumes the same thread. The node re-executes with the saved idempotency key: the naive service duplicates the order, while the idempotent service reuses it.

The process-crash experiment uses durable SQLite databases and terminates a blocked worker after its order transaction commits. A fresh worker resumes the same LangGraph thread from its synced checkpoint, demonstrating real process termination rather than an in-process exception.

## Limitations

- Each middleware instance supports one exact tool-name policy.
- Attempt schedules are in-memory and scoped to one middleware instance; recreating middleware resets the counter.
- Post-execution injection cannot undo side effects produced by the tool before its response is replaced.
- The idempotency experiment uses process-local locks and memory; it does not model durability or coordination across processes.
- Its concurrency checks coordinate threads deterministically, but still model only in-process behavior.
- Checkpoint recovery simulates a node failure; it is not an actual process crash or a durability test.
- Process-crash recovery uses local SQLite files and process termination; it does not model machine loss, filesystem corruption, or distributed databases.
- A single scenario runs its baseline and faulted executions sequentially; separate async scenarios can run concurrently.
- ScenarioRunner isolates agents, collectors, traces, and copied inputs, but not mutable dependencies captured by an agent factory. Use separate factory closures or another per-scenario provider for isolated in-memory state.
- ScenarioEvaluator treats check exceptions as inconclusive unless another check has a known failure; it does not assign scores or infer correctness beyond configured checks.
- Trace events live only in their collector instance and are lost when it is discarded.
- The fake model covers one fixed tool-call path; it does not interpret arbitrary user prompts.
- The order data is hard-coded and the example has no persistence or external integrations.
- There are no retries, delays, timeouts, or generalized fault types.
