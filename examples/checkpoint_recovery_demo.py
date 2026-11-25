"""Deterministic checkpoint recovery after an order commit."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

try:
    from examples.idempotency_demo import (
        IDEMPOTENCY_KEY,
        IdempotentOrderService,
        NaiveOrderService,
        OrderService,
    )
except ModuleNotFoundError:  # Supports `python examples/checkpoint_recovery_demo.py`.
    from idempotency_demo import (
        IDEMPOTENCY_KEY,
        IdempotentOrderService,
        NaiveOrderService,
        OrderService,
    )


class OrderState(TypedDict):
    idempotency_key: str
    item: str
    order_id: str


class SimulatedNodeFailure(RuntimeError):
    """Raised after the external commit but before the graph node completes."""


@dataclass(slots=True)
class FailAfterFirstCommit:
    raised: bool = False

    def raise_once(self) -> None:
        if not self.raised:
            self.raised = True
            raise SimulatedNodeFailure("simulated failure after order commit")


@dataclass(frozen=True, slots=True)
class RecoveryExperiment:
    service: OrderService
    thread_id: str
    node_keys: tuple[str, ...]
    pending_nodes: tuple[str, ...]
    saved_state: OrderState
    resumed_state: OrderState


def run_recovery_experiment(
    service_factory: Callable[[], OrderService],
    *,
    thread_id: str,
) -> RecoveryExperiment:
    """Commit once, fail the node, then resume its saved thread."""
    service = service_factory()
    failure = FailAfterFirstCommit()
    node_keys: list[str] = []

    def create_order(state: OrderState) -> dict[str, str]:
        node_keys.append(state["idempotency_key"])
        order_id = service.create_order(state["idempotency_key"], state["item"])
        failure.raise_once()
        return {"order_id": order_id}

    graph_builder = StateGraph(OrderState)
    graph_builder.add_node("create_order", create_order)
    graph_builder.add_edge(START, "create_order")
    graph_builder.add_edge("create_order", END)
    graph = graph_builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": thread_id}}
    initial_state: OrderState = {
        "idempotency_key": IDEMPOTENCY_KEY,
        "item": "reliability-handbook",
    }

    try:
        graph.invoke(initial_state, config)
    except SimulatedNodeFailure:
        pass
    else:
        raise AssertionError("the controlled failure did not run")

    checkpoint = graph.get_state(config)
    saved_state: OrderState = dict(checkpoint.values)  # type: ignore[assignment]
    resumed_state: OrderState = graph.invoke(None, config)
    return RecoveryExperiment(
        service=service,
        thread_id=thread_id,
        node_keys=tuple(node_keys),
        pending_nodes=checkpoint.next,
        saved_state=saved_state,
        resumed_state=resumed_state,
    )


def run_demo() -> tuple[RecoveryExperiment, RecoveryExperiment]:
    return (
        run_recovery_experiment(NaiveOrderService, thread_id="naive-recovery"),
        run_recovery_experiment(
            IdempotentOrderService, thread_id="idempotent-recovery"
        ),
    )


if __name__ == "__main__":
    naive, idempotent = run_demo()
    print(f"Naive orders: {len(naive.service.orders)}")
    print(f"Idempotent orders: {len(idempotent.service.orders)}")
