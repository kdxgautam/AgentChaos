from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from examples.idempotency_demo import (
    IDEMPOTENCY_KEY,
    IdempotentOrderService,
    NaiveOrderService,
    run_demo,
)


def events(execution) -> list[tuple[str, str]]:
    return [(event.event, event.call_id) for event in execution.trace_events]


def concurrently_create(service, keys: tuple[str, str]) -> list[str]:
    start = Barrier(3)

    def create(key: str) -> str:
        start.wait()
        return service.create_order(key, "reliability-handbook")

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(create, key) for key in keys]
        start.wait()
        return [future.result() for future in futures]


def test_persisted_counts_expose_duplicate_order_risk() -> None:
    demonstration = run_demo()

    assert len(demonstration.naive.baseline_store.orders) == 1
    assert len(demonstration.naive.faulted_store.orders) == 2
    assert len(demonstration.idempotent.baseline_store.orders) == 1
    assert len(demonstration.idempotent.faulted_store.orders) == 1
    assert [
        order.order_id for order in demonstration.naive.faulted_store.orders
    ] == ["ORDER-001", "ORDER-002"]
    assert [
        order.order_id for order in demonstration.idempotent.faulted_store.orders
    ] == ["ORDER-001"]
    assert demonstration.naive.scenario.faulted.output["messages"][-1].content == (
        "Order confirmed: ORDER-002"
    )
    assert (
        demonstration.idempotent.scenario.faulted.output["messages"][-1].content
        == "Order confirmed: ORDER-001"
    )

    assert demonstration.naive.evaluation.status == "FAIL"
    assert demonstration.idempotent.evaluation.status == "PASS"
    assert demonstration.naive.evaluation.checks[0].status == "FAIL"
    assert demonstration.idempotent.evaluation.checks[0].status == "PASS"


def test_faulted_runs_retry_once_with_the_same_key_and_accurate_traces() -> None:
    demonstration = run_demo()
    expected_events = [
        ("start", "create-order-initial"),
        ("success", "create-order-initial"),
        ("response_replaced", "create-order-initial"),
        ("start", "create-order-retry"),
        ("success", "create-order-retry"),
    ]

    for experiment in (demonstration.naive, demonstration.idempotent):
        assert experiment.scenario.faulted.fault_triggered is True
        assert experiment.faulted_store.attempted_keys == (
            IDEMPOTENCY_KEY,
            IDEMPOTENCY_KEY,
        )
        assert events(experiment.scenario.faulted) == expected_events
        assert experiment.scenario.faulted.output["messages"][-1].content.startswith(
            "Order confirmed:"
        )


def test_baseline_and_faulted_stores_are_isolated() -> None:
    demonstration = run_demo()

    for experiment in (demonstration.naive, demonstration.idempotent):
        assert experiment.baseline_store is not experiment.faulted_store
        assert experiment.baseline_store.attempted_keys == (IDEMPOTENCY_KEY,)
        assert events(experiment.scenario.baseline) == [
            ("start", "create-order-initial"),
            ("success", "create-order-initial"),
        ]

    assert demonstration.naive.baseline_store is not (
        demonstration.idempotent.baseline_store
    )
    assert demonstration.naive.faulted_store is not (
        demonstration.idempotent.faulted_store
    )


def test_fault_policy_targets_first_successful_response() -> None:
    policy = run_demo().fault_policy

    assert policy.tool_name == "create_order"
    assert policy.stage == "after"
    assert policy.on_call == 1


def test_naive_service_can_persist_duplicate_concurrent_orders() -> None:
    ready_to_create = Barrier(2)
    service = NaiveOrderService(before_create=ready_to_create.wait)

    order_ids = concurrently_create(service, (IDEMPOTENCY_KEY, IDEMPOTENCY_KEY))

    assert sorted(order_ids) == ["ORDER-001", "ORDER-002"]
    assert len(service.orders) == 2
    assert [order.idempotency_key for order in service.orders] == [
        IDEMPOTENCY_KEY,
        IDEMPOTENCY_KEY,
    ]


def test_idempotent_service_concurrently_reuses_the_same_order() -> None:
    service = IdempotentOrderService()

    order_ids = concurrently_create(service, (IDEMPOTENCY_KEY, IDEMPOTENCY_KEY))

    assert order_ids == ["ORDER-001", "ORDER-001"]
    assert len(service.orders) == 1
    assert service.orders[0].idempotency_key == IDEMPOTENCY_KEY


@pytest.mark.parametrize("service_type", [NaiveOrderService, IdempotentOrderService])
def test_different_idempotency_keys_create_independent_orders(service_type) -> None:
    service = service_type()

    order_ids = concurrently_create(service, ("checkout-A", "checkout-B"))

    assert sorted(order_ids) == ["ORDER-001", "ORDER-002"]
    assert {order.idempotency_key for order in service.orders} == {
        "checkout-A",
        "checkout-B",
    }
