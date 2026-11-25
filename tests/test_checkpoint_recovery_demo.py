from examples.checkpoint_recovery_demo import run_demo
from examples.idempotency_demo import IDEMPOTENCY_KEY


def test_checkpoint_resume_reexecutes_naive_node_and_duplicates_order() -> None:
    naive, _ = run_demo()

    assert naive.thread_id == "naive-recovery"
    assert naive.node_keys == (IDEMPOTENCY_KEY, IDEMPOTENCY_KEY)
    assert naive.pending_nodes == ("create_order",)
    assert naive.saved_state["idempotency_key"] == IDEMPOTENCY_KEY
    assert naive.resumed_state["idempotency_key"] == IDEMPOTENCY_KEY
    assert naive.resumed_state["order_id"] == "ORDER-002"
    assert [order.order_id for order in naive.service.orders] == [
        "ORDER-001",
        "ORDER-002",
    ]
    assert naive.service.attempted_keys == (IDEMPOTENCY_KEY, IDEMPOTENCY_KEY)


def test_checkpoint_resume_reexecutes_idempotent_node_without_duplicate() -> None:
    _, idempotent = run_demo()

    assert idempotent.thread_id == "idempotent-recovery"
    assert idempotent.node_keys == (IDEMPOTENCY_KEY, IDEMPOTENCY_KEY)
    assert idempotent.pending_nodes == ("create_order",)
    assert idempotent.saved_state["idempotency_key"] == IDEMPOTENCY_KEY
    assert idempotent.resumed_state["idempotency_key"] == IDEMPOTENCY_KEY
    assert idempotent.resumed_state["order_id"] == "ORDER-001"
    assert [order.order_id for order in idempotent.service.orders] == ["ORDER-001"]
    assert idempotent.service.attempted_keys == (IDEMPOTENCY_KEY, IDEMPOTENCY_KEY)


def test_recovery_runs_use_independent_services() -> None:
    first_naive, first_idempotent = run_demo()
    second_naive, second_idempotent = run_demo()

    assert first_naive.service is not second_naive.service
    assert first_idempotent.service is not second_idempotent.service
    assert len(second_naive.service.orders) == 2
    assert len(second_idempotent.service.orders) == 1
