from examples.idempotency_demo import IDEMPOTENCY_KEY
from examples.process_crash_recovery_demo import run_demo


def test_real_process_crash_recovery_distinguishes_order_services() -> None:
    naive, idempotent = run_demo()

    for experiment in (naive, idempotent):
        assert experiment.crash_worker_terminated is True
        assert experiment.order_count_before_crash == 1
        assert experiment.checkpoint_next_before_crash == ("create_order",)
        assert experiment.checkpoint_key_before_crash == IDEMPOTENCY_KEY
        assert experiment.node_execution_count == 2
        assert experiment.attempted_keys == (IDEMPOTENCY_KEY, IDEMPOTENCY_KEY)
        assert experiment.resumed_state["idempotency_key"] == IDEMPOTENCY_KEY

    assert naive.order_ids == ("ORDER-001", "ORDER-002")
    assert naive.resumed_state["order_id"] == "ORDER-002"
    assert idempotent.order_ids == ("ORDER-001",)
    assert idempotent.resumed_state["order_id"] == "ORDER-001"
