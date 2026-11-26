"""Real subprocess-crash recovery for an external SQLite order commit."""

from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, TypedDict
import multiprocessing
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

try:
    from examples.idempotency_demo import IDEMPOTENCY_KEY
except ModuleNotFoundError:  # Supports `python examples/process_crash_recovery_demo.py`.
    from idempotency_demo import IDEMPOTENCY_KEY


ServiceKind = Literal["naive", "idempotent"]
THREAD_ID = "process-crash-recovery"
ITEM = "reliability-handbook"


class OrderState(TypedDict):
    idempotency_key: str
    item: str
    order_id: str


class SQLiteOrderService:
    """Persistent order service used by one worker process."""

    def __init__(self, database_path: str, *, idempotent: bool) -> None:
        self.connection = sqlite3.connect(database_path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS orders (
                order_id INTEGER PRIMARY KEY AUTOINCREMENT,
                idempotency_key TEXT NOT NULL,
                item TEXT NOT NULL
            )
            """
        )
        if idempotent:
            self.connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS orders_idempotency_key "
                "ON orders (idempotency_key)"
            )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS node_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                idempotency_key TEXT NOT NULL
            )
            """
        )
        self.connection.commit()
        self.idempotent = idempotent

    def record_node_execution(self, thread_id: str, idempotency_key: str) -> None:
        self.connection.execute(
            "INSERT INTO node_executions (thread_id, idempotency_key) VALUES (?, ?)",
            (thread_id, idempotency_key),
        )
        self.connection.commit()

    def create_order(self, idempotency_key: str, item: str) -> str:
        if self.idempotent:
            self.connection.execute(
                "INSERT INTO orders (idempotency_key, item) VALUES (?, ?) "
                "ON CONFLICT(idempotency_key) DO NOTHING",
                (idempotency_key, item),
            )
            row = self.connection.execute(
                "SELECT order_id FROM orders WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            assert row is not None
            self.connection.commit()
            return f"ORDER-{row[0]:03}"

        cursor = self.connection.execute(
            "INSERT INTO orders (idempotency_key, item) VALUES (?, ?)",
            (idempotency_key, item),
        )
        self.connection.commit()
        return f"ORDER-{cursor.lastrowid:03}"

    def close(self) -> None:
        self.connection.close()


def _service(database_path: str, kind: ServiceKind) -> SQLiteOrderService:
    return SQLiteOrderService(database_path, idempotent=kind == "idempotent")


def _build_graph(
    service: SQLiteOrderService,
    *,
    after_commit: Callable[[str], None] | None = None,
):
    def create_order(state: OrderState) -> dict[str, str]:
        service.record_node_execution(THREAD_ID, state["idempotency_key"])
        order_id = service.create_order(state["idempotency_key"], state["item"])
        if after_commit is not None:
            after_commit(order_id)
        return {"order_id": order_id}

    builder = StateGraph(OrderState)
    builder.add_node("create_order", create_order)
    builder.add_edge(START, "create_order")
    builder.add_edge("create_order", END)
    return builder


def _worker(
    mode: Literal["crash", "resume"],
    kind: ServiceKind,
    order_database: str,
    checkpoint_database: str,
    signal: Connection,
    blocker: Any | None = None,
) -> None:
    service = _service(order_database, kind)
    config = {"configurable": {"thread_id": THREAD_ID}}
    try:
        with SqliteSaver.from_conn_string(checkpoint_database) as checkpointer:
            if mode == "crash":
                assert blocker is not None

                def after_commit(order_id: str) -> None:
                    signal.send(("committed", order_id))
                    blocker.wait()

                graph = _build_graph(service, after_commit=after_commit).compile(
                    checkpointer=checkpointer
                )
                graph.invoke(
                    {"idempotency_key": IDEMPOTENCY_KEY, "item": ITEM},
                    config,
                    durability="sync",
                )
            else:
                graph = _build_graph(service).compile(checkpointer=checkpointer)
                result = graph.invoke(None, config, durability="sync")
                signal.send(("resumed", result))
    except Exception as exception:
        signal.send(("error", type(exception).__name__, str(exception)))
    finally:
        service.close()
        signal.close()


@dataclass(frozen=True, slots=True)
class ProcessCrashExperiment:
    service_kind: ServiceKind
    crash_worker_terminated: bool
    order_count_before_crash: int
    checkpoint_next_before_crash: tuple[str, ...]
    checkpoint_key_before_crash: str
    order_ids: tuple[str, ...]
    attempted_keys: tuple[str, ...]
    node_execution_count: int
    resumed_state: OrderState


def _receive(connection: Connection, timeout: float) -> tuple[Any, ...]:
    if not connection.poll(timeout):
        raise TimeoutError("worker did not reach the expected coordination point")
    message = connection.recv()
    if message[0] == "error":
        raise RuntimeError(f"worker failed: {message[1]}: {message[2]}")
    return message


def _terminate(process: multiprocessing.Process, timeout: float) -> bool:
    process.terminate()
    process.join(timeout)
    if process.is_alive():
        process.kill()
        process.join(timeout)
    return not process.is_alive()


def _read_orders(database_path: str) -> tuple[tuple[str, ...], tuple[str, ...], int]:
    with sqlite3.connect(database_path) as connection:
        order_ids = tuple(
            f"ORDER-{row[0]:03}"
            for row in connection.execute("SELECT order_id FROM orders ORDER BY order_id")
        )
        attempted_keys = tuple(
            row[0]
            for row in connection.execute(
                "SELECT idempotency_key FROM node_executions ORDER BY id"
            )
        )
    return order_ids, attempted_keys, len(attempted_keys)


def _checkpoint_before_crash(
    checkpoint_path: str,
    order_database: str,
    service_kind: ServiceKind,
) -> tuple[tuple[str, ...], str]:
    config = {"configurable": {"thread_id": THREAD_ID}}
    service = _service(order_database, service_kind)
    try:
        with SqliteSaver.from_conn_string(checkpoint_path) as checkpointer:
            graph = _build_graph(service).compile(checkpointer=checkpointer)
            snapshot = graph.get_state(config)
    finally:
        service.close()
    return snapshot.next, snapshot.values["idempotency_key"]


def run_process_crash_experiment(
    service_kind: ServiceKind,
    *,
    timeout: float = 10,
) -> ProcessCrashExperiment:
    """Kill a worker after its order commit, then resume its saved thread."""
    context = multiprocessing.get_context("spawn")
    with TemporaryDirectory(prefix="agentchaos-crash-") as directory:
        root = Path(directory)
        order_database = str(root / "orders.sqlite")
        checkpoint_database = str(root / "checkpoints.sqlite")
        parent_signal, child_signal = context.Pipe(duplex=False)
        blocker = context.Event()
        crash_worker = context.Process(
            target=_worker,
            args=(
                "crash",
                service_kind,
                order_database,
                checkpoint_database,
                child_signal,
                blocker,
            ),
        )
        crash_worker.start()
        child_signal.close()
        try:
            message = _receive(parent_signal, timeout)
            if message[0] != "committed":
                raise AssertionError(f"unexpected worker signal: {message[0]}")
            order_ids_before, _, order_count_before = _read_orders(order_database)
            if len(order_ids_before) != 1:
                raise AssertionError("order was not committed before termination")
            checkpoint_next, checkpoint_key = _checkpoint_before_crash(
                checkpoint_database, order_database, service_kind
            )
            terminated = _terminate(crash_worker, timeout)
        finally:
            if crash_worker.is_alive():
                _terminate(crash_worker, timeout)
            parent_signal.close()

        parent_signal, child_signal = context.Pipe(duplex=False)
        resume_worker = context.Process(
            target=_worker,
            args=(
                "resume",
                service_kind,
                order_database,
                checkpoint_database,
                child_signal,
            ),
        )
        resume_worker.start()
        child_signal.close()
        try:
            message = _receive(parent_signal, timeout)
            if message[0] != "resumed":
                raise AssertionError(f"unexpected worker signal: {message[0]}")
            resume_worker.join(timeout)
            if resume_worker.is_alive():
                raise TimeoutError("recovery worker did not finish")
            order_ids, attempted_keys, node_execution_count = _read_orders(
                order_database
            )
        finally:
            if resume_worker.is_alive():
                _terminate(resume_worker, timeout)
            parent_signal.close()

    return ProcessCrashExperiment(
        service_kind=service_kind,
        crash_worker_terminated=terminated,
        order_count_before_crash=order_count_before,
        checkpoint_next_before_crash=checkpoint_next,
        checkpoint_key_before_crash=checkpoint_key,
        order_ids=order_ids,
        attempted_keys=attempted_keys,
        node_execution_count=node_execution_count,
        resumed_state=message[1],
    )


def run_demo() -> tuple[ProcessCrashExperiment, ProcessCrashExperiment]:
    return (
        run_process_crash_experiment("naive"),
        run_process_crash_experiment("idempotent"),
    )


if __name__ == "__main__":
    naive, idempotent = run_demo()
    print(f"Naive orders after recovery: {len(naive.order_ids)}")
    print(f"Idempotent orders after recovery: {len(idempotent.order_ids)}")
