"""End-to-end deterministic duplicate-order reliability experiment."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from threading import Lock
from typing import Any, Self

from agentchaos import (
    ChaosMiddleware,
    EvaluationResult,
    FaultPolicy,
    ScenarioEvaluator,
    ScenarioResult,
    ScenarioRunner,
)
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool, StructuredTool


AGENT_INPUT = {"messages": [{"role": "user", "content": "Create my order."}]}
IDEMPOTENCY_KEY = "checkout-ORD-101"
CREATE_ORDER_FAULT = FaultPolicy(
    tool_name="create_order",
    error_message="Order service response was lost.",
    stage="after",
    on_call=1,
)


@dataclass(frozen=True, slots=True)
class Order:
    order_id: str
    idempotency_key: str
    item: str


class NaiveOrderService:
    """Persist a new order for every call, without atomic key reuse."""

    def __init__(self, before_create: Callable[[], None] | None = None) -> None:
        self._orders: list[Order] = []
        self._attempted_keys: list[str] = []
        self._lock = Lock()
        self._before_create = before_create

    def create_order(self, idempotency_key: str, item: str) -> str:
        with self._lock:
            self._attempted_keys.append(idempotency_key)
        if self._before_create is not None:
            self._before_create()
        with self._lock:
            order = Order(
                order_id=f"ORDER-{len(self._orders) + 1:03}",
                idempotency_key=idempotency_key,
                item=item,
            )
            self._orders.append(order)
            return order.order_id

    @property
    def orders(self) -> tuple[Order, ...]:
        with self._lock:
            return tuple(self._orders)

    @property
    def attempted_keys(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._attempted_keys)


class IdempotentOrderService:
    """Atomically reuse the order already stored for an idempotency key."""

    def __init__(self) -> None:
        self._orders: list[Order] = []
        self._orders_by_key: dict[str, Order] = {}
        self._attempted_keys: list[str] = []
        self._lock = Lock()

    def create_order(self, idempotency_key: str, item: str) -> str:
        with self._lock:
            self._attempted_keys.append(idempotency_key)
            if existing := self._orders_by_key.get(idempotency_key):
                return existing.order_id
            order = Order(
                order_id=f"ORDER-{len(self._orders) + 1:03}",
                idempotency_key=idempotency_key,
                item=item,
            )
            self._orders.append(order)
            self._orders_by_key[idempotency_key] = order
            return order.order_id

    @property
    def orders(self) -> tuple[Order, ...]:
        with self._lock:
            return tuple(self._orders)

    @property
    def attempted_keys(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._attempted_keys)


OrderService = NaiveOrderService | IdempotentOrderService


class DeterministicOrderModel(BaseChatModel):
    """Call create_order and retry once when its response is an error."""

    @property
    def _llm_type(self) -> str:
        return "deterministic-order-creator"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Self:
        return self

    def _next_message(self, messages: list[BaseMessage]) -> AIMessage:
        tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
        if not tool_messages:
            return self._tool_call("create-order-initial")

        latest = tool_messages[-1]
        if latest.status == "error" and len(tool_messages) == 1:
            return self._tool_call("create-order-retry")
        if latest.status == "error":
            return AIMessage(content=f"Order creation failed: {latest.content}")
        return AIMessage(content=f"Order confirmed: {latest.content}")

    @staticmethod
    def _tool_call(call_id: str) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "create_order",
                    "args": {
                        "idempotency_key": IDEMPOTENCY_KEY,
                        "item": "reliability-handbook",
                    },
                    "id": call_id,
                    "type": "tool_call",
                }
            ],
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next_message(messages))])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next_message(messages))])


def create_order_agent(middleware: ChaosMiddleware, service: OrderService):
    def create_order(idempotency_key: str, item: str) -> str:
        """Create an order using a caller-provided idempotency key."""
        return service.create_order(idempotency_key, item)

    async def acreate_order(idempotency_key: str, item: str) -> str:
        return create_order(idempotency_key, item)

    tool = StructuredTool.from_function(
        func=create_order,
        coroutine=acreate_order,
        name="create_order",
    )
    return create_agent(
        model=DeterministicOrderModel(),
        tools=[tool],
        middleware=[middleware],
    )


@dataclass(frozen=True, slots=True)
class ServiceExperiment:
    scenario: ScenarioResult
    evaluation: EvaluationResult
    baseline_store: OrderService
    faulted_store: OrderService


@dataclass(frozen=True, slots=True)
class IdempotencyDemonstration:
    fault_policy: FaultPolicy
    naive: ServiceExperiment
    idempotent: ServiceExperiment


def run_service_experiment(
    service_factory: Callable[[], OrderService],
) -> ServiceExperiment:
    stores: list[OrderService] = []

    def agent_factory(middleware: ChaosMiddleware):
        service = service_factory()
        stores.append(service)
        return create_order_agent(middleware, service)

    scenario = ScenarioRunner(agent_factory).run(AGENT_INPUT, CREATE_ORDER_FAULT)
    baseline_store, faulted_store = stores
    evaluation = ScenarioEvaluator(
        {"persists exactly one order": lambda _: len(faulted_store.orders) == 1}
    ).evaluate(scenario)
    return ServiceExperiment(
        scenario=scenario,
        evaluation=evaluation,
        baseline_store=baseline_store,
        faulted_store=faulted_store,
    )


def run_demo() -> IdempotencyDemonstration:
    return IdempotencyDemonstration(
        fault_policy=CREATE_ORDER_FAULT,
        naive=run_service_experiment(NaiveOrderService),
        idempotent=run_service_experiment(IdempotentOrderService),
    )


if __name__ == "__main__":
    demonstration = run_demo()
    print(f"Naive: {demonstration.naive.evaluation.status}")
    print(f"Idempotent: {demonstration.idempotent.evaluation.status}")
