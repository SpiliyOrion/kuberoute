"""Per-process, opt-in feedback state for the pure performance policy."""

import asyncio
import os
from typing import Literal

from kuberoute_gateway.performance_policy import (
    InferenceObservation,
    PerformancePolicyConfig,
    PerformanceSnapshot,
    RollingPerformanceWindow,
    select_performance_model,
)
from kuberoute_gateway.routing import RoutingDecision

AutoRoutingPolicy = Literal["first_eligible", "performance_v1"]


def auto_routing_policy_from_env() -> AutoRoutingPolicy:
    """Read an exact, bounded policy identifier without echoing invalid values."""
    value = os.getenv("KUBEROUTE_AUTO_ROUTING_POLICY", "first_eligible")
    if value not in ("first_eligible", "performance_v1"):
        raise ValueError("KUBEROUTE_AUTO_ROUTING_POLICY must be first_eligible or performance_v1")
    return value


class AdaptiveRoutingState:
    """Coordinate decisions and completed-call feedback within one app lifespan."""

    def __init__(
        self,
        eligible_model_ids: tuple[str, ...],
        config: PerformancePolicyConfig | None = None,
    ) -> None:
        self._config = config or PerformancePolicyConfig()
        self._model_ids = eligible_model_ids
        self._window = RollingPerformanceWindow(
            eligible_model_ids, window_size=self._config.window_size
        )
        self._next_decision_index = 0
        self._lock = asyncio.Lock()

    async def select(self) -> RoutingDecision:
        async with self._lock:
            decision = select_performance_model(
                self._model_ids,
                self._window.snapshots(),
                self._config,
                decision_index=self._next_decision_index,
            )
            self._next_decision_index += 1
            return decision

    async def record(self, model_id: str, observation: InferenceObservation) -> None:
        async with self._lock:
            self._window.record(model_id, observation)

    async def inspect(self) -> tuple[int, dict[str, PerformanceSnapshot]]:
        """Return an atomic copy for tests and local diagnostics; no HTTP endpoint exposes it."""
        async with self._lock:
            return self._next_decision_index, self._window.snapshots()
