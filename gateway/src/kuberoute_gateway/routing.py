"""Deterministic model selection, independent of HTTP and inference execution."""

from dataclasses import dataclass
from typing import Literal

from kuberoute_gateway.registry import ModelRegistry

AUTO_MODEL_ID = "auto"


class NoEligibleModelError(LookupError):
    """Raised when automatic routing has no configured inference candidate."""


@dataclass(frozen=True)
class RoutingDecision:
    model_id: str
    policy: Literal["explicit", "first_eligible", "performance_v1"]


def select_model(requested_model_id: str, registry: ModelRegistry) -> RoutingDecision:
    """Honor explicit IDs or choose the first explicitly eligible backend."""
    if requested_model_id != AUTO_MODEL_ID:
        registry.get(requested_model_id)
        return RoutingDecision(model_id=requested_model_id, policy="explicit")

    candidates = registry.auto_eligible_model_ids()
    if not candidates:
        raise NoEligibleModelError("No eligible inference model is configured")
    return RoutingDecision(model_id=candidates[0], policy="first_eligible")
