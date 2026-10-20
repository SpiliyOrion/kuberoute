"""Deterministic routing decisions without HTTP or real inference."""

from typing import cast

import pytest
from kuberoute_gateway.backends import ChatBackend
from kuberoute_gateway.registry import ModelRegistry, UnknownModelError
from kuberoute_gateway.routing import NoEligibleModelError, RoutingDecision, select_model


def test_explicit_model_is_used_even_when_an_auto_candidate_exists() -> None:
    registry = ModelRegistry()
    registry.register("mock-small", cast(ChatBackend, object()))
    registry.register("real-one", cast(ChatBackend, object()), auto_eligible=True)

    assert select_model("mock-small", registry) == RoutingDecision("mock-small", "explicit")
    with pytest.raises(UnknownModelError, match="missing"):
        select_model("missing", registry)


def test_auto_never_substitutes_an_ineligible_mock() -> None:
    registry = ModelRegistry()
    registry.register("mock-small", cast(ChatBackend, object()))

    with pytest.raises(NoEligibleModelError, match="No eligible inference model"):
        select_model("auto", registry)


def test_auto_selects_first_eligible_model_in_registration_order() -> None:
    registry = ModelRegistry()
    registry.register("mock-small", cast(ChatBackend, object()))
    registry.register("real-one", cast(ChatBackend, object()), auto_eligible=True)
    assert select_model("auto", registry) == RoutingDecision("real-one", "first_eligible")

    registry.register("real-two", cast(ChatBackend, object()), auto_eligible=True)
    assert select_model("auto", registry) == RoutingDecision("real-one", "first_eligible")
