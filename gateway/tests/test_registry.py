"""Registry lookup and uniqueness tests."""

from typing import cast

import pytest
from kuberoute_gateway.backends import ChatBackend
from kuberoute_gateway.registry import (
    DuplicateModelError,
    ModelRegistry,
    ReservedModelError,
    UnknownModelError,
)


def test_registry_returns_registered_backend() -> None:
    registry = ModelRegistry()
    backend = cast(ChatBackend, object())
    registry.register("local-small", backend)
    assert registry.get("local-small") is backend


def test_registry_rejects_duplicate_model_ids() -> None:
    registry = ModelRegistry()
    registry.register("local-small", cast(ChatBackend, object()))
    with pytest.raises(DuplicateModelError, match="local-small"):
        registry.register("local-small", cast(ChatBackend, object()))


def test_registry_reports_unknown_model_explicitly() -> None:
    registry = ModelRegistry()
    with pytest.raises(UnknownModelError, match="missing"):
        registry.get("missing")


def test_registry_tracks_only_explicitly_auto_eligible_models() -> None:
    registry = ModelRegistry()
    registry.register("mock-small", cast(ChatBackend, object()))
    registry.register("real-one", cast(ChatBackend, object()), auto_eligible=True)
    registry.register("real-two", cast(ChatBackend, object()), auto_eligible=True)
    assert registry.model_ids() == ("mock-small", "real-one", "real-two")
    assert registry.auto_eligible_model_ids() == ("real-one", "real-two")


def test_registry_rejects_reserved_auto_identifier() -> None:
    registry = ModelRegistry()
    with pytest.raises(ReservedModelError, match="auto"):
        registry.register("auto", cast(ChatBackend, object()), auto_eligible=True)
    assert registry.model_ids() == ()
