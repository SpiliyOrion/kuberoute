"""Registry lookup and uniqueness tests."""

from typing import cast

import pytest
from kuberoute_gateway.backends import ChatBackend
from kuberoute_gateway.registry import DuplicateModelError, ModelRegistry, UnknownModelError


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
