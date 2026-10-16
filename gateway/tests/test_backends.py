"""Type contract tests without an inference implementation."""

from dataclasses import FrozenInstanceError
from typing import get_type_hints

import pytest
from kuberoute_gateway.backends import (
    ChatBackend,
    ChatInputMessage,
    ModelOutput,
    NormalizedChatRequest,
)


def test_backend_contract_uses_transport_independent_types() -> None:
    hints = get_type_hints(ChatBackend.complete)
    assert hints["request"] is NormalizedChatRequest
    assert hints["return"] is ModelOutput


def test_normalized_request_and_output_are_typed_immutable_values() -> None:
    request = NormalizedChatRequest(
        model="local-small", messages=(ChatInputMessage(role="user", content="Hi"),)
    )
    assert request.temperature is None
    assert ModelOutput(content="Hello").content == "Hello"
    with pytest.raises(FrozenInstanceError):
        request.model = "other"
