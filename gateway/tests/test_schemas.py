"""Validation tests for the supported chat completion subset."""

import pytest
from kuberoute_gateway.schemas import (
    MAX_CHAT_MESSAGE_CHARS,
    MAX_CHAT_MESSAGES,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from pydantic import ValidationError


def test_request_accepts_supported_messages_and_options() -> None:
    request = ChatCompletionRequest.model_validate(
        {
            "model": "local-small",
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
            "temperature": 0.5,
            "seed": 42,
            "max_tokens": 32,
            "stream": False,
        }
    )
    assert request.stream is False
    assert request.messages[1].content == "Hello"
    assert request.seed == 42
    assert request.max_tokens == 32


def test_request_omits_optional_generation_controls_by_default() -> None:
    request = ChatCompletionRequest.model_validate(
        {"model": "local-small", "messages": [{"role": "user", "content": "Hello"}]}
    )
    assert request.seed is None
    assert request.max_tokens is None
    assert request.reasoning_effort is None


def test_request_accepts_opt_in_no_thinking() -> None:
    request = ChatCompletionRequest.model_validate(
        {
            "model": "remote-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "reasoning_effort": "none",
        }
    )
    assert request.reasoning_effort == "none"


@pytest.mark.parametrize(
    "change",
    [
        {"model": ""},
        {"messages": []},
        {"messages": [{"role": "tool", "content": "Hello"}]},
        {"messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]},
        {"temperature": -0.1},
        {"temperature": 2.1},
        {"seed": -1},
        {"seed": True},
        {"max_tokens": 0},
        {"max_tokens": 4097},
        {"max_tokens": 3.5},
        {"reasoning_effort": "low"},
        {"stream": True},
        {"tools": []},
    ],
)
def test_request_rejects_unsupported_or_invalid_input(change: dict[str, object]) -> None:
    payload: dict[str, object] = {
        "model": "local-small",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    payload.update(change)
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(payload)


def test_chat_request_size_limits_preserve_valid_boundaries() -> None:
    request = ChatCompletionRequest.model_validate(
        {
            "model": "m" * 128,
            "messages": [{"role": "user", "content": "x"} for _ in range(MAX_CHAT_MESSAGES)],
        }
    )
    assert len(request.messages) == MAX_CHAT_MESSAGES
    single = ChatCompletionRequest.model_validate(
        {
            "model": "mock-small",
            "messages": [{"role": "user", "content": "x" * MAX_CHAT_MESSAGE_CHARS}],
        }
    )
    assert len(single.messages[0].content) == MAX_CHAT_MESSAGE_CHARS


@pytest.mark.parametrize(
    "change",
    [
        {"model": "m" * 129},
        {"messages": [{"role": "user", "content": "x" * (MAX_CHAT_MESSAGE_CHARS + 1)}]},
        {"messages": [{"role": "user", "content": "x"} for _ in range(MAX_CHAT_MESSAGES + 1)]},
        {"messages": [{"role": "user", "content": "x" * MAX_CHAT_MESSAGE_CHARS} for _ in range(5)]},
    ],
)
def test_chat_request_rejects_oversized_fields(change: dict[str, object]) -> None:
    payload: dict[str, object] = {
        "model": "mock-small",
        "messages": [{"role": "user", "content": "hi"}],
    }
    payload.update(change)
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(payload)


def test_response_has_openai_style_shape() -> None:
    response = ChatCompletionResponse.model_validate(
        {
            "id": "chatcmpl-1",
            "created": 1,
            "model": "local-small",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}}],
        }
    )
    assert response.model_dump()["object"] == "chat.completion"
    assert response.choices[0].finish_reason == "stop"


def test_response_rejects_unsupported_finish_reason() -> None:
    with pytest.raises(ValidationError):
        ChatCompletionResponse.model_validate(
            {
                "id": "chatcmpl-1",
                "created": 1,
                "model": "local-small",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hi"},
                        "finish_reason": "tool_calls",
                    }
                ],
            }
        )


def test_response_accepts_length_finish_reason() -> None:
    response = ChatCompletionResponse.model_validate(
        {
            "id": "chatcmpl-2",
            "created": 2,
            "model": "remote-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Partial"},
                    "finish_reason": "length",
                }
            ],
        }
    )
    assert response.choices[0].finish_reason == "length"
