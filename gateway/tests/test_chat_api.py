"""HTTP integration tests for the simulated chat gateway."""

import pytest
from fastapi.testclient import TestClient
from kuberoute_gateway.backends import ModelOutput, NormalizedChatRequest
from kuberoute_gateway.main import app, registry
from kuberoute_gateway.schemas import ChatCompletionResponse


def test_models_lists_registered_ids() -> None:
    with TestClient(app) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    assert response.json() == {
        "object": "list",
        "data": [
            {"id": "mock-small", "object": "model"},
            {"id": "mock-large", "object": "model"},
        ],
    }


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("mock-small", "mock-small: Hello"),
        ("mock-large", "mock-large: received 2 messages; Hello"),
    ],
)
def test_chat_routes_to_selected_model(model: str, expected: str) -> None:
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "Be concise"},
                    {"role": "user", "content": "Hello"},
                ],
                "stream": False,
            },
        )

    assert response.status_code == 200
    completion = ChatCompletionResponse.model_validate(response.json())
    assert completion.id.startswith("chatcmpl-")
    assert completion.model == model
    assert completion.choices[0].message.content == expected
    assert completion.choices[0].finish_reason == "stop"


@pytest.mark.parametrize(
    "change",
    [
        {"stream": True},
        {"tools": []},
        {"messages": []},
        {"messages": [{"role": "tool", "content": "Hello"}]},
        {"temperature": 3},
    ],
)
def test_invalid_chat_requests_return_422(change: dict[str, object]) -> None:
    payload: dict[str, object] = {
        "model": "mock-small",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    payload.update(change)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 422


def test_unknown_model_returns_404() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "missing", "messages": [{"role": "user", "content": "Hello"}]},
        )
    assert response.status_code == 404
    assert response.json() == {"detail": "Unknown model: missing"}


def test_backend_failure_returns_safe_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fail(request: NormalizedChatRequest) -> ModelOutput:
        raise RuntimeError("secret internal failure")

    monkeypatch.setattr(registry.get("mock-small"), "complete", fail)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "mock-small", "messages": [{"role": "user", "content": "Hi"}]},
        )
    assert response.status_code == 500
    assert response.json() == {"detail": "Model inference failed"}
    assert "secret internal failure" not in response.text
