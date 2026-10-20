"""HTTP integration tests for the simulated chat gateway."""

import kuberoute_gateway.main as gateway_main
import pytest
from fastapi.testclient import TestClient
from kuberoute_gateway.backends import ModelOutput, NormalizedChatRequest
from kuberoute_gateway.main import app, registry
from kuberoute_gateway.mock_backends import MockSmallBackend
from kuberoute_gateway.registry import ModelRegistry, ReservedModelError
from kuberoute_gateway.remote_backend import BackendConnectionError
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


def test_gateway_passes_generation_controls_and_returns_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[NormalizedChatRequest] = []

    async def complete(request: NormalizedChatRequest) -> ModelOutput:
        observed.append(request)
        return ModelOutput(content="Four", prompt_tokens=6, output_tokens=2)

    monkeypatch.setattr(registry.get("mock-small"), "complete", complete)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "mock-small",
                "messages": [{"role": "user", "content": "2 + 2?"}],
                "temperature": 0,
                "seed": 42,
                "max_tokens": 32,
                "reasoning_effort": "none",
            },
        )
    assert response.status_code == 200
    assert observed[0].temperature == 0
    assert observed[0].seed == 42
    assert observed[0].max_tokens == 32
    assert observed[0].reasoning_effort == "none"
    assert response.json()["usage"] == {
        "prompt_tokens": 6,
        "completion_tokens": 2,
        "total_tokens": 8,
    }


def test_auto_returns_503_when_only_mocks_are_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    models = ModelRegistry()
    models.register("mock-small", MockSmallBackend())
    monkeypatch.setattr(gateway_main, "registry", models)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "auto", "messages": [{"role": "user", "content": "Hello"}]},
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "No eligible inference model is configured"}


def test_auto_uses_real_candidate_and_reports_selected_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[NormalizedChatRequest] = []

    class RecordingBackend:
        async def complete(self, request: NormalizedChatRequest) -> ModelOutput:
            observed.append(request)
            return ModelOutput(content="real output", finish_reason="length")

    models = ModelRegistry()
    models.register("mock-small", MockSmallBackend())
    models.register("real-one", RecordingBackend(), auto_eligible=True)
    monkeypatch.setattr(gateway_main, "registry", models)

    with TestClient(app) as client:
        listed = client.get("/v1/models")
        auto = client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0,
                "seed": 42,
                "max_tokens": 32,
            },
        )
        explicit = client.post(
            "/v1/chat/completions",
            json={"model": "mock-small", "messages": [{"role": "user", "content": "Hello"}]},
        )

    assert [item["id"] for item in listed.json()["data"]] == ["mock-small", "real-one"]
    assert auto.status_code == 200
    assert auto.json()["model"] == "real-one"
    assert auto.json()["choices"][0]["message"]["content"] == "real output"
    assert auto.json()["choices"][0]["finish_reason"] == "length"
    assert observed[0].model == "real-one"
    assert observed[0].temperature == 0
    assert observed[0].seed == 42
    assert observed[0].max_tokens == 32
    assert explicit.status_code == 200
    assert explicit.json()["model"] == "mock-small"
    assert explicit.json()["choices"][0]["message"]["content"] == "mock-small: Hello"


def test_auto_preserves_upstream_error_without_mock_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingBackend:
        async def complete(self, request: NormalizedChatRequest) -> ModelOutput:
            raise BackendConnectionError("private upstream detail")

    models = ModelRegistry()
    models.register("mock-small", MockSmallBackend())
    models.register("real-one", FailingBackend(), auto_eligible=True)
    monkeypatch.setattr(gateway_main, "registry", models)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "auto", "messages": [{"role": "user", "content": "Hello"}]},
        )

    assert response.status_code == 502
    assert response.json() == {"detail": "Upstream model failed"}
    assert "private upstream detail" not in response.text


def test_remote_model_cannot_use_reserved_auto_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KUBEROUTE_REMOTE_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("KUBEROUTE_REMOTE_MODEL_ID", "auto")
    with pytest.raises(ReservedModelError, match="auto"):
        gateway_main.create_registry()


def test_configured_remote_model_is_auto_eligible(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KUBEROUTE_REMOTE_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("KUBEROUTE_REMOTE_MODEL_ID", "real-one")
    models = gateway_main.create_registry()
    assert models.model_ids() == ("mock-small", "mock-large", "real-one")
    assert models.auto_eligible_model_ids() == ("real-one",)
