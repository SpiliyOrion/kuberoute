"""Inference observations count selected backend calls, separately from HTTP traffic."""

import httpx
import kuberoute_gateway.main as gateway_main
import pytest
from fastapi.testclient import TestClient
from kuberoute_gateway.backends import ModelOutput, NormalizedChatRequest
from kuberoute_gateway.metrics import METRICS_REGISTRY
from kuberoute_gateway.mock_backends import MockSmallBackend
from kuberoute_gateway.registry import ModelRegistry
from kuberoute_gateway.remote_backend import (
    BackendConnectionError,
    BackendHTTPError,
    BackendResponseError,
    BackendTimeoutError,
    OpenAICompatibleBackend,
)


def sample(name: str, labels: dict[str, str]) -> float:
    return METRICS_REGISTRY.get_sample_value(name, labels) or 0.0


def total_inference_calls() -> float:
    return sum(
        point.value
        for metric in METRICS_REGISTRY.collect()
        for point in metric.samples
        if point.name == "kuberoute_inference_requests_total"
    )


def completion(client: TestClient, model: str, prompt: str = "Hello") -> httpx.Response:
    return client.post(
        "/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": prompt}]},
    )


def test_explicit_mock_call_is_labeled_and_observed_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models = ModelRegistry()
    models.register("mock-small", MockSmallBackend())
    monkeypatch.setattr(gateway_main, "registry", models)
    labels = {
        "model": "mock-small",
        "backend_kind": "mock",
        "policy": "explicit",
        "outcome": "success",
    }
    before_calls = sample("kuberoute_inference_requests_total", labels)
    before_observations = sample("kuberoute_inference_duration_seconds_count", labels)
    before_duration = sample("kuberoute_inference_duration_seconds_sum", labels)

    with TestClient(gateway_main.app) as client:
        response = completion(client, "mock-small")
        exposition = client.get("/metrics").text

    assert response.status_code == 200
    assert sample("kuberoute_inference_requests_total", labels) == before_calls + 1
    assert sample("kuberoute_inference_duration_seconds_count", labels) == before_observations + 1
    assert sample("kuberoute_inference_duration_seconds_sum", labels) > before_duration
    assert "# TYPE kuberoute_inference_requests_total counter" in exposition
    assert "# TYPE kuberoute_inference_duration_seconds histogram" in exposition
    assert 'backend_kind="mock"' in exposition


def test_auto_call_uses_selected_model_and_never_records_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Backend:
        async def complete(self, request: NormalizedChatRequest) -> ModelOutput:
            assert request.model == "real-one"
            return ModelOutput(content="sensitive response marker")

    models = ModelRegistry()
    models.register("mock-small", MockSmallBackend())
    models.register("real-one", Backend(), auto_eligible=True)
    monkeypatch.setattr(gateway_main, "registry", models)
    labels = {
        "model": "real-one",
        "backend_kind": "other",
        "policy": "first_eligible",
        "outcome": "success",
    }
    before = sample("kuberoute_inference_requests_total", labels)
    prompt = "sensitive prompt marker"

    with TestClient(gateway_main.app) as client:
        response = completion(client, "auto", prompt)
        exposition = client.get("/metrics").text

    assert response.status_code == 200
    assert response.json()["model"] == "real-one"
    assert sample("kuberoute_inference_requests_total", labels) == before + 1
    assert sample("kuberoute_inference_duration_seconds_count", labels) >= 1
    assert 'model="auto"' not in exposition
    assert prompt not in exposition
    assert "sensitive response marker" not in exposition


def test_configured_remote_backend_has_separate_kind_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RemoteBackend(OpenAICompatibleBackend):
        async def complete(self, request: NormalizedChatRequest) -> ModelOutput:
            return ModelOutput(content="ok")

    models = ModelRegistry()
    models.register(
        "real-one", RemoteBackend("http://127.0.0.1:11434", "real-one"), auto_eligible=True
    )
    monkeypatch.setattr(gateway_main, "registry", models)
    labels = {
        "model": "real-one",
        "backend_kind": "remote",
        "policy": "first_eligible",
        "outcome": "success",
    }
    before = sample("kuberoute_inference_requests_total", labels)

    with TestClient(gateway_main.app) as client:
        response = completion(client, "auto")

    assert response.status_code == 200
    assert sample("kuberoute_inference_requests_total", labels) == before + 1


@pytest.mark.parametrize(
    ("failure", "outcome", "status", "detail"),
    [
        (BackendTimeoutError("private timeout"), "timeout", 504, "Upstream model timed out"),
        (BackendConnectionError("private connection"), "error", 502, "Upstream model failed"),
        (BackendHTTPError("private upstream status"), "error", 502, "Upstream model failed"),
        (BackendResponseError("private payload"), "error", 502, "Upstream model failed"),
        (RuntimeError("private unexpected"), "error", 500, "Model inference failed"),
    ],
)
def test_failed_auto_call_is_observed_once_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
    outcome: str,
    status: int,
    detail: str,
) -> None:
    class Backend:
        async def complete(self, request: NormalizedChatRequest) -> ModelOutput:
            raise failure

    models = ModelRegistry()
    models.register("mock-small", MockSmallBackend())
    models.register("real-one", Backend(), auto_eligible=True)
    monkeypatch.setattr(gateway_main, "registry", models)
    labels = {
        "model": "real-one",
        "backend_kind": "other",
        "policy": "first_eligible",
        "outcome": outcome,
    }
    before = sample("kuberoute_inference_requests_total", labels)
    before_observations = sample("kuberoute_inference_duration_seconds_count", labels)
    before_duration = sample("kuberoute_inference_duration_seconds_sum", labels)
    before_total = total_inference_calls()

    with TestClient(gateway_main.app) as client:
        response = completion(client, "auto")
        exposition = client.get("/metrics").text

    assert response.status_code == status
    assert response.json() == {"detail": detail}
    assert sample("kuberoute_inference_requests_total", labels) == before + 1
    assert sample("kuberoute_inference_duration_seconds_count", labels) == before_observations + 1
    assert sample("kuberoute_inference_duration_seconds_sum", labels) > before_duration
    assert total_inference_calls() == before_total + 1
    assert str(failure) not in exposition


@pytest.mark.parametrize(
    ("model", "status"),
    [("auto", 503), ("unknown-private-id", 404)],
)
def test_no_backend_execution_produces_no_inference_observation(
    monkeypatch: pytest.MonkeyPatch, model: str, status: int
) -> None:
    models = ModelRegistry()
    models.register("mock-small", MockSmallBackend())
    monkeypatch.setattr(gateway_main, "registry", models)
    before = total_inference_calls()

    with TestClient(gateway_main.app) as client:
        response = completion(client, model)

    assert response.status_code == status
    assert total_inference_calls() == before
