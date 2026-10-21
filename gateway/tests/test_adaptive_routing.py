"""Opt-in adaptive routing through the HTTP gateway without external inference."""

import asyncio

import httpx
import kuberoute_gateway.adaptive_routing as adaptive_routing
import kuberoute_gateway.main as main
import pytest
from fastapi.testclient import TestClient
from kuberoute_gateway.backends import ModelOutput, NormalizedChatRequest
from kuberoute_gateway.metrics import METRICS_REGISTRY
from kuberoute_gateway.mock_backends import MockSmallBackend
from kuberoute_gateway.performance_policy import InferenceObservation
from kuberoute_gateway.registry import ModelRegistry
from kuberoute_gateway.remote_backend import (
    BackendConnectionError,
    BackendHTTPError,
    BackendTimeoutError,
    OpenAICompatibleBackend,
)


class StubRemote(OpenAICompatibleBackend):
    def __init__(self, model_id: str, failure: Exception | None = None) -> None:
        super().__init__("http://models.example", model_id)
        self.failure = failure
        self.calls: list[NormalizedChatRequest] = []
        self.starts = 0
        self.closes = 0

    async def start(self) -> None:
        self.starts += 1

    async def close(self) -> None:
        self.closes += 1

    async def complete(self, request: NormalizedChatRequest) -> ModelOutput:
        self.calls.append(request)
        if self.failure is not None:
            raise self.failure
        return ModelOutput(content="synthetic output marker")


def models_with(*backends: StubRemote, mock_alias: bool = False) -> ModelRegistry:
    models = ModelRegistry()
    models.register("mock-small", MockSmallBackend())
    if mock_alias:
        models.register("custom-named-mock", MockSmallBackend(), auto_eligible=True)
    for backend in backends:
        models.register(backend._model_id, backend, auto_eligible=True)
    return models


def post(client: TestClient | httpx.AsyncClient, model: str, prompt: str = "Hello") -> object:
    return client.post(
        "/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": prompt}]},
    )


def metric(model_id: str, policy: str, outcome: str) -> float:
    labels = {
        "model": model_id,
        "backend_kind": "remote",
        "policy": policy,
        "outcome": outcome,
    }
    return METRICS_REGISTRY.get_sample_value("kuberoute_inference_requests_total", labels) or 0.0


def duration_count(model_id: str, policy: str, outcome: str) -> float:
    labels = {
        "model": model_id,
        "backend_kind": "remote",
        "policy": policy,
        "outcome": outcome,
    }
    return (
        METRICS_REGISTRY.get_sample_value("kuberoute_inference_duration_seconds_count", labels)
        or 0.0
    )


def inspect(client: TestClient) -> tuple[int, dict[str, object]]:
    assert client.portal is not None
    return client.portal.call(main.app.state.adaptive_routing.inspect)


def test_default_policy_preserves_first_eligible_and_explicit_routing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KUBEROUTE_AUTO_ROUTING_POLICY", raising=False)
    first, second = StubRemote("real-a"), StubRemote("real-b")
    monkeypatch.setattr(main, "registry", models_with(first, second))
    with TestClient(main.app) as client:
        assert main.app.state.adaptive_routing is None
        assert post(client, "auto").json()["model"] == "real-a"
        assert post(client, "auto").json()["model"] == "real-a"
        assert post(client, "real-b").json()["model"] == "real-b"
    assert (len(first.calls), len(second.calls)) == (2, 1)
    assert (first.starts, first.closes, second.starts, second.closes) == (1, 1, 1, 1)
    assert not hasattr(main.app.state, "adaptive_routing")


@pytest.mark.parametrize("invalid", ["", "unknown-private-policy", "performance_v1 "])
def test_invalid_policy_fails_at_startup_without_echoing_value(
    monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    monkeypatch.setenv("KUBEROUTE_AUTO_ROUTING_POLICY", invalid)
    with pytest.raises(ValueError, match="must be first_eligible or performance_v1") as error:
        with TestClient(main.app):
            pass
    if invalid:
        assert invalid not in str(error.value)
    assert not hasattr(main.app.state, "adaptive_routing")


def test_adaptive_routing_uses_real_candidates_and_records_selected_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KUBEROUTE_AUTO_ROUTING_POLICY", "performance_v1")
    first, second = StubRemote("real-a"), StubRemote("real-b")
    monkeypatch.setattr(main, "registry", models_with(first, second, mock_alias=True))
    before_a = metric("real-a", "performance_v1", "success")
    before_b = metric("real-b", "performance_v1", "success")
    before_duration_a = duration_count("real-a", "performance_v1", "success")
    before_duration_b = duration_count("real-b", "performance_v1", "success")
    prompt = "private prompt marker"
    with TestClient(main.app) as client:
        responses = [post(client, "auto", prompt) for _ in range(3)]
        count, snapshots = inspect(client)
        exposition = client.get("/metrics").text
    assert [response.status_code for response in responses] == [200, 200, 200]
    assert [response.json()["model"] for response in responses] == ["real-a", "real-b", "real-a"]
    assert all(call.model == "real-a" for call in first.calls)
    assert all(call.model == "real-b" for call in second.calls)
    assert count == 3
    assert snapshots["real-a"].successful_attempts == 2
    assert snapshots["real-b"].successful_attempts == 1
    assert "custom-named-mock" not in snapshots
    assert prompt not in repr(snapshots)
    assert "synthetic output marker" not in repr(snapshots)
    assert prompt not in exposition
    assert "synthetic output marker" not in exposition
    assert metric("real-a", "performance_v1", "success") == before_a + 2
    assert metric("real-b", "performance_v1", "success") == before_b + 1
    assert duration_count("real-a", "performance_v1", "success") == before_duration_a + 2
    assert duration_count("real-b", "performance_v1", "success") == before_duration_b + 1
    assert 'policy="performance_v1"' in exposition


def test_no_eligible_real_model_returns_503_even_with_eligible_mock_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KUBEROUTE_AUTO_ROUTING_POLICY", "performance_v1")
    monkeypatch.setattr(main, "registry", models_with(mock_alias=True))
    with TestClient(main.app) as client:
        auto = post(client, "auto")
        explicit = post(client, "mock-small")
        count, snapshots = inspect(client)
    assert auto.status_code == 503
    assert auto.json() == {"detail": "No eligible inference model is configured"}
    assert explicit.status_code == 200
    assert count == 0
    assert snapshots == {}


def test_one_remote_model_and_explicit_requests_do_not_change_adaptive_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KUBEROUTE_AUTO_ROUTING_POLICY", "performance_v1")
    remote = StubRemote("real-a")
    monkeypatch.setattr(main, "registry", models_with(remote))
    before_explicit = metric("real-a", "explicit", "success")
    with TestClient(main.app) as client:
        explicit = post(client, "real-a")
        mock = post(client, "mock-small")
        unknown = post(client, "missing")
        before_count, before_snapshots = inspect(client)
        automatic = post(client, "auto")
        after_count, after_snapshots = inspect(client)
    assert explicit.status_code == 200
    assert mock.status_code == 200
    assert unknown.status_code == 404
    assert before_count == 0
    assert before_snapshots["real-a"].completed_attempts == 0
    assert automatic.status_code == 200
    assert automatic.json()["model"] == "real-a"
    assert after_count == 1
    assert after_snapshots["real-a"].successful_attempts == 1
    assert metric("real-a", "explicit", "success") == before_explicit + 1


@pytest.mark.parametrize(
    ("failure", "expected_status", "outcome"),
    [
        (BackendTimeoutError("private timeout"), 504, "timeout"),
        (BackendConnectionError("private connection"), 502, "error"),
        (BackendHTTPError("private upstream"), 502, "error"),
        (RuntimeError("private unexpected"), 500, "error"),
    ],
)
def test_adaptive_failure_feedback_preserves_http_and_metric_outcomes(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
    expected_status: int,
    outcome: str,
) -> None:
    monkeypatch.setenv("KUBEROUTE_AUTO_ROUTING_POLICY", "performance_v1")
    remote = StubRemote("real-a", failure)
    monkeypatch.setattr(main, "registry", models_with(remote))
    before = metric("real-a", "performance_v1", outcome)
    before_duration = duration_count("real-a", "performance_v1", outcome)
    with TestClient(main.app) as client:
        response = post(client, "auto")
        count, snapshots = inspect(client)
        exposition = client.get("/metrics").text
    assert response.status_code == expected_status
    assert "private" not in response.text
    assert count == 1
    assert snapshots["real-a"].completed_attempts == 1
    assert snapshots["real-a"].failed_attempts == 1
    assert snapshots["real-a"].successful_latencies_seconds == ()
    assert metric("real-a", "performance_v1", outcome) == before + 1
    assert duration_count("real-a", "performance_v1", outcome) == before_duration + 1
    assert "private" not in exposition


def test_state_is_new_each_lifespan_and_removed_after_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KUBEROUTE_AUTO_ROUTING_POLICY", "performance_v1")
    remote = StubRemote("real-a")
    monkeypatch.setattr(main, "registry", models_with(remote))
    with TestClient(main.app) as client:
        first_state = main.app.state.adaptive_routing
        assert post(client, "auto").status_code == 200
        assert inspect(client)[0] == 1
    assert not hasattr(main.app.state, "adaptive_routing")
    with TestClient(main.app) as client:
        second_state = main.app.state.adaptive_routing
        assert second_state is not first_state
        assert inspect(client)[0] == 0
        assert inspect(client)[1]["real-a"].completed_attempts == 0
    assert (remote.starts, remote.closes) == (2, 2)
    assert not hasattr(main.app.state, "adaptive_routing")


def test_state_cleanup_after_backend_startup_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KUBEROUTE_AUTO_ROUTING_POLICY", "performance_v1")

    class BrokenRemote(StubRemote):
        async def start(self) -> None:
            raise RuntimeError("private startup detail")

    remote = BrokenRemote("real-a")
    monkeypatch.setattr(main, "registry", models_with(remote))
    with pytest.raises(RuntimeError, match="Remote backend initialization failed") as error:
        with TestClient(main.app):
            pass
    assert "private" not in str(error.value)
    assert remote.closes == 1
    assert not hasattr(main.app.state, "adaptive_routing")


def test_concurrent_requests_use_unique_indices_and_keep_all_observations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KUBEROUTE_AUTO_ROUTING_POLICY", "performance_v1")
    seen_indices: list[int] = []
    original = adaptive_routing.select_performance_model

    def traced(*args: object, decision_index: int, **kwargs: object) -> object:
        seen_indices.append(decision_index)
        return original(*args, decision_index=decision_index, **kwargs)

    monkeypatch.setattr(adaptive_routing, "select_performance_model", traced)

    async def run() -> None:
        release = asyncio.Event()
        all_started = asyncio.Event()

        class WaitingRemote(StubRemote):
            async def complete(self, request: NormalizedChatRequest) -> ModelOutput:
                self.calls.append(request)
                if len(self.calls) == 20:
                    all_started.set()
                await release.wait()
                return ModelOutput(content="synthetic output marker")

        remote = WaitingRemote("real-a")
        monkeypatch.setattr(main, "registry", models_with(remote))
        before = metric("real-a", "performance_v1", "success")
        async with main.lifespan(main.app):
            transport = httpx.ASGITransport(app=main.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                requests = [asyncio.create_task(post(client, "auto")) for _ in range(20)]
                await asyncio.wait_for(all_started.wait(), timeout=5)
                state = main.app.state.adaptive_routing
                count, snapshots = await state.inspect()
                assert count == 20
                assert snapshots["real-a"].completed_attempts == 0
                release.set()
                responses = await asyncio.gather(*requests)
                assert all(response.status_code == 200 for response in responses)
                count, snapshots = await state.inspect()
                assert count == 20
                assert snapshots["real-a"].successful_attempts == 20
        assert metric("real-a", "performance_v1", "success") == before + 20

    asyncio.run(run())
    assert seen_indices == list(range(20))


def test_concurrent_results_are_attributed_to_the_selected_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KUBEROUTE_AUTO_ROUTING_POLICY", "performance_v1")

    async def run() -> None:
        first_started, second_started = asyncio.Event(), asyncio.Event()
        release_first, release_second = asyncio.Event(), asyncio.Event()

        class WaitingRemote(StubRemote):
            def __init__(
                self, model_id: str, started: asyncio.Event, release: asyncio.Event
            ) -> None:
                super().__init__(model_id)
                self.started = started
                self.release = release

            async def complete(self, request: NormalizedChatRequest) -> ModelOutput:
                self.calls.append(request)
                self.started.set()
                await self.release.wait()
                return ModelOutput(content="synthetic output marker")

        first = WaitingRemote("real-a", first_started, release_first)
        second = WaitingRemote("real-b", second_started, release_second)
        monkeypatch.setattr(main, "registry", models_with(first, second))
        async with main.lifespan(main.app):
            transport = httpx.ASGITransport(app=main.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                first_task = asyncio.create_task(post(client, "auto"))
                await asyncio.wait_for(first_started.wait(), timeout=5)
                state = main.app.state.adaptive_routing
                await state.record("real-a", InferenceObservation("success", 0.01))
                second_task = asyncio.create_task(post(client, "auto"))
                await asyncio.wait_for(second_started.wait(), timeout=5)
                release_second.set()
                second_response = await second_task
                _, snapshots = await state.inspect()
                assert second_response.json()["model"] == "real-b"
                assert snapshots["real-a"].completed_attempts == 1
                assert snapshots["real-b"].successful_attempts == 1
                release_first.set()
                first_response = await first_task
                _, snapshots = await state.inspect()
                assert first_response.json()["model"] == "real-a"
                assert snapshots["real-a"].successful_attempts == 2
                assert snapshots["real-b"].successful_attempts == 1

    asyncio.run(run())


def test_cancelled_inference_adds_no_misleading_feedback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KUBEROUTE_AUTO_ROUTING_POLICY", "performance_v1")

    async def run() -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        class WaitingRemote(StubRemote):
            async def complete(self, request: NormalizedChatRequest) -> ModelOutput:
                started.set()
                await release.wait()
                return ModelOutput(content="synthetic output marker")

        remote = WaitingRemote("real-a")
        monkeypatch.setattr(main, "registry", models_with(remote))
        before = metric("real-a", "performance_v1", "error")
        async with main.lifespan(main.app):
            transport = httpx.ASGITransport(app=main.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                task = asyncio.create_task(post(client, "auto"))
                await asyncio.wait_for(started.wait(), timeout=5)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                count, snapshots = await main.app.state.adaptive_routing.inspect()
                assert count == 1
                assert snapshots["real-a"].completed_attempts == 0
        assert metric("real-a", "performance_v1", "error") == before

    asyncio.run(run())
