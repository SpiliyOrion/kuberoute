"""Ordered remote configuration, gateway routing, and client lifecycle."""

import asyncio
import json

import httpx
import kuberoute_gateway.main as main
import kuberoute_gateway.remote_backend as remote_backend
import pytest
from fastapi.testclient import TestClient
from kuberoute_gateway.metrics import METRICS_REGISTRY
from kuberoute_gateway.registry import ModelRegistry
from kuberoute_gateway.remote_backend import (
    OpenAICompatibleBackend,
    remote_config_from_env,
    remote_configs_from_env,
)

LEGACY_ENVS = (
    "KUBEROUTE_REMOTE_BASE_URL",
    "KUBEROUTE_REMOTE_MODEL_ID",
    "KUBEROUTE_REMOTE_API_KEY",
    "KUBEROUTE_REMOTE_DEADLINE_SECONDS",
)


@pytest.fixture
def clean_remote_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (*LEGACY_ENVS, "KUBEROUTE_REMOTE_MODELS_JSON"):
        monkeypatch.delenv(name, raising=False)


def configure(monkeypatch: pytest.MonkeyPatch, entries: list[dict[str, object]]) -> None:
    monkeypatch.setenv("KUBEROUTE_REMOTE_MODELS_JSON", json.dumps(entries))


def entry(model_id: str, base_url: str = "https://models.example") -> dict[str, object]:
    return {"model_id": model_id, "base_url": base_url}


def success_response(model_id: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": f"output from {model_id}"},
                    "finish_reason": "stop",
                }
            ]
        },
    )


def inference_sample(model_id: str, policy: str, outcome: str) -> float:
    return (
        METRICS_REGISTRY.get_sample_value(
            "kuberoute_inference_requests_total",
            {"model": model_id, "backend_kind": "remote", "policy": policy, "outcome": outcome},
        )
        or 0.0
    )


def post(client: TestClient, model_id: str) -> httpx.Response:
    return client.post(
        "/v1/chat/completions",
        json={"model": model_id, "messages": [{"role": "user", "content": "Hello"}]},
    )


def test_legacy_config_remains_compatible(
    clean_remote_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert remote_configs_from_env() == ()
    monkeypatch.setenv("KUBEROUTE_REMOTE_BASE_URL", "https://models.example")
    monkeypatch.setenv("KUBEROUTE_REMOTE_MODEL_ID", "legacy-model")
    monkeypatch.setenv("KUBEROUTE_REMOTE_API_KEY", "fake-test-key")
    monkeypatch.setenv("KUBEROUTE_REMOTE_DEADLINE_SECONDS", "60")

    assert remote_configs_from_env() == (remote_config_from_env(),)
    assert main.create_registry().model_ids() == ("mock-small", "mock-large", "legacy-model")
    assert "fake-test-key" not in repr(remote_config_from_env())


def test_eight_remote_models_and_numeric_deadline_are_accepted(
    clean_remote_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries = [entry(f"real-{index}") for index in range(8)]
    entries[0]["deadline_seconds"] = 60
    configure(monkeypatch, entries)
    configs = remote_configs_from_env()
    assert len(configs) == 8
    assert configs[0].deadline_seconds == 60


def test_invalid_legacy_url_does_not_echo_credentials(
    clean_remote_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KUBEROUTE_REMOTE_BASE_URL", "http://user:private-password@host:bad")
    monkeypatch.setenv("KUBEROUTE_REMOTE_MODEL_ID", "legacy-model")
    with pytest.raises(ValueError, match="Remote model configuration is invalid") as error:
        main.create_registry()
    assert "private-password" not in str(error.value)


@pytest.mark.parametrize(
    "base_urls",
    [
        ["https://one.example", "https://one.example"],
        ["https://one.example", "https://two.example"],
        ["https://one.example", "https://two.example", "https://three.example"],
    ],
)
def test_multi_model_registration_routing_metrics_and_cleanup(
    clean_remote_env: None, monkeypatch: pytest.MonkeyPatch, base_urls: list[str]
) -> None:
    ids = [f"real-{index}" for index in range(len(base_urls))]
    configure(
        monkeypatch, [entry(model_id, url) for model_id, url in zip(ids, base_urls, strict=True)]
    )
    models = main.create_registry()
    assert models.model_ids() == ("mock-small", "mock-large", *ids)
    assert models.auto_eligible_model_ids() == tuple(ids)
    monkeypatch.setattr(main, "registry", models)

    calls: list[tuple[str, str]] = []
    created: list[httpx.AsyncClient] = []
    original_client = httpx.AsyncClient

    def respond(request: httpx.Request) -> httpx.Response:
        model_id = json.loads(request.content)["model"]
        calls.append((model_id, request.url.host))
        return success_response(model_id)

    def new_client() -> httpx.AsyncClient:
        client = original_client(transport=httpx.MockTransport(respond))
        created.append(client)
        return client

    monkeypatch.setattr(remote_backend.httpx, "AsyncClient", new_client)
    before = {model_id: inference_sample(model_id, "explicit", "success") for model_id in ids}
    before_auto = inference_sample(ids[0], "first_eligible", "success")

    with TestClient(main.app) as client:
        listing = client.get("/v1/models")
        assert [item["id"] for item in listing.json()["data"]] == [
            "mock-small",
            "mock-large",
            *ids,
        ]
        for model_id in ids:
            response = post(client, model_id)
            assert response.status_code == 200
            assert response.json()["model"] == model_id
            assert response.json()["choices"][0]["message"]["content"] == (
                f"output from {model_id}"
            )
        auto = post(client, "auto")
        assert auto.status_code == 200
        assert auto.json()["model"] == ids[0]
        assert all(not client.is_closed for client in created)

    assert len(created) == len(ids)
    assert all(client.is_closed for client in created)
    assert calls == [
        *((model_id, httpx.URL(url).host) for model_id, url in zip(ids, base_urls, strict=True)),
        (ids[0], httpx.URL(base_urls[0]).host),
    ]
    for model_id in ids:
        assert inference_sample(model_id, "explicit", "success") == before[model_id] + 1
    assert inference_sample(ids[0], "first_eligible", "success") == before_auto + 1


@pytest.mark.parametrize(
    ("upstream", "status", "outcome"),
    [
        (httpx.ConnectError("private connection"), 502, "error"),
        (httpx.ReadTimeout("private timeout"), 504, "timeout"),
        (httpx.Response(503, text="private upstream"), 502, "error"),
        (httpx.Response(200, text="private invalid response"), 502, "error"),
    ],
)
def test_one_backend_failure_does_not_affect_another(
    clean_remote_env: None,
    monkeypatch: pytest.MonkeyPatch,
    upstream: httpx.Response | Exception,
    status: int,
    outcome: str,
) -> None:
    configure(monkeypatch, [entry("real-a"), entry("real-b")])
    models = main.create_registry()
    monkeypatch.setattr(main, "registry", models)
    calls: list[str] = []
    original_client = httpx.AsyncClient

    def respond(request: httpx.Request) -> httpx.Response:
        model_id = json.loads(request.content)["model"]
        calls.append(model_id)
        if model_id == "real-a":
            if isinstance(upstream, Exception):
                raise upstream
            return upstream
        return success_response(model_id)

    monkeypatch.setattr(
        remote_backend.httpx,
        "AsyncClient",
        lambda: original_client(transport=httpx.MockTransport(respond)),
    )
    before_failure = inference_sample("real-a", "first_eligible", outcome)
    before_success = inference_sample("real-b", "explicit", "success")

    with TestClient(main.app) as client:
        failed = post(client, "auto")
        succeeded = post(client, "real-b")
        assert failed.status_code == status
        assert succeeded.status_code == 200
        assert succeeded.json()["model"] == "real-b"

    assert calls == ["real-a", "real-b"]
    assert inference_sample("real-a", "first_eligible", outcome) == before_failure + 1
    assert inference_sample("real-b", "explicit", "success") == before_success + 1
    assert "private" not in failed.text


@pytest.mark.parametrize(
    "raw",
    [
        "not-json",
        "[]",
        "{}",
        json.dumps([{"model_id": "real-a"}]),
        json.dumps([entry("real-a"), entry("real-a")]),
        json.dumps([entry("auto")]),
        json.dumps([entry("mock-small")]),
        json.dumps([entry("x" * 129)]),
        json.dumps([entry("real-a", "not-a-url")]),
        json.dumps([entry("real-a", "http://user:private-password@models.example")]),
        json.dumps([entry("real-a", "http://user:private-password@host:bad")]),
        json.dumps([entry("real-a", "https://models.example?token=private-token")]),
        json.dumps([{**entry("real-a"), "deadline_seconds": 0}]),
        json.dumps([{**entry("real-a"), "deadline_seconds": "60"}]),
        json.dumps([{**entry("real-a"), "deadline_seconds": True}]),
        json.dumps([{**entry("real-a"), "deadline_seconds": 3601}]),
        json.dumps([{**entry("real-a"), "unexpected": "private value"}]),
        json.dumps([{**entry("real-a"), "api_key": "private-literal-key"}]),
        json.dumps([entry(f"real-{index}") for index in range(9)]),
    ],
)
def test_invalid_multi_config_is_rejected_without_echoing_values(
    clean_remote_env: None, monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv("KUBEROUTE_REMOTE_MODELS_JSON", raw)
    with pytest.raises(ValueError) as error:
        main.create_registry()
    message = str(error.value)
    assert "private" not in message
    assert "models.example" not in message
    assert "real-a" not in message


@pytest.mark.parametrize("legacy_name", LEGACY_ENVS)
def test_configuration_modes_cannot_be_combined(
    clean_remote_env: None, monkeypatch: pytest.MonkeyPatch, legacy_name: str
) -> None:
    configure(monkeypatch, [entry("real-a")])
    monkeypatch.setenv(legacy_name, "private-legacy-value")
    with pytest.raises(ValueError, match="either multi-model or legacy") as error:
        main.create_registry()
    assert "private-legacy-value" not in str(error.value)


def test_api_key_environment_reference_is_resolved_without_literal_config_key(
    clean_remote_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODEL_SERVER_API_KEY", "fake-test-key")
    configure(monkeypatch, [{**entry("real-a"), "api_key_env": "MODEL_SERVER_API_KEY"}])
    assert remote_configs_from_env()[0].api_key == "fake-test-key"
    assert "fake-test-key" not in repr(remote_configs_from_env()[0])

    monkeypatch.delenv("MODEL_SERVER_API_KEY")
    with pytest.raises(ValueError, match="environment variable is unset") as error:
        remote_configs_from_env()
    assert "MODEL_SERVER_API_KEY" not in str(error.value)


def test_each_remote_uses_its_own_api_key_reference(
    clean_remote_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODEL_A_KEY", "fake-key-a")
    monkeypatch.setenv("MODEL_B_KEY", "fake-key-b")
    configure(
        monkeypatch,
        [
            {**entry("real-a"), "api_key_env": "MODEL_A_KEY"},
            {**entry("real-b"), "api_key_env": "MODEL_B_KEY"},
        ],
    )
    models = main.create_registry()
    monkeypatch.setattr(main, "registry", models)
    observed: list[tuple[str, str | None]] = []
    original_client = httpx.AsyncClient

    def respond(request: httpx.Request) -> httpx.Response:
        model_id = json.loads(request.content)["model"]
        observed.append((model_id, request.headers.get("authorization")))
        return success_response(model_id)

    monkeypatch.setattr(
        remote_backend.httpx,
        "AsyncClient",
        lambda: original_client(transport=httpx.MockTransport(respond)),
    )
    with TestClient(main.app) as client:
        assert post(client, "real-a").status_code == 200
        assert post(client, "real-b").status_code == 200

    assert observed == [("real-a", "Bearer fake-key-a"), ("real-b", "Bearer fake-key-b")]


@pytest.mark.parametrize("fail_second", [False, True])
def test_lifespan_closes_every_owned_client_even_after_partial_startup(
    monkeypatch: pytest.MonkeyPatch, fail_second: bool
) -> None:
    clients: list[httpx.AsyncClient] = []
    close_order: list[str] = []

    class TrackingBackend(OpenAICompatibleBackend):
        async def start(self) -> None:
            await super().start()
            assert self._client is not None
            clients.append(self._client)
            if fail_second and self._model_id == "real-b":
                raise RuntimeError("private startup credential")

        async def close(self) -> None:
            close_order.append(self._model_id)
            await super().close()

    models = ModelRegistry()
    for model_id in ("real-a", "real-b", "real-c"):
        models.register(model_id, TrackingBackend("https://models.example", model_id))
    monkeypatch.setattr(main, "registry", models)

    if fail_second:
        with pytest.raises(RuntimeError, match="Remote backend initialization failed") as error:
            with TestClient(main.app):
                pass
        assert "private" not in str(error.value)
        assert len(clients) == 2
        assert close_order == ["real-b", "real-a"]
    else:
        with TestClient(main.app) as client:
            assert client.get("/health").status_code == 200
            assert len(clients) == 3
            assert all(not item.is_closed for item in clients)
        assert close_order == ["real-c", "real-b", "real-a"]
    assert all(item.is_closed for item in clients)


def test_shutdown_failure_is_sanitized_and_other_clients_still_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients: list[httpx.AsyncClient] = []

    class TrackingBackend(OpenAICompatibleBackend):
        async def start(self) -> None:
            await super().start()
            assert self._client is not None
            clients.append(self._client)

        async def close(self) -> None:
            await super().close()
            if self._model_id == "real-b":
                raise RuntimeError("private shutdown credential")

    models = ModelRegistry()
    for model_id in ("real-a", "real-b"):
        models.register(model_id, TrackingBackend("https://models.example", model_id))
    monkeypatch.setattr(main, "registry", models)

    with pytest.raises(RuntimeError, match="Remote backend shutdown failed") as error:
        with TestClient(main.app) as client:
            assert client.get("/health").status_code == 200
    assert "private" not in str(error.value)
    assert len(clients) == 2
    assert all(item.is_closed for item in clients)


def test_lifespan_does_not_close_injected_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> None:
        async with httpx.AsyncClient() as first, httpx.AsyncClient() as second:
            models = ModelRegistry()
            models.register(
                "real-a", OpenAICompatibleBackend("https://one.example", "real-a", client=first)
            )
            models.register(
                "real-b", OpenAICompatibleBackend("https://two.example", "real-b", client=second)
            )
            monkeypatch.setattr(main, "registry", models)
            with TestClient(main.app) as client:
                assert client.get("/v1/models").status_code == 200
            assert not first.is_closed
            assert not second.is_closed

    asyncio.run(run())
