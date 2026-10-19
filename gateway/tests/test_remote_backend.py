"""HTTP adapter tests using in-process mocked upstream responses."""

import asyncio
import json
from dataclasses import replace

import httpx
import pytest
from fastapi.testclient import TestClient
from kuberoute_gateway import main, remote_backend
from kuberoute_gateway.backends import ChatInputMessage, NormalizedChatRequest
from kuberoute_gateway.main import create_registry
from kuberoute_gateway.registry import ModelRegistry
from kuberoute_gateway.remote_backend import (
    BackendConnectionError,
    BackendHTTPError,
    BackendResponseError,
    BackendTimeoutError,
    OpenAICompatibleBackend,
    remote_config_from_env,
)


def _request() -> NormalizedChatRequest:
    return NormalizedChatRequest(
        model="remote-model",
        messages=(ChatInputMessage(role="user", content="Hello"),),
        temperature=0.5,
    )


def test_remote_success_passes_request_and_reads_completion() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://models.example/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-key"
        assert json.loads(request.content) == {
            "model": "remote-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.5,
            "seed": 42,
            "max_tokens": 32,
            "reasoning_effort": "none",
            "stream": False,
        }
        assert request.extensions["timeout"] == {
            "connect": 3.0,
            "read": 30.0,
            "write": 10.0,
            "pool": 3.0,
        }
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-upstream",
                "model": "remote-model",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Remote reply"},
                        "index": 0,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
            },
        )

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            backend = OpenAICompatibleBackend(
                "https://models.example", "remote-model", "test-key", client
            )
            output = await backend.complete(
                replace(_request(), seed=42, max_tokens=32, reasoning_effort="none")
            )
            assert output.content == "Remote reply"
            assert output.prompt_tokens == 2
            assert output.output_tokens == 2

    asyncio.run(run())


@pytest.mark.parametrize(
    ("upstream_error", "expected_error"),
    [
        (httpx.ReadTimeout("secret timeout"), BackendTimeoutError),
        (httpx.ConnectError("secret connection"), BackendConnectionError),
    ],
)
def test_remote_network_errors_are_classified(
    upstream_error: Exception, expected_error: type[Exception]
) -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise upstream_error

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as client:
            backend = OpenAICompatibleBackend(
                "https://models.example", "remote-model", client=client
            )
            with pytest.raises(expected_error) as error:
                await backend.complete(_request())
            assert "secret" not in str(error.value)

    asyncio.run(run())


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="not JSON"),
        httpx.Response(200, json={"choices": []}),
        httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "user", "content": "x"},
                        "finish_reason": "stop",
                    }
                ]
            },
        ),
        httpx.Response(200, json={"choices": [{"message": {"role": "assistant"}}]}),
        httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "x"},
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        ),
        httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "x"},
                        "finish_reason": "stop",
                    },
                    {
                        "index": 1,
                        "message": {"role": "assistant", "content": "y"},
                        "finish_reason": "stop",
                    },
                ]
            },
        ),
    ],
)
def test_remote_invalid_response_is_rejected(response: httpx.Response) -> None:
    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: response)
        ) as client:
            backend = OpenAICompatibleBackend(
                "https://models.example", "remote-model", client=client
            )
            with pytest.raises(BackendResponseError):
                await backend.complete(_request())

    asyncio.run(run())


def test_remote_http_error_is_classified() -> None:
    async def run() -> None:
        transport = httpx.MockTransport(lambda request: httpx.Response(503, text="secret error"))
        async with httpx.AsyncClient(transport=transport) as client:
            backend = OpenAICompatibleBackend(
                "https://models.example", "remote-model", client=client
            )
            with pytest.raises(BackendHTTPError) as error:
                await backend.complete(_request())
            assert "secret" not in str(error.value)

    asyncio.run(run())


def test_remote_configuration_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "KUBEROUTE_REMOTE_BASE_URL",
        "KUBEROUTE_REMOTE_MODEL_ID",
        "KUBEROUTE_REMOTE_API_KEY",
        "KUBEROUTE_REMOTE_DEADLINE_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    assert remote_config_from_env() is None
    assert create_registry().model_ids() == ("mock-small", "mock-large")

    monkeypatch.setenv("KUBEROUTE_REMOTE_BASE_URL", "https://models.example")
    with pytest.raises(ValueError, match="both"):
        remote_config_from_env()
    monkeypatch.setenv("KUBEROUTE_REMOTE_MODEL_ID", "remote-model")
    monkeypatch.setenv("KUBEROUTE_REMOTE_API_KEY", "test-key")
    monkeypatch.setenv("KUBEROUTE_REMOTE_DEADLINE_SECONDS", "60")
    assert create_registry().model_ids() == ("mock-small", "mock-large", "remote-model")
    assert remote_config_from_env().deadline_seconds == 60

    monkeypatch.setenv("KUBEROUTE_REMOTE_DEADLINE_SECONDS", "nan")
    with pytest.raises(ValueError, match="positive"):
        remote_config_from_env()


@pytest.mark.parametrize(
    ("upstream", "status", "detail"),
    [
        (
            httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Hi"},
                            "finish_reason": "stop",
                        }
                    ]
                },
            ),
            200,
            None,
        ),
        (httpx.ReadTimeout("secret timeout"), 504, "Upstream model timed out"),
        (httpx.ConnectError("secret connection"), 502, "Upstream model failed"),
        (httpx.Response(503, text="secret failure"), 502, "Upstream model failed"),
        (httpx.Response(200, text="secret invalid JSON"), 502, "Upstream model failed"),
        (
            httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Hi"},
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
            ),
            502,
            "Upstream model failed",
        ),
    ],
)
def test_gateway_maps_remote_results_safely(
    monkeypatch: pytest.MonkeyPatch,
    upstream: httpx.Response | Exception,
    status: int,
    detail: str | None,
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if isinstance(upstream, Exception):
            raise upstream
        return upstream

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            models = ModelRegistry()
            models.register(
                "remote-model",
                OpenAICompatibleBackend("https://models.example", "remote-model", client=client),
            )
            monkeypatch.setattr(main, "registry", models)
            with TestClient(main.app) as api:
                response = api.post(
                    "/v1/chat/completions",
                    json={"model": "remote-model", "messages": [{"role": "user", "content": "Hi"}]},
                )
            assert response.status_code == status
            if detail is None:
                assert response.json()["choices"][0]["message"]["content"] == "Hi"
            else:
                assert response.json() == {"detail": detail}
                assert "secret" not in response.text

    asyncio.run(run())


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_gateway_preserves_remote_finish_reason(
    monkeypatch: pytest.MonkeyPatch, finish_reason: str
) -> None:
    response = httpx.Response(
        200,
        json={
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Partial answer"},
                    "finish_reason": finish_reason,
                }
            ]
        },
    )

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: response)
        ) as client:
            models = ModelRegistry()
            models.register(
                "remote-model",
                OpenAICompatibleBackend("https://models.example", "remote-model", client=client),
            )
            monkeypatch.setattr(main, "registry", models)
            with TestClient(main.app) as api:
                result = api.post(
                    "/v1/chat/completions",
                    json={"model": "remote-model", "messages": [{"role": "user", "content": "Hi"}]},
                )
            assert result.status_code == 200
            assert result.json()["choices"][0]["finish_reason"] == finish_reason

    asyncio.run(run())


def test_concurrent_requests_reuse_injected_client() -> None:
    seen = 0
    in_flight = 0
    peak = 0

    async def respond(request: httpx.Request) -> httpx.Response:
        nonlocal seen, in_flight, peak
        seen += 1
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hi"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            backend = OpenAICompatibleBackend(
                "https://models.example", "remote-model", client=client
            )
            results = await asyncio.gather(*(backend.complete(_request()) for _ in range(5)))
            assert [result.content for result in results] == ["Hi"] * 5
            assert seen == 5
            assert peak > 1
            await backend.close()
            assert not client.is_closed

    asyncio.run(run())


def test_lifespan_starts_and_closes_owned_client_once(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[httpx.AsyncClient] = []
    original_client = httpx.AsyncClient
    requests = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hi"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    def new_client() -> httpx.AsyncClient:
        client = original_client(transport=httpx.MockTransport(respond))
        created.append(client)
        return client

    backend = OpenAICompatibleBackend("https://models.example", "remote-model")
    models = ModelRegistry()
    models.register("remote-model", backend)
    monkeypatch.setattr(main, "registry", models)
    monkeypatch.setattr(remote_backend.httpx, "AsyncClient", new_client)
    with TestClient(main.app) as api:
        assert api.get("/v1/models").status_code == 200
        assert api.get("/health").status_code == 200
        for _ in range(2):
            assert (
                api.post(
                    "/v1/chat/completions",
                    json={"model": "remote-model", "messages": [{"role": "user", "content": "Hi"}]},
                ).status_code
                == 200
            )
        assert len(created) == 1
        assert not created[0].is_closed
    assert created[0].is_closed
    assert requests == 2
    asyncio.run(backend.close())
    assert len(created) == 1


def test_overall_deadline_times_out_and_maps_to_504(monkeypatch: pytest.MonkeyPatch) -> None:
    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.05)
        return httpx.Response(200, json={"choices": []})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(slow)) as client:
            backend = OpenAICompatibleBackend(
                "https://models.example", "remote-model", client=client, deadline_seconds=0.01
            )
            models = ModelRegistry()
            models.register("remote-model", backend)
            monkeypatch.setattr(main, "registry", models)
            with TestClient(main.app) as api:
                result = api.post(
                    "/v1/chat/completions",
                    json={"model": "remote-model", "messages": [{"role": "user", "content": "Hi"}]},
                )
            assert result.status_code == 504
            assert result.json() == {"detail": "Upstream model timed out"}

    asyncio.run(run())
