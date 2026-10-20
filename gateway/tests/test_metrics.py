"""HTTP metrics exposition and bounded-label behavior."""

import asyncio

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from kuberoute_gateway.main import app
from kuberoute_gateway.metrics import METRICS_REGISTRY, HTTPMetricsMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.types import Receive, Scope, Send


def sample(name: str, labels: dict[str, str] | None = None) -> float:
    return METRICS_REGISTRY.get_sample_value(name, labels or {}) or 0.0


def test_metrics_exposition_and_health_observations() -> None:
    labels = {"method": "GET", "route": "/health", "status_code": "200"}
    before_count = sample("kuberoute_http_requests_total", labels)
    before_duration_count = sample("kuberoute_http_request_duration_seconds_count", labels)
    before_duration_sum = sample("kuberoute_http_request_duration_seconds_sum", labels)

    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        response = client.get("/metrics")
        again = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"] == CONTENT_TYPE_LATEST
    assert "# TYPE kuberoute_http_requests_total counter" in response.text
    assert "# TYPE kuberoute_http_request_duration_seconds histogram" in response.text
    assert "# TYPE kuberoute_http_requests_in_progress gauge" in response.text
    assert "kuberoute_http_requests_total" in again.text
    assert sample("kuberoute_http_requests_total", labels) == before_count + 1
    assert sample("kuberoute_http_request_duration_seconds_count", labels) == (
        before_duration_count + 1
    )
    assert sample("kuberoute_http_request_duration_seconds_sum", labels) > before_duration_sum
    assert sample("kuberoute_http_requests_in_progress") == 0
    assert 'route="/metrics"' not in response.text


def test_http_status_and_unknown_paths_have_bounded_labels() -> None:
    unknown_labels = {"method": "GET", "route": "unmatched", "status_code": "404"}
    model_labels = {"method": "POST", "route": "/v1/chat/completions", "status_code": "404"}
    validation_labels = {"method": "POST", "route": "/v1/chat/completions", "status_code": "422"}
    before = [
        sample("kuberoute_http_requests_total", labels)
        for labels in (unknown_labels, model_labels, validation_labels)
    ]

    with TestClient(app) as client:
        assert client.get("/private-path-one").status_code == 404
        assert client.get("/private-path-two").status_code == 404
        assert (
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "missing",
                    "messages": [{"role": "user", "content": "private prompt"}],
                },
            ).status_code
            == 404
        )
        assert client.post("/v1/chat/completions", json={}).status_code == 422
        metrics = client.get("/metrics").text

    assert sample("kuberoute_http_requests_total", unknown_labels) == before[0] + 2
    assert sample("kuberoute_http_requests_total", model_labels) == before[1] + 1
    assert sample("kuberoute_http_requests_total", validation_labels) == before[2] + 1
    assert "private-path-one" not in metrics
    assert "private-path-two" not in metrics
    assert "private prompt" not in metrics
    assert "missing" not in metrics


def test_parameterized_route_uses_template_instead_of_request_path() -> None:
    test_app = FastAPI()
    test_app.add_middleware(HTTPMetricsMiddleware)

    @test_app.get("/items/{item_id}")
    def get_item(item_id: str) -> dict[str, str]:
        return {"item_id": item_id}

    labels = {"method": "GET", "route": "/items/{item_id}", "status_code": "200"}
    before = sample("kuberoute_http_requests_total", labels)
    with TestClient(test_app) as client:
        assert client.get("/items/private-identifier").status_code == 200

    assert sample("kuberoute_http_requests_total", labels) == before + 1
    assert "private-identifier" not in generate_latest(METRICS_REGISTRY).decode()


def test_unexpected_endpoint_failure_records_500() -> None:
    test_app = FastAPI()
    test_app.add_middleware(HTTPMetricsMiddleware)

    @test_app.get("/fails")
    def fails() -> None:
        raise RuntimeError("internal details")

    labels = {"method": "GET", "route": "/fails", "status_code": "500"}
    before = sample("kuberoute_http_requests_total", labels)
    with TestClient(test_app, raise_server_exceptions=False) as client:
        response = client.get("/fails")

    assert response.status_code == 500
    assert sample("kuberoute_http_requests_total", labels) == before + 1
    assert sample("kuberoute_http_requests_in_progress") == 0
    assert "internal details" not in generate_latest(METRICS_REGISTRY).decode()


def test_in_progress_gauge_tracks_a_concurrent_request() -> None:
    async def run() -> None:
        entered = asyncio.Event()
        release = asyncio.Event()

        async def slow_app(scope: Scope, receive: Receive, send: Send) -> None:
            entered.set()
            await release.wait()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        transport = httpx.ASGITransport(app=HTTPMetricsMiddleware(slow_app))
        before = sample("kuberoute_http_requests_in_progress")
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            request = asyncio.create_task(client.get("/slow"))
            await entered.wait()
            assert sample("kuberoute_http_requests_in_progress") == before + 1
            release.set()
            response = await request
        assert response.status_code == 200
        assert sample("kuberoute_http_requests_in_progress") == before

    asyncio.run(run())
