"""Initial gateway contract tests."""

from fastapi.testclient import TestClient
from kuberoute_gateway.main import app


def test_health_returns_ok() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_unknown_route_returns_404() -> None:
    with TestClient(app) as client:
        response = client.get("/does-not-exist")
    assert response.status_code == 404
