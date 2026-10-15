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


def test_liveness_returns_alive() -> None:
    with TestClient(app) as client:
        response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_readiness_returns_ready() -> None:
    with TestClient(app) as client:
        response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
