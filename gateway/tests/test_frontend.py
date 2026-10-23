"""Optional prebuilt dashboard serving keeps the gateway API intact."""

from pathlib import Path

from fastapi.testclient import TestClient
from kuberoute_gateway.main import app


def test_dashboard_is_absent_when_no_compiled_files(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KUBEROUTE_FRONTEND_DIST_DIR", str(tmp_path))
    with TestClient(app) as client:
        assert client.get("/").status_code == 404
        assert client.get("/assets/main.js").status_code == 404
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/v1/models").status_code == 200


def test_serves_compiled_dashboard_and_assets_on_gateway_origin(
    monkeypatch, tmp_path: Path
) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    (tmp_path / "index.html").write_text('<div id="root"></div>', encoding="utf-8")
    (assets / "main.js").write_text("export const ready = true;", encoding="utf-8")
    monkeypatch.setenv("KUBEROUTE_FRONTEND_DIST_DIR", str(tmp_path))

    with TestClient(app) as client:
        index = client.get("/")
        script = client.get("/assets/main.js")
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 200
        assert client.get("/metrics").status_code == 200
        completion = client.post(
            "/v1/chat/completions",
            json={"model": "mock-small", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert index.status_code == 200
    assert index.headers["content-type"].startswith("text/html")
    assert 'id="root"' in index.text
    assert index.headers["cache-control"] == "no-store"
    assert "default-src 'self'" in index.headers["content-security-policy"]
    assert script.status_code == 200
    assert script.headers["content-type"].split(";")[0] in {
        "text/javascript",
        "application/javascript",
    }
    assert completion.status_code == 200


def test_asset_route_cannot_escape_compiled_asset_directory(monkeypatch, tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    (tmp_path / "index.html").write_text("dashboard", encoding="utf-8")
    (tmp_path / "private.txt").write_text("private", encoding="utf-8")
    monkeypatch.setenv("KUBEROUTE_FRONTEND_DIST_DIR", str(tmp_path))

    with TestClient(app) as client:
        response = client.get("/assets/..%2Fprivate.txt")
        missing = client.get("/assets/not-found.js")

    assert response.status_code == 404
    assert "private" not in response.text
    assert missing.status_code == 404
