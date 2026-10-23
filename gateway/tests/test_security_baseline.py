"""HTTP privacy and security-header regression tests."""

from fastapi.testclient import TestClient
from kuberoute_gateway.main import app
from kuberoute_gateway.schemas import MAX_CHAT_MESSAGE_CHARS, MAX_CHAT_MESSAGES


def test_invalid_chat_requests_do_not_echo_submitted_content() -> None:
    marker = "synthetic-private-text"
    with TestClient(app) as client:
        malformed = client.post(
            "/v1/chat/completions",
            json={"model": "mock-small", "messages": [{"role": "tool", "content": marker}]},
        )
        oversized = client.post(
            "/v1/chat/completions",
            json={
                "model": "mock-small",
                "messages": [{"role": "user", "content": marker + "x" * MAX_CHAT_MESSAGE_CHARS}],
            },
        )
        too_many = client.post(
            "/v1/chat/completions",
            json={
                "model": "mock-small",
                "messages": [
                    {"role": "user", "content": marker} for _ in range(MAX_CHAT_MESSAGES + 1)
                ],
            },
        )
        unknown = client.post(
            "/v1/chat/completions",
            json={"model": marker, "messages": [{"role": "user", "content": "hi"}]},
        )
        invalid_json = client.post(
            "/v1/chat/completions",
            content='{"model": "mock-small", "messages": ["' + marker,
            headers={"content-type": "application/json"},
        )

    for response in (malformed, oversized, too_many, invalid_json):
        assert response.status_code == 422
        assert response.json() == {"detail": "Chat request is invalid"}
        assert marker not in response.text
    assert unknown.status_code == 404
    assert unknown.json() == {"detail": "Unknown model"}
    assert marker not in unknown.text


def test_security_headers_cover_existing_api_and_static_responses() -> None:
    with TestClient(app) as client:
        responses = [
            client.get("/health"),
            client.get("/v1/models"),
            client.get("/metrics"),
            client.get("/"),
            client.post("/v1/chat/completions", json={"model": "mock-small", "messages": []}),
        ]

    for response in responses:
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["referrer-policy"] == "no-referrer"
        assert response.headers["x-frame-options"] == "DENY"
