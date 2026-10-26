"""Network-free tests for opt-in, isolated, memory-only document uploads."""

import json
from pathlib import Path

import kuberoute_gateway.main as gateway_main
import pytest
from fastapi.testclient import TestClient
from kuberoute_gateway.rag.uploads import MAX_UPLOAD_BYTES


class FakeProvider:
    calls = 0
    started = 0
    closed = 0
    fail_on_call: int | None = None

    def __init__(self, base_url: str, model_id: str):
        self.model_id = model_id

    async def start(self) -> None:
        type(self).started += 1

    async def close(self) -> None:
        type(self).closed += 1

    async def embed(self, texts: list[str]) -> tuple[tuple[float, ...], ...]:
        type(self).calls += 1
        if type(self).calls == type(self).fail_on_call:
            raise RuntimeError("private embedding failure")
        return tuple((1.0, 0.0) for _ in texts)


@pytest.fixture
def enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeProvider.calls = FakeProvider.started = FakeProvider.closed = 0
    FakeProvider.fail_on_call = None
    monkeypatch.setenv("KUBEROUTE_UPLOADS_ENABLED", "true")
    monkeypatch.setenv("KUBEROUTE_RAG_ENABLED", "false")
    monkeypatch.setenv("KUBEROUTE_RAG_EMBEDDING_MODEL", "synthetic-embed")
    monkeypatch.setenv("KUBEROUTE_RAG_EMBEDDING_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setattr(gateway_main, "OllamaEmbeddingProvider", FakeProvider)


def _upload(client: TestClient, name: str, data: bytes, media_type: str = "text/plain"):
    return client.post(
        "/v1/rag/uploads",
        content=data,
        headers={"X-Document-Name": name, "Content-Type": media_type},
    )


def test_upload_and_list_only_successful_safe_metadata(enabled: None) -> None:
    with TestClient(gateway_main.app) as client:
        assert client.get("/v1/rag/uploads").json() == {"object": "list", "data": []}
        result = _upload(client, "Synthetic Study.md", b"Training improves strength.\n")
        assert result.status_code == 201
        document = result.json()
        assert set(document) == {"id", "name", "byte_size", "chunk_count"}
        assert len(document["id"]) == 64
        assert document["name"] == "Synthetic Study.md"
        assert document["byte_size"] == len(b"Training improves strength.\n")
        assert document["chunk_count"] == 1
        assert "no-store" in result.headers["cache-control"]
        listing = client.get("/v1/rag/uploads")
        assert listing.status_code == 200
        assert listing.json() == {"object": "list", "data": [document]}
        assert listing.headers["cache-control"] == "no-store"
        assert "Training improves" not in listing.text
        assert len(client.app.state.upload_runtime.retriever) == 1
    assert FakeProvider.started == FakeProvider.closed == 1

    # A new application lifespan owns a fresh in-memory collection.
    with TestClient(gateway_main.app) as client:
        assert client.get("/v1/rag/uploads").json()["data"] == []
    assert FakeProvider.started == FakeProvider.closed == 2


@pytest.mark.parametrize(
    ("name", "content", "media_type", "status"),
    [
        ("bad.md", b"\xff", "text/markdown", 422),
        ("file.pdf", b"%PDF-1.7", "application/pdf", 415),
        ("empty.txt", b"", "text/plain", 422),
        ("blank.md", b"  \n\t", "text/markdown", 422),
        ("../private.txt", b"private", "text/plain", 422),
        ("C:\\private.txt", b"private", "text/plain", 422),
        ("markup.md", b"text", "text/html", 415),
    ],
)
def test_invalid_uploads_are_sanitized_and_never_listed(
    enabled: None, name: str, content: bytes, media_type: str, status: int
) -> None:
    with TestClient(gateway_main.app) as client:
        response = _upload(client, name, content, media_type)
        assert response.status_code == status
        assert name not in response.text
        assert "private" not in response.text
        assert client.get("/v1/rag/uploads").json()["data"] == []
        assert len(client.app.state.upload_runtime.retriever) == 0


def test_oversized_body_is_rejected_before_indexing(enabled: None) -> None:
    with TestClient(gateway_main.app) as client:
        response = _upload(client, "oversized.txt", b"a" * (MAX_UPLOAD_BYTES + 1))
        assert response.status_code == 413
        assert client.get("/v1/rag/uploads").json()["data"] == []
        assert len(client.app.state.upload_runtime.retriever) == 0


def test_stream_limit_does_not_trust_content_length(enabled: None) -> None:
    with TestClient(gateway_main.app) as client:
        response = client.post(
            "/v1/rag/uploads",
            content=b"a" * (MAX_UPLOAD_BYTES + 1),
            headers={
                "X-Document-Name": "oversized.txt",
                "Content-Type": "text/plain",
                "Content-Length": "1",
            },
        )
        assert response.status_code == 413
        assert client.get("/v1/rag/uploads").json()["data"] == []


def test_second_batch_index_failure_preserves_previous_document(enabled: None) -> None:
    with TestClient(gateway_main.app) as client:
        first = _upload(client, "first.txt", b"First source text.")
        assert first.status_code == 201
        FakeProvider.fail_on_call = 3
        failure = _upload(client, "second.txt", b"Second source text. " * 900)
        assert failure.status_code == 502
        assert failure.json() == {"detail": "Document indexing failed"}
        assert "private embedding failure" not in failure.text
        assert client.get("/v1/rag/uploads").json()["data"] == [first.json()]
        assert len(client.app.state.upload_runtime.retriever) == 1


def test_curated_index_is_unchanged_by_uploads(
    enabled: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    curated = tmp_path / "curated.md"
    curated.write_text("Curated baseline evidence.", encoding="utf-8")
    monkeypatch.setenv("KUBEROUTE_RAG_ENABLED", "true")
    monkeypatch.setenv("KUBEROUTE_RAG_DOCUMENT_PATHS_JSON", json.dumps([str(curated)]))
    with TestClient(gateway_main.app) as client:
        curated_runtime = client.app.state.rag_runtime
        upload_runtime = client.app.state.upload_runtime
        assert curated_runtime is not None
        assert curated_runtime.retriever is not upload_runtime.retriever
        assert len(curated_runtime.retriever) == 1
        curated_chunk_ids = curated_runtime.retriever._semantic.chunk_ids()
        assert _upload(client, "uploaded.md", b"Uploaded evidence only.").status_code == 201
        assert len(curated_runtime.retriever) == 1
        assert len(upload_runtime.retriever) == 1
        assert curated_runtime.retriever._semantic.chunk_ids() == curated_chunk_ids
        assert upload_runtime.retriever._semantic.chunk_ids().isdisjoint(curated_chunk_ids)


def test_disabled_and_unavailable_uploads_do_not_affect_health(
    monkeypatch: pytest.MonkeyPatch, enabled: None
) -> None:
    monkeypatch.setenv("KUBEROUTE_UPLOADS_ENABLED", "false")
    with TestClient(gateway_main.app) as client:
        assert client.get("/v1/rag/uploads").status_code == 503
        assert _upload(client, "test.txt", b"text").status_code == 503
        assert client.get("/health").status_code == 200
    monkeypatch.setenv("KUBEROUTE_UPLOADS_ENABLED", "true")
    monkeypatch.delenv("KUBEROUTE_RAG_EMBEDDING_MODEL")
    with TestClient(gateway_main.app) as client:
        response = client.get("/v1/rag/uploads")
        assert response.status_code == 503
        assert response.json() == {"detail": "Document upload index is unavailable"}
        assert client.get("/health").status_code == 200


def test_upload_provider_start_failure_closes_client_and_preserves_other_routes(
    enabled: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailingProvider(FakeProvider):
        async def start(self) -> None:
            raise RuntimeError("private connection detail")

    monkeypatch.setattr(gateway_main, "OllamaEmbeddingProvider", FailingProvider)
    with TestClient(gateway_main.app) as client:
        assert client.get("/v1/rag/uploads").status_code == 503
        assert client.get("/health/live").status_code == 200
    assert FailingProvider.closed == 1
