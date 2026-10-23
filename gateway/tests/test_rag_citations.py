"""Optional verified citations preserve the existing hash-based RAG contract."""

import json
from pathlib import Path

import kuberoute_gateway.main as gateway_main
import pytest
from fastapi.testclient import TestClient
from kuberoute_gateway.backends import ModelOutput
from kuberoute_gateway.knowledge_catalog import CatalogError, load_catalog
from kuberoute_gateway.rag.ingestion import load_document


class LocalProvider:
    model_id = "synthetic-embed"

    def __init__(self, base_url: str, model_id: str):
        self.model_id = model_id

    async def start(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def embed(self, texts: list[str]) -> tuple[tuple[float, ...], ...]:
        return tuple((1.0, 0.0) for _ in texts)


def _configure(monkeypatch: pytest.MonkeyPatch, paths: list[Path], manifest: Path | None) -> None:
    monkeypatch.setenv("KUBEROUTE_RAG_ENABLED", "true")
    monkeypatch.setenv("KUBEROUTE_RAG_DOCUMENT_PATHS_JSON", json.dumps([str(p) for p in paths]))
    monkeypatch.setenv("KUBEROUTE_RAG_EMBEDDING_MODEL", "synthetic-embed")
    monkeypatch.setenv("KUBEROUTE_RAG_EMBEDDING_BASE_URL", "http://127.0.0.1:11434")
    if manifest is None:
        monkeypatch.delenv("KUBEROUTE_RAG_SOURCE_MANIFEST", raising=False)
    else:
        monkeypatch.setenv("KUBEROUTE_RAG_SOURCE_MANIFEST", str(manifest))
    monkeypatch.setattr(gateway_main, "OllamaEmbeddingProvider", LocalProvider)


def _manifest(tmp_path: Path, document: Path, *, year: int | None = 2023) -> Path:
    record = {
        "id": "verified_study",
        "document": f"sources/{document.name}",
        "title": "Synthetic resistance training study",
        "authors": "Research Group",
        "publication_year": year,
        "original_url": "https://example.org/publication",
        "source_type": "systematic_review",
        "topics": ["training"],
        "doi": None,
        "pmid": None,
        "verified_on": "2026-10-23",
    }
    if year is None:
        record["publication_date_note"] = "Date not stated"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"schema_version": 1, "collection_id": "synthetic_collection", "sources": [record]}
        ),
        encoding="utf-8",
    )
    return manifest


@pytest.mark.parametrize("year", [2023, None])
def test_verified_citation_maps_exact_ingested_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, year: int | None
) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    document = sources / "private-name.md"
    document.write_text("Synthetic training evidence.", encoding="utf-8")
    manifest = _manifest(tmp_path, document, year=year)
    _configure(monkeypatch, [document], manifest)

    async def complete(request):
        return ModelOutput("Evidence [1]", prompt_tokens=2, output_tokens=3)

    monkeypatch.setattr(gateway_main.registry.get("mock-small"), "complete", complete)
    with TestClient(gateway_main.app) as client:
        runtime = client.app.state.rag_runtime
        assert set(runtime.catalog_by_source_id) == {load_document(document).source_id}
        response = client.post(
            "/v1/rag/chat/completions", json={"model": "mock-small", "question": "Evidence?"}
        )
    assert response.status_code == 200
    source = response.json()["sources"][0]
    assert source["marker"] == "[1]"
    assert source["source_id"] == load_document(document).source_id
    assert source["catalog_id"] == "verified_study"
    assert source["title"] == "Synthetic resistance training study"
    assert source["original_url"] == "https://example.org/publication"
    assert source.get("publication_year") == year
    assert str(tmp_path) not in response.text
    assert "private-name" not in response.text


def test_uncataloged_source_keeps_hash_reference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    document = tmp_path / "uncataloged.md"
    document.write_text("Synthetic training evidence.", encoding="utf-8")
    _configure(monkeypatch, [document], None)

    async def complete(request):
        return ModelOutput("Evidence [1]")

    monkeypatch.setattr(gateway_main.registry.get("mock-small"), "complete", complete)
    with TestClient(gateway_main.app) as client:
        response = client.post(
            "/v1/rag/chat/completions", json={"model": "mock-small", "question": "Evidence?"}
        )
    assert response.status_code == 200
    source = response.json()["sources"][0]
    assert set(source) == {"marker", "source_id", "chunk_id", "chunk_index"}


def test_configured_catalog_does_not_label_an_unknown_or_uncited_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    catalog_document = sources / "catalog.md"
    catalog_document.write_text("Catalog evidence.", encoding="utf-8")
    unknown_document = tmp_path / "unknown.md"
    unknown_document.write_text("Unknown evidence.", encoding="utf-8")
    manifest = _manifest(tmp_path, catalog_document)
    _configure(monkeypatch, [unknown_document], manifest)

    async def complete(request):
        return ModelOutput("Unknown evidence [1]")

    monkeypatch.setattr(gateway_main.registry.get("mock-small"), "complete", complete)
    with TestClient(gateway_main.app) as client:
        response = client.post(
            "/v1/rag/chat/completions", json={"model": "mock-small", "question": "Unknown?"}
        )
        assert response.status_code == 200
        assert set(response.json()["sources"][0]) == {
            "marker",
            "source_id",
            "chunk_id",
            "chunk_index",
        }

        async def uncited(request):
            return ModelOutput("Answer without citation")

        monkeypatch.setattr(gateway_main.registry.get("mock-small"), "complete", uncited)
        no_citation = client.post(
            "/v1/rag/chat/completions", json={"model": "mock-small", "question": "Unknown?"}
        )
        assert no_citation.status_code == 200
        assert no_citation.json()["sources"] == []


def test_invalid_configured_manifest_fails_rag_safely(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    document = tmp_path / "private-document.md"
    document.write_text("Synthetic evidence.", encoding="utf-8")
    _configure(monkeypatch, [document], tmp_path / "private-missing-manifest.json")
    with TestClient(gateway_main.app) as client:
        response = client.post(
            "/v1/rag/chat/completions", json={"model": "mock-small", "question": "Evidence?"}
        )
        assert response.status_code == 503
        assert client.get("/health").status_code == 200
        assert str(tmp_path) not in response.text


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "file:///private",
        "https://127.0.0.1/private",
        "https://example.org/path#fragment",
        "https://example.org:8443/path",
    ],
)
def test_unsafe_manifest_links_fail_without_exposing_details(tmp_path: Path, url: str) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    document = sources / "example.md"
    document.write_text("Synthetic evidence.", encoding="utf-8")
    manifest = _manifest(tmp_path, document)
    data = json.loads(manifest.read_text("utf-8"))
    data["sources"][0]["original_url"] = url
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(CatalogError) as error:
        load_catalog(manifest)
    assert url not in str(error.value)
    assert str(tmp_path) not in str(error.value)
