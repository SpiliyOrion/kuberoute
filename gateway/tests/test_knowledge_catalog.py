"""Trust-boundary tests for a general-purpose explicit document catalog."""

import json
from pathlib import Path

import pytest
from kuberoute_gateway.knowledge_catalog import CatalogError, load_catalog
from kuberoute_gateway.knowledge_evaluation import (
    EvaluationQuestion,
    load_questions,
    summarize_retrieval,
)
from kuberoute_gateway.rag.ingestion import chunk_document, load_documents

ROOT = Path(__file__).resolve().parents[2]
COLLECTION = ROOT / "knowledge" / "exercise_science"


def test_checked_in_catalog_is_complete_and_ingestion_compatible() -> None:
    sources = load_catalog(COLLECTION / "manifest.json")
    assert len(sources) == 12
    assert len({source.id for source in sources}) == len(sources)
    documents = load_documents([source.document for source in sources])
    chunks = tuple(chunk for document in documents for chunk in chunk_document(document))
    assert len(documents) == 12
    assert len(chunks) >= 12
    assert all(chunk.text.strip() for chunk in chunks)
    assert all(
        chunk.source_id in {document.source_id for document in documents} for chunk in chunks
    )
    manifest = json.loads((COLLECTION / "manifest.json").read_text("utf-8"))
    for source, record in zip(sources, manifest["sources"], strict=True):
        summary = source.document.read_text("utf-8")
        assert f"- Source ID: {source.id}" in summary
        assert f"- Original URL: {record['original_url']}" in summary
        assert "## Findings" in summary or "## Framework" in summary
        assert "## Practical interpretation" in summary
        assert "## Limitations and uncertainty" in summary


def test_evaluation_set_has_unique_questions_and_valid_source_references() -> None:
    catalog_ids = {source.id for source in load_catalog(COLLECTION / "manifest.json")}
    data = json.loads((COLLECTION / "evaluation" / "questions.json").read_text("utf-8"))
    assert data["schema_version"] == 1
    assert data["collection_id"] == "exercise_science_v1"
    questions = data["questions"]
    assert 15 <= len(questions) <= 20
    assert len({record["id"] for record in questions}) == len(questions)
    assert len({record["question"] for record in questions}) == len(questions)
    for record in questions:
        assert set(record) == {
            "id",
            "category",
            "question",
            "expected_source_ids",
            "answer_points",
            "caveats",
            "sufficient_evidence",
        }
        assert isinstance(record["sufficient_evidence"], bool)
        assert record["answer_points"] and record["caveats"]
        assert set(record["expected_source_ids"]).issubset(catalog_ids)
        if not record["sufficient_evidence"]:
            assert record["expected_source_ids"] == []
    loaded = load_questions(COLLECTION / "evaluation" / "questions.json", catalog_ids)
    assert len(loaded) == 18
    assert "What" not in repr(loaded[0])


def test_retrieval_summary_does_not_treat_unanswerable_as_correctness() -> None:
    questions = [
        EvaluationQuestion(
            "first", "Private synthetic question", ("source_one", "source_two"), True
        ),
        EvaluationQuestion("second", "Another question", ("source_two",), True),
        EvaluationQuestion("third", "Unanswerable", (), False),
    ]
    summary = summarize_retrieval(
        "hybrid",
        [
            (questions[0], ("other", "source_two")),
            (questions[1], ("other",)),
            (questions[2], ("source_one",)),
        ],
    )
    assert summary.questions == 3
    assert summary.answerable == 2
    assert summary.hit_at_k == 1
    assert summary.mean_source_recall == 0.25
    assert summary.mean_reciprocal_rank == 0.25
    assert summary.unanswerable_with_results == 1
    assert "Private synthetic question" not in repr(summary)


def test_malformed_evaluation_record_fails_safely(tmp_path: Path) -> None:
    path = tmp_path / "questions.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "questions": [
                    {
                        "id": "test",
                        "question": "Sensitive synthetic question",
                        "expected_source_ids": ["missing"],
                        "sufficient_evidence": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(CatalogError) as error:
        load_questions(path, {"known"})
    assert "Sensitive" not in str(error.value)
    assert str(path) not in str(error.value)


@pytest.fixture
def sample_catalog(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "sample.md").write_text("Harmless synthetic content.", encoding="utf-8")
    record = {
        "id": "example_source",
        "document": "sources/sample.md",
        "title": "Synthetic source",
        "authors": "Example Research Group",
        "publication_year": 2024,
        "original_url": "https://example.org/study",
        "doi": None,
        "pmid": None,
        "source_type": "review",
        "topics": ["example"],
        "verified_on": "2026-10-23",
    }
    data: dict[str, object] = {
        "schema_version": 1,
        "collection_id": "example_collection",
        "sources": [record],
    }
    return tmp_path / "manifest.json", data


def _write(manifest: Path, data: dict[str, object]) -> None:
    manifest.write_text(json.dumps(data), encoding="utf-8")


def test_valid_synthetic_catalog(sample_catalog: tuple[Path, dict[str, object]]) -> None:
    manifest, data = sample_catalog
    _write(manifest, data)
    assert [source.id for source in load_catalog(manifest)] == ["example_source"]
    assert "sources" not in repr(load_catalog(manifest)[0])


@pytest.mark.parametrize(
    "field,value",
    [
        ("document", "../private.md"),
        ("document", "C:\\private\\file.md"),
        ("document", "sources/sample.pdf"),
        ("document", "sources/missing.md"),
        ("title", ""),
        ("authors", ""),
        ("publication_year", "2024"),
        ("original_url", "https://" + "user:placeholder@" + "example.org/study"),
        ("original_url", "http://example.org/study"),
        ("original_url", "https://127.0.0.1/study"),
        ("original_url", "https://intranet.local/study"),
        ("original_url", "https://example.org:8443/study"),
        ("verified_on", "not-a-date"),
        ("topics", []),
    ],
)
def test_invalid_records_fail_without_echoing_values(
    sample_catalog: tuple[Path, dict[str, object]], field: str, value: object
) -> None:
    manifest, data = sample_catalog
    record = data["sources"][0]
    record[field] = value
    _write(manifest, data)
    with pytest.raises(CatalogError) as error:
        load_catalog(manifest)
    assert "placeholder" not in str(error.value)
    assert str(manifest.parent) not in str(error.value)
    assert "Harmless synthetic content" not in str(error.value)


def test_duplicate_ids_and_paths_fail(sample_catalog: tuple[Path, dict[str, object]]) -> None:
    manifest, data = sample_catalog
    record = data["sources"][0]
    data["sources"] = [record, dict(record)]
    _write(manifest, data)
    with pytest.raises(CatalogError, match="duplicate source IDs"):
        load_catalog(manifest)
    data["sources"][1]["id"] = "another_source"
    _write(manifest, data)
    with pytest.raises(CatalogError, match="duplicate documents"):
        load_catalog(manifest)


def test_malformed_missing_and_duplicate_json_keys_fail(
    sample_catalog: tuple[Path, dict[str, object]],
) -> None:
    manifest, data = sample_catalog
    with pytest.raises(CatalogError, match="unavailable"):
        load_catalog(manifest)
    manifest.write_text("{not JSON}", encoding="utf-8")
    with pytest.raises(CatalogError, match="JSON is invalid"):
        load_catalog(manifest)
    _write(manifest, data)
    manifest.write_text(
        manifest.read_text("utf-8").replace(
            '"schema_version": 1', '"schema_version": 1, "schema_version": 1'
        ),
        encoding="utf-8",
    )
    with pytest.raises(CatalogError, match="duplicate JSON keys"):
        load_catalog(manifest)


def test_unreadable_and_empty_documents_fail_safely(
    sample_catalog: tuple[Path, dict[str, object]],
) -> None:
    manifest, data = sample_catalog
    _write(manifest, data)
    (manifest.parent / "sources" / "sample.md").write_text(" ", encoding="utf-8")
    with pytest.raises(CatalogError, match="unreadable document"):
        load_catalog(manifest)
