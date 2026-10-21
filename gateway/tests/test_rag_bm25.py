"""Deterministic synthetic tests for the local BM25 index."""

import math

import pytest
from kuberoute_gateway.rag import (
    BM25Index,
    Chunk,
    RetrievalInputError,
    SourceMetadata,
    tokenize,
)


def _chunk(source: str, index: int, text: str, *, chunk_id: str | None = None) -> Chunk:
    return Chunk(source, chunk_id or f"{source}-{index}", index, text, SourceMetadata(".txt", 20))


def test_unicode_and_technical_tokenization_is_deterministic() -> None:
    text = "CAFÉ Straße HTTP 504 qwen3:0.6b v1.36.1 KubeRoute_inference_requests_total"
    expected = (
        "café",
        "strasse",
        "http",
        "504",
        "qwen3:0.6b",
        "v1.36.1",
        "kuberoute_inference_requests_total",
    )
    assert tokenize(text) == expected
    assert tokenize(text) == expected
    assert tokenize("self-healing") == ("self", "healing")


def test_bm25_standard_idf_term_frequency_and_repeated_queries() -> None:
    index = BM25Index()
    index.add_chunks([_chunk("a", 0, "timeout timeout"), _chunk("b", 0, "timeout")])
    results = index.search("TIMEOUT")
    assert [result.chunk_id for result in results] == ["a-0", "b-0"]
    idf = math.log1p((2 - 2 + 0.5) / (2 + 0.5))
    expected = idf * 2 * (1.2 + 1) / (2 + 1.2 * (1 - 0.75 + 0.75 * 2 / 1.5))
    assert results[0].score == pytest.approx(expected)
    assert index.search("timeout timeout") == results  # Repetition cannot game query weight.
    assert index.search("absent") == ()
    assert index.search("!!!") == ()
    assert all(item.score > 0 for item in results)


def test_technical_keyword_matches_and_casefolding() -> None:
    index = BM25Index()
    index.add_chunks(
        [
            _chunk("api", 0, "HTTP 504 means the gateway timed out"),
            _chunk("model", 0, "qwen3:0.6b uses an embedding model"),
            _chunk("deployment", 0, "Kubernetes v1.36.1 deployment"),
        ]
    )
    assert index.search("http 504")[0].source_id == "api"
    assert index.search("QWEN3:0.6B")[0].source_id == "model"
    assert index.search("V1.36.1")[0].source_id == "deployment"


def test_ties_use_source_index_and_chunk_id_and_keep_distinct_texts() -> None:
    index = BM25Index()
    index.add_chunks(
        [
            _chunk("b", 0, "monitor", chunk_id="b-0"),
            _chunk("a", 1, "monitor", chunk_id="a-1"),
            _chunk("a", 0, "monitor", chunk_id="a-0-z"),
            _chunk("a", 0, "monitor", chunk_id="a-0-a"),
        ]
    )
    results = index.search("MONITOR", top_k=4)
    assert [result.chunk_id for result in results] == ["a-0-a", "a-0-z", "a-1", "b-0"]
    assert all(result.text == "monitor" for result in results)
    assert "monitor" not in repr(results[0])
    assert results[0].metadata == SourceMetadata(".txt", 20)
    assert index.chunk_ids() == frozenset(result.chunk_id for result in results)


def test_empty_token_corpus_is_safe() -> None:
    index = BM25Index()
    assert index.search("query") == ()
    index.add_chunks([_chunk("emoji", 0, "☀️")])
    assert index.search("query") == ()
    assert len(index) == 1


@pytest.mark.parametrize("query", ["", "  ", "private\ud800query", "x" * 10_001, 123])
def test_bad_queries_are_sanitized(query: object) -> None:
    index = BM25Index()
    with pytest.raises(RetrievalInputError) as error:
        index.search(query)
    assert "private" not in str(error.value)


@pytest.mark.parametrize("top_k", [0, 101, True, 1.5])
def test_invalid_top_k_is_rejected(top_k: object) -> None:
    with pytest.raises(RetrievalInputError):
        BM25Index().search("query", top_k=top_k)


@pytest.mark.parametrize(
    "k1,b",
    [
        (0, 0.75),
        (-1, 0.75),
        (11, 0.75),
        (math.nan, 0.75),
        (1, -0.1),
        (1, 1.1),
        (1, math.inf),
        (True, 0.5),
        (1, False),
    ],
)
def test_bm25_parameters_are_finite_and_bounded(k1: object, b: object) -> None:
    with pytest.raises(RetrievalInputError):
        BM25Index(k1=k1, b=b)


@pytest.mark.parametrize(
    "metadata",
    [SourceMetadata(".pdf", 20), SourceMetadata(".txt", -1), SourceMetadata(".txt", 10_485_761)],
)
def test_duplicate_and_invalid_chunk_additions_are_atomic_and_private(
    metadata: SourceMetadata,
) -> None:
    index = BM25Index()
    valid = _chunk("a", 0, "private timeout")
    invalid = Chunk("b", "b-0", 0, "private text", metadata)
    with pytest.raises(RetrievalInputError) as error:
        index.add_chunks([valid, invalid])
    assert "private" not in str(error.value)
    assert len(index) == 0
    with pytest.raises(RetrievalInputError, match="Duplicate"):
        index.add_chunks([valid, valid])
    assert len(index) == 0
    index.add_chunks([valid])
    with pytest.raises(RetrievalInputError, match="Duplicate"):
        index.add_chunks([valid])
    assert len(index) == 1


def test_bm25_result_metadata_and_top_k() -> None:
    index = BM25Index(k1=1.5, b=0.5)
    index.add_chunks([_chunk("a", 0, "monitor timeout"), _chunk("b", 2, "monitor metrics")])
    result = index.search("monitor", top_k=1)[0]
    assert result.source_id == "a"
    assert result.index == 0
    assert result.text == "monitor timeout"
    assert result.metadata.extension == ".txt"
    assert math.isfinite(result.score) and result.score > 0
