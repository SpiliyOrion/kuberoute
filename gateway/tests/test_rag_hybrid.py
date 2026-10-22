"""Hybrid retrieval tests use deterministic fake embeddings and synthetic files."""

import asyncio
import math
from pathlib import Path

import pytest
from kuberoute_gateway.rag import (
    Chunk,
    EmbeddingInputError,
    EmbeddingServiceError,
    HybridRetriever,
    HybridSearchConfig,
    RetrievalConsistencyError,
    RetrievalInputError,
    SourceMetadata,
    ingest_paths,
)


class FakeEmbeddingProvider:
    def __init__(self, vectors: dict[str, tuple[float, ...]], model_id: str = "test-embed"):
        self.vectors = vectors
        self.model_id = model_id
        self.calls: list[tuple[str, ...]] = []

    async def embed(self, texts: list[str]) -> tuple[tuple[float, ...], ...]:
        self.calls.append(tuple(texts))
        return tuple(self.vectors[text] for text in texts)


def _chunk(source: str, text: str, index: int = 0) -> Chunk:
    return Chunk(source, f"{source}-{index}", index, text, SourceMetadata(".md", 40))


def _example() -> tuple[HybridRetriever, FakeEmbeddingProvider, list[Chunk]]:
    chunks = [
        _chunk("http", "HTTP 504 means gateway timeout"),
        _chunk("metrics", "Prometheus tracks inference timeout failures"),
        _chunk("routing", "adaptive model routing selects a backend"),
    ]
    provider = FakeEmbeddingProvider(
        {
            chunks[0].text: (0.0, 1.0),
            chunks[1].text: (1.0, 0.0),
            chunks[2].text: (-1.0, 0.0),
            "HTTP 504": (1.0, 0.0),
        }
    )
    return HybridRetriever(provider.model_id), provider, chunks


def test_hybrid_rrf_deduplicates_and_keeps_scores_and_metadata() -> None:
    async def run() -> None:
        retriever, provider, chunks = _example()
        await retriever.add_chunks(chunks, provider)
        results = await retriever.search("HTTP 504", provider, top_k=3)
        assert [result.chunk_id for result in results] == ["http-0", "metrics-0", "routing-0"]
        assert len({result.chunk_id for result in results}) == 3
        first = results[0]
        assert first.fused_score == pytest.approx(1 / (60 + 2) + 1 / (60 + 1))
        assert first.semantic_rank == 2
        assert first.bm25_rank == 1
        assert first.semantic_score == pytest.approx(0)
        assert first.bm25_score is not None and first.bm25_score > 0
        assert first.text == chunks[0].text
        assert first.source_id == "http"
        assert first.index == 0
        assert first.metadata == chunks[0].metadata
        assert "gateway timeout" not in repr(first)
        assert results[1].bm25_score is None
        assert results[1].bm25_rank is None
        assert len(retriever) == 3

    asyncio.run(run())


def test_semantic_only_lexical_only_and_weighted_hybrid() -> None:
    async def run() -> None:
        retriever, provider, chunks = _example()
        await retriever.add_chunks(chunks, provider)
        before = len(provider.calls)
        lexical = await retriever.search(
            "HTTP 504", config=HybridSearchConfig(mode="lexical"), top_k=1
        )
        assert len(provider.calls) == before
        assert lexical[0].chunk_id == "http-0"
        assert lexical[0].semantic_score is None
        semantic = await retriever.search(
            "HTTP 504", provider, config=HybridSearchConfig(mode="semantic"), top_k=1
        )
        assert semantic[0].chunk_id == "metrics-0"
        assert semantic[0].bm25_score is None
        weighted = await retriever.search(
            "HTTP 504",
            provider,
            config=HybridSearchConfig(semantic_weight=8.0, lexical_weight=0.5, rrf_k=10),
            top_k=2,
        )
        assert weighted[0].chunk_id == "metrics-0"
        assert weighted[0].fused_score == pytest.approx(8 / 11)
        assert weighted[1].fused_score == pytest.approx(8 / 12 + 0.5 / 11)
        semantic_via_zero_weight = await retriever.search(
            "HTTP 504",
            provider,
            config=HybridSearchConfig(lexical_weight=0),
            top_k=1,
        )
        assert semantic_via_zero_weight[0].chunk_id == "metrics-0"

    asyncio.run(run())


def test_hybrid_can_include_lexical_only_candidates() -> None:
    async def run() -> None:
        retriever, provider, chunks = _example()
        provider.vectors["adaptive routing"] = (1.0, 0.0)
        await retriever.add_chunks(chunks, provider)
        results = await retriever.search(
            "adaptive routing",
            provider,
            top_k=2,
            config=HybridSearchConfig(candidate_k=2),
        )
        lexical_only = next(result for result in results if result.chunk_id == "routing-0")
        assert lexical_only.semantic_score is None
        assert lexical_only.bm25_score is not None

    asyncio.run(run())


def test_synthetic_technical_collection_shows_complementary_retrieval(tmp_path: Path) -> None:
    documents = {
        "kubernetes.md": "Kubernetes deployments use readiness probes for healthy Pods.",
        "timeouts.md": "Inference requests can exceed deadlines and return timeout failures.",
        "prometheus.md": "Prometheus metrics track failed requests and monitoring signals.",
        "http.md": "HTTP 504 is a gateway timeout response.",
        "embeddings.md": "Embedding models map document passages into vectors.",
        "routing.md": "Adaptive model routing selects which LLM receives a request.",
    }
    paths = []
    for name, text in documents.items():
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    chunks = ingest_paths(paths)
    vectors = {chunk.text: (0.0, 0.0, 0.0, 1.0) for chunk in chunks}
    vectors[documents["timeouts.md"]] = (1.0, 0.0, 0.0, 0.0)
    vectors[documents["prometheus.md"]] = (0.0, 1.0, 0.0, 0.0)
    vectors[documents["routing.md"]] = (0.0, 0.0, 1.0, 0.0)
    vectors.update(
        {
            "How can I tell whether my inference service is experiencing too many timeouts?": (
                0.0,
                1.0,
                0.0,
                0.0,
            ),
            "What does HTTP 504 mean in our gateway?": (1.0, 0.0, 0.0, 0.0),
            "How do I monitor which LLM receives a request?": (0.0, 0.0, 1.0, 0.0),
        }
    )

    async def run() -> None:
        provider = FakeEmbeddingProvider(vectors)
        retriever = HybridRetriever(provider.model_id)
        await retriever.add_chunks(chunks, provider)
        question_a = (
            "How can I tell whether my inference service is experiencing too many timeouts?"
        )
        semantic_a = await retriever.search(
            question_a, provider, top_k=1, config=HybridSearchConfig(mode="semantic")
        )
        lexical_a = await retriever.search(
            question_a, top_k=3, config=HybridSearchConfig(mode="lexical")
        )
        assert semantic_a[0].text == documents["prometheus.md"]
        assert documents["timeouts.md"] in [item.text for item in lexical_a]
        question_b = "What does HTTP 504 mean in our gateway?"
        assert (
            await retriever.search(question_b, top_k=1, config=HybridSearchConfig(mode="lexical"))
        )[0].text == documents["http.md"]
        question_c = "How do I monitor which LLM receives a request?"
        assert (
            await retriever.search(question_c, top_k=2, config=HybridSearchConfig(mode="lexical"))
        )[0].text == documents["routing.md"]
        assert await retriever.search(question_c, provider, top_k=2)

    asyncio.run(run())


def test_failed_second_embedding_batch_leaves_both_indexes_empty() -> None:
    async def run() -> None:
        retriever, provider, chunks = _example()

        class FailingProvider(FakeEmbeddingProvider):
            async def embed(self, texts: list[str]) -> tuple[tuple[float, ...], ...]:
                if len(self.calls) == 1:
                    raise RuntimeError("private document content")
                return await super().embed(texts)

        failing = FailingProvider(provider.vectors)
        with pytest.raises(EmbeddingServiceError) as error:
            await retriever.add_chunks(chunks[:2], failing, batch_size=1)
        assert "private" not in str(error.value)
        assert len(retriever) == 0
        assert retriever._semantic.chunk_ids() == retriever._lexical.chunk_ids() == frozenset()
        await retriever.add_chunks(chunks[:2], provider)
        assert len(retriever) == 2

    asyncio.run(run())


def test_failed_query_embedding_is_sanitized_and_lexical_search_still_works() -> None:
    async def run() -> None:
        retriever, provider, chunks = _example()
        await retriever.add_chunks(chunks, provider)

        class FailingQueryProvider(FakeEmbeddingProvider):
            async def embed(self, texts: list[str]) -> tuple[tuple[float, ...], ...]:
                raise RuntimeError("private query and upstream details")

        failing = FailingQueryProvider(provider.vectors)
        with pytest.raises(EmbeddingServiceError) as error:
            await retriever.search("HTTP 504", failing)
        assert "private" not in str(error.value)
        assert (await retriever.search("HTTP 504", config=HybridSearchConfig(mode="lexical")))[
            0
        ].chunk_id == "http-0"

    asyncio.run(run())


def test_concurrent_duplicate_additions_preserve_consistency() -> None:
    async def run() -> None:
        retriever, provider, chunks = _example()
        started = asyncio.Event()
        release = asyncio.Event()

        class SlowProvider(FakeEmbeddingProvider):
            async def embed(self, texts: list[str]) -> tuple[tuple[float, ...], ...]:
                started.set()
                await release.wait()
                return await super().embed(texts)

        slow = SlowProvider(provider.vectors)
        first = asyncio.create_task(retriever.add_chunks(chunks[:1], slow))
        await started.wait()
        second = asyncio.create_task(retriever.add_chunks(chunks[:1], slow))
        release.set()
        outcomes = await asyncio.gather(first, second, return_exceptions=True)
        assert outcomes[0] is None
        assert isinstance(outcomes[1], RetrievalInputError)
        assert len(retriever) == 1
        assert retriever._semantic.chunk_ids() == retriever._lexical.chunk_ids()

    asyncio.run(run())


def test_mismatched_provider_dimensions_and_inconsistent_indexes_fail_safely() -> None:
    async def run() -> None:
        retriever, provider, chunks = _example()
        wrong = FakeEmbeddingProvider(provider.vectors, "different-model")
        with pytest.raises(EmbeddingInputError, match="model"):
            await retriever.add_chunks(chunks, wrong)
        assert len(retriever) == 0
        await retriever.add_chunks(chunks[:1], provider)
        with pytest.raises(EmbeddingInputError, match="model"):
            await retriever.search("HTTP 504", wrong)
        provider.vectors["HTTP 504"] = (1.0, 0.0, 0.0)
        with pytest.raises(EmbeddingInputError, match="dimensions"):
            await retriever.search("HTTP 504", provider)
        retriever._semantic._entries.clear()
        with pytest.raises(RetrievalConsistencyError, match="different chunk IDs"):
            await retriever.search("HTTP 504", config=HybridSearchConfig(mode="lexical"))

    asyncio.run(run())


def test_empty_index_duplicate_text_and_privacy_safe_errors() -> None:
    async def run() -> None:
        retriever, provider, _ = _example()
        assert await retriever.search("HTTP 504", provider) == ()
        same_text = [_chunk("a", "private material"), _chunk("b", "private material")]
        provider.vectors["private material"] = (1.0, 0.0)
        provider.vectors["private query"] = (1.0, 0.0)
        await retriever.add_chunks(same_text, provider)
        results = await retriever.search("private query", provider)
        assert [item.chunk_id for item in results] == ["a-0", "b-0"]
        assert all(item.semantic_rank is not None for item in results)
        assert "private material" not in repr(results[0])
        with pytest.raises(RetrievalInputError) as error:
            await retriever.search("private\ud800query", provider)
        assert "private" not in str(error.value)

    asyncio.run(run())


@pytest.mark.parametrize(
    "settings",
    [
        {"mode": "wrong"},
        {"semantic_weight": -1},
        {"lexical_weight": math.nan},
        {"semantic_weight": math.inf},
        {"semantic_weight": 0, "lexical_weight": 0},
        {"mode": "semantic", "semantic_weight": 0},
        {"mode": "lexical", "lexical_weight": 0},
        {"rrf_k": 0},
        {"rrf_k": True},
        {"candidate_k": 101},
        {"candidate_k": True},
    ],
)
def test_invalid_hybrid_settings(settings: dict[str, object]) -> None:
    with pytest.raises(RetrievalInputError):
        HybridSearchConfig(**settings)


def test_invalid_top_k_and_missing_provider() -> None:
    async def run() -> None:
        retriever, provider, chunks = _example()
        await retriever.add_chunks(chunks, provider)
        with pytest.raises(RetrievalInputError, match="provider"):
            await retriever.search("HTTP 504")
        for top_k in (0, 101, True):
            with pytest.raises(RetrievalInputError):
                await retriever.search("HTTP 504", provider, top_k=top_k)

    asyncio.run(run())
