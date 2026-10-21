"""Synthetic vector-search and mocked Ollama embedding tests."""

import asyncio
import json
import math
from pathlib import Path

import httpx
import pytest
from kuberoute_gateway.rag import (
    Chunk,
    ChunkingConfig,
    EmbeddingInputError,
    EmbeddingServiceError,
    InMemoryVectorIndex,
    OllamaEmbeddingProvider,
    SourceMetadata,
    ingest_paths,
)


class FakeEmbeddingProvider:
    def __init__(self, vectors: dict[str, tuple[float, ...]], model_id: str = "test-embed"):
        self.model_id = model_id
        self.vectors = vectors
        self.calls: list[tuple[str, ...]] = []

    async def embed(self, texts: list[str]) -> tuple[tuple[float, ...], ...]:
        self.calls.append(tuple(texts))
        return tuple(self.vectors[text] for text in texts)


def _chunk(source: str, index: int, text: str) -> Chunk:
    return Chunk(source, f"{source}-{index}", index, text, SourceMetadata(".txt", 10))


def test_indexing_search_ranking_ties_metadata_and_bounded_batches() -> None:
    async def run() -> None:
        chunks = (
            _chunk("b", 0, "north"),
            _chunk("a", 1, "east"),
            _chunk("a", 0, "north again"),
            _chunk("c", 0, "south"),
        )
        provider = FakeEmbeddingProvider(
            {
                "north": (1.0, 0.0),
                "east": (0.0, 1.0),
                "north again": (2.0, 0.0),
                "south": (-1.0, 0.0),
                "question": (1.0, 0.0),
            }
        )
        index = InMemoryVectorIndex("test-embed")
        await index.add_chunks(chunks, provider, batch_size=2)
        assert provider.calls[:2] == [("north", "east"), ("north again", "south")]
        results = await index.search("question", provider, top_k=3)
        assert [(item.source_id, item.index) for item in results] == [("a", 0), ("b", 0), ("a", 1)]
        assert [item.score for item in results] == [1.0, 1.0, 0.0]
        assert results[0].chunk_id == "a-0"
        assert results[0].text == "north again"
        assert results[0].metadata == SourceMetadata(".txt", 10)
        assert "north again" not in repr(results[0])
        assert len(index) == 4
        assert await index.search("question", provider, top_k=1) == results[:1]

    asyncio.run(run())


def test_ingestion_to_embedding_to_search_uses_real_chunk_contract(tmp_path: Path) -> None:
    async def run() -> None:
        source = tmp_path / "synthetic.md"
        source.write_text("alpha beta gamma", encoding="utf-8")
        chunks = ingest_paths([source], config=ChunkingConfig(size=10, overlap=2))
        provider = FakeEmbeddingProvider(
            {chunk.text: (1.0, 2.0) for chunk in chunks} | {"q": (1.0, 2.0)}
        )
        index = InMemoryVectorIndex(provider.model_id)
        await index.add_chunks(chunks, provider)
        results = await index.search("q", provider)
        assert len(results) == len(chunks)
        assert [result.chunk_id for result in results] == [chunk.chunk_id for chunk in chunks]

    asyncio.run(run())


def test_duplicate_ids_and_dimension_mismatch_do_not_partially_mutate_index() -> None:
    async def run() -> None:
        index = InMemoryVectorIndex("test-embed")
        provider = FakeEmbeddingProvider({"one": (1.0, 0.0), "two": (1.0, 0.0, 0.0)})
        with pytest.raises(EmbeddingInputError, match="dimensions"):
            await index.add_chunks(
                [_chunk("a", 0, "one"), _chunk("b", 0, "two")], provider, batch_size=1
            )
        assert len(index) == 0
        await index.add_chunks([_chunk("a", 0, "one")], provider)
        with pytest.raises(EmbeddingInputError, match="Duplicate"):
            await index.add_chunks([_chunk("a", 0, "one")], provider)
        assert len(index) == 1
        with pytest.raises(EmbeddingInputError, match="dimensions"):
            await index.search("two", provider)

    asyncio.run(run())


def test_concurrent_additions_reject_duplicate_ids() -> None:
    async def run() -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        class SlowProvider(FakeEmbeddingProvider):
            async def embed(self, texts: list[str]) -> tuple[tuple[float, ...], ...]:
                started.set()
                await release.wait()
                return await super().embed(texts)

        provider = SlowProvider({"one": (1.0, 0.0)})
        index = InMemoryVectorIndex(provider.model_id)
        chunks = [_chunk("a", 0, "one")]
        first = asyncio.create_task(index.add_chunks(chunks, provider))
        await started.wait()
        second = asyncio.create_task(index.add_chunks(chunks, provider))
        release.set()
        outcomes = await asyncio.gather(first, second, return_exceptions=True)
        assert outcomes[0] is None
        assert isinstance(outcomes[1], EmbeddingInputError)
        assert len(index) == 1
        assert provider.calls == [("one",)]
        assert "1.0" not in repr(next(iter(index._entries.values())))

    asyncio.run(run())


def test_model_mismatch_is_rejected_before_embedding() -> None:
    async def run() -> None:
        index = InMemoryVectorIndex("model-a")
        provider = FakeEmbeddingProvider({"one": (1.0,), "q": (1.0,)}, "model-b")
        with pytest.raises(EmbeddingInputError, match="model"):
            await index.add_chunks([_chunk("a", 0, "one")], provider)
        with pytest.raises(EmbeddingInputError, match="model"):
            await index.search("q", provider)
        assert provider.calls == []

    asyncio.run(run())


@pytest.mark.parametrize("vector", [(), (0.0, 0.0), (math.nan,), (math.inf,), (True,), ("1",)])
def test_invalid_vectors_are_rejected_without_values_in_errors(vector: tuple[object, ...]) -> None:
    async def run() -> None:
        index = InMemoryVectorIndex("test-embed")
        provider = FakeEmbeddingProvider({"one": vector})
        with pytest.raises(EmbeddingInputError) as error:
            await index.add_chunks([_chunk("a", 0, "one")], provider)
        assert "one" not in str(error.value)
        assert len(index) == 0

    asyncio.run(run())


def test_invalid_batch_query_and_top_k_are_rejected() -> None:
    async def run() -> None:
        index = InMemoryVectorIndex("test-embed")
        provider = FakeEmbeddingProvider({"one": (1.0,), "q": (1.0,)})
        for batch_size in (0, 33, True):
            with pytest.raises(EmbeddingInputError):
                await index.add_chunks([_chunk("a", 0, "one")], provider, batch_size=batch_size)
        with pytest.raises(EmbeddingInputError):
            await index.add_chunks([], provider)
        await index.add_chunks([_chunk("a", 0, "one")], provider)
        for query in ("", " ", "x" * 10_001, "private\ud800query"):
            with pytest.raises(EmbeddingInputError):
                await index.search(query, provider)
        for top_k in (0, 101, True):
            with pytest.raises(EmbeddingInputError):
                await index.search("q", provider, top_k=top_k)

    asyncio.run(run())


def test_ollama_adapter_uses_batched_api_and_preserves_injected_client() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://localhost:11434/api/embed"
        assert json.loads(request.content) == {
            "model": "nomic-embed-text",
            "input": ["first", "second"],
            "truncate": False,
        }
        assert request.extensions["timeout"] == {
            "connect": 3.0,
            "read": 60.0,
            "write": 10.0,
            "pool": 3.0,
        }
        return httpx.Response(
            200, json={"model": "nomic-embed-text", "embeddings": [[3, 4], [0, 5]]}
        )

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            async with OllamaEmbeddingProvider(
                "http://localhost:11434", "nomic-embed-text", client=client
            ) as provider:
                vectors = await provider.embed(["first", "second"])
                assert vectors == ((0.6, 0.8), (0.0, 1.0))
            assert not client.is_closed

    asyncio.run(run())


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="secret invalid JSON"),
        httpx.Response(200, json={"embeddings": []}),
        httpx.Response(200, json={"embeddings": [[0, 0]]}),
        httpx.Response(200, text='{"embeddings": [[NaN, 1]]}'),
        httpx.Response(200, json={"embeddings": [[1, 2], [3, 4]]}),
        httpx.Response(200, json={"embeddings": [[1, 2]], "model": "wrong-model"}),
        httpx.Response(200, json={"embeddings": "secret"}),
        httpx.Response(503, text="secret upstream error"),
    ],
)
def test_ollama_invalid_responses_are_sanitized(response: httpx.Response) -> None:
    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)) as client:
            provider = OllamaEmbeddingProvider(
                "http://localhost:11434", "test-embed", client=client
            )
            with pytest.raises(EmbeddingServiceError) as error:
                await provider.embed(["secret input"])
            assert "secret" not in str(error.value)
            assert "localhost" not in str(error.value)

    asyncio.run(run())


@pytest.mark.parametrize(
    "failure", [httpx.ConnectError("secret URL"), httpx.ReadTimeout("secret URL")]
)
def test_ollama_network_errors_are_sanitized(failure: Exception) -> None:
    def fail(_: httpx.Request) -> httpx.Response:
        raise failure

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as client:
            provider = OllamaEmbeddingProvider(
                "http://localhost:11434", "test-embed", client=client
            )
            with pytest.raises(EmbeddingServiceError) as error:
                await provider.embed(["private query"])
            assert "secret" not in str(error.value)
            assert "private" not in str(error.value)

    asyncio.run(run())


def test_ollama_input_validation_and_owned_client_lifecycle() -> None:
    async def run() -> None:
        provider = OllamaEmbeddingProvider("http://localhost:11434", "test-embed")
        with pytest.raises(EmbeddingServiceError, match="not started"):
            await provider.embed(["text"])
        await provider.start()
        client = provider._client
        assert client is not None and not client.is_closed
        await provider.close()
        await provider.close()
        assert client.is_closed
        for texts in ([], [""], ["x" * 10_001], ["x"] * 33):
            with pytest.raises(EmbeddingInputError):
                await provider.embed(texts)

    asyncio.run(run())
    for url in ("not-a-url", "http://secret@example.com", "http://localhost?q=secret"):
        with pytest.raises(EmbeddingInputError) as error:
            OllamaEmbeddingProvider(url, "test-embed")
        assert "secret" not in str(error.value)
    for model_id in ("", "private key", "x" * 129):
        with pytest.raises(EmbeddingInputError):
            InMemoryVectorIndex(model_id)
