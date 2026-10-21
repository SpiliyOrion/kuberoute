"""Bounded in-memory exact cosine search for Phase 9A chunks."""

import asyncio
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from kuberoute_gateway.rag.embeddings import (
    MAX_BATCH_SIZE,
    MAX_TEXT_CHARS,
    EmbeddingInputError,
    EmbeddingProvider,
    EmbeddingServiceError,
    normalize_vector,
    validate_model_id,
    validate_texts,
)
from kuberoute_gateway.rag.ingestion import Chunk, SourceMetadata

MAX_INDEX_CHUNKS = 10_000
MAX_TOP_K = 100


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    source_id: str
    text: str = field(repr=False)
    score: float
    index: int
    metadata: SourceMetadata


@dataclass(frozen=True)
class _Entry:
    chunk: Chunk
    vector: tuple[float, ...] = field(repr=False)


class InMemoryVectorIndex:
    """One model per index; additions are atomic across embedding batches."""

    def __init__(self, model_id: str) -> None:
        validate_model_id(model_id)
        self.model_id = model_id
        self._entries: dict[str, _Entry] = {}
        self._dimensions: int | None = None
        self._add_lock = asyncio.Lock()

    def __len__(self) -> int:
        return len(self._entries)

    def _check_provider(self, provider: EmbeddingProvider) -> None:
        if provider.model_id != self.model_id:
            raise EmbeddingInputError("Embedding model does not match the index")

    async def add_chunks(
        self,
        chunks: Sequence[Chunk],
        provider: EmbeddingProvider,
        *,
        batch_size: int = 16,
    ) -> None:
        async with self._add_lock:
            await self._add_chunks_unlocked(chunks, provider, batch_size=batch_size)

    async def _add_chunks_unlocked(
        self,
        chunks: Sequence[Chunk],
        provider: EmbeddingProvider,
        *,
        batch_size: int,
    ) -> None:
        self._check_provider(provider)
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or not 1 <= batch_size <= MAX_BATCH_SIZE
        ):
            raise EmbeddingInputError("Embedding batch size is invalid")
        if not isinstance(chunks, Sequence) or not 1 <= len(chunks) <= MAX_INDEX_CHUNKS:
            raise EmbeddingInputError("Chunk count is invalid")
        if len(self) + len(chunks) > MAX_INDEX_CHUNKS:
            raise EmbeddingInputError("Vector index capacity exceeded")
        seen: set[str] = set()
        for chunk in chunks:
            if (
                not isinstance(chunk, Chunk)
                or not isinstance(chunk.chunk_id, str)
                or not 1 <= len(chunk.chunk_id) <= 128
                or not isinstance(chunk.source_id, str)
                or not 1 <= len(chunk.source_id) <= 128
                or isinstance(chunk.index, bool)
                or not isinstance(chunk.index, int)
                or chunk.index < 0
                or not isinstance(chunk.metadata, SourceMetadata)
                or not isinstance(chunk.text, str)
                or not chunk.text.strip()
                or len(chunk.text) > MAX_TEXT_CHARS
            ):
                raise EmbeddingInputError("Chunk input is invalid")
            if chunk.chunk_id in seen or chunk.chunk_id in self._entries:
                raise EmbeddingInputError("Duplicate chunk ID")
            seen.add(chunk.chunk_id)

        staged: dict[str, _Entry] = {}
        dimensions = self._dimensions
        for offset in range(0, len(chunks), batch_size):
            batch = chunks[offset : offset + batch_size]
            texts = [chunk.text for chunk in batch]
            try:
                vectors = await provider.embed(texts)
            except Exception:
                raise EmbeddingServiceError("Embedding provider failed") from None
            if not isinstance(vectors, (list, tuple)) or len(vectors) != len(batch):
                raise EmbeddingInputError("Embedding result count is invalid")
            for chunk, vector in zip(batch, vectors, strict=True):
                normalized = normalize_vector(vector)
                if dimensions is None:
                    dimensions = len(normalized)
                elif len(normalized) != dimensions:
                    raise EmbeddingInputError("Embedding dimensions do not match the index")
                staged[chunk.chunk_id] = _Entry(chunk, normalized)
        self._entries.update(staged)
        self._dimensions = dimensions

    async def search(
        self, query: str, provider: EmbeddingProvider, *, top_k: int = 5
    ) -> tuple[SearchResult, ...]:
        self._check_provider(provider)
        validate_texts([query])
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= MAX_TOP_K:
            raise EmbeddingInputError("Search top_k is invalid")
        if not self._entries:
            return ()
        try:
            vectors = await provider.embed([query])
        except Exception:
            raise EmbeddingServiceError("Embedding provider failed") from None
        if not isinstance(vectors, (list, tuple)) or len(vectors) != 1:
            raise EmbeddingInputError("Query embedding result is invalid")
        vector = normalize_vector(vectors[0])
        if len(vector) != self._dimensions:
            raise EmbeddingInputError("Query embedding dimensions do not match the index")

        ranked: list[SearchResult] = []
        for entry in self._entries.values():
            score = math.fsum(a * b for a, b in zip(vector, entry.vector, strict=True))
            chunk = entry.chunk
            ranked.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    source_id=chunk.source_id,
                    text=chunk.text,
                    score=max(-1.0, min(1.0, score)),
                    index=chunk.index,
                    metadata=chunk.metadata,
                )
            )
        ranked.sort(
            key=lambda result: (-result.score, result.source_id, result.index, result.chunk_id)
        )
        return tuple(ranked[:top_k])
