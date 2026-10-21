"""Deterministic reciprocal-rank fusion over shared dense and BM25 indexes."""

import asyncio
import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

from kuberoute_gateway.rag.bm25 import (
    BM25Index,
    RetrievalConsistencyError,
    RetrievalInputError,
)
from kuberoute_gateway.rag.embeddings import (
    EmbeddingInputError,
    EmbeddingProvider,
    validate_model_id,
    validate_texts,
)
from kuberoute_gateway.rag.ingestion import Chunk, SourceMetadata
from kuberoute_gateway.rag.vector_index import MAX_TOP_K, InMemoryVectorIndex

RetrievalMode = Literal["hybrid", "semantic", "lexical"]


@dataclass(frozen=True)
class HybridSearchConfig:
    mode: RetrievalMode = "hybrid"
    semantic_weight: float = 1.0
    lexical_weight: float = 1.0
    rrf_k: int = 60
    candidate_k: int = 20

    def __post_init__(self) -> None:
        if self.mode not in ("hybrid", "semantic", "lexical"):
            raise RetrievalInputError("Retrieval mode is invalid")
        for weight in (self.semantic_weight, self.lexical_weight):
            if (
                isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not math.isfinite(weight)
                or not 0 <= weight <= 10
            ):
                raise RetrievalInputError("Retrieval weight must be finite and between 0 and 10")
        if self.mode == "semantic" and self.semantic_weight == 0:
            raise RetrievalInputError("Semantic retrieval weight must be positive")
        if self.mode == "lexical" and self.lexical_weight == 0:
            raise RetrievalInputError("Lexical retrieval weight must be positive")
        if self.mode == "hybrid" and self.semantic_weight == self.lexical_weight == 0:
            raise RetrievalInputError("At least one retrieval weight must be positive")
        if (
            isinstance(self.rrf_k, bool)
            or not isinstance(self.rrf_k, int)
            or not 1 <= self.rrf_k <= 1000
        ):
            raise RetrievalInputError("RRF rank constant is invalid")
        if (
            isinstance(self.candidate_k, bool)
            or not isinstance(self.candidate_k, int)
            or not 1 <= self.candidate_k <= MAX_TOP_K
        ):
            raise RetrievalInputError("Retrieval candidate count is invalid")


@dataclass(frozen=True)
class HybridResult:
    chunk_id: str
    source_id: str
    index: int
    text: str = field(repr=False)
    metadata: SourceMetadata
    fused_score: float
    semantic_score: float | None = None
    semantic_rank: int | None = None
    bm25_score: float | None = None
    bm25_rank: int | None = None


class HybridRetriever:
    """Own both indexes and serialize indexing with search to hide partial updates."""

    def __init__(self, embedding_model_id: str, *, k1: float = 1.2, b: float = 0.75) -> None:
        validate_model_id(embedding_model_id)
        self.model_id = embedding_model_id
        self._semantic = InMemoryVectorIndex(embedding_model_id)
        self._lexical = BM25Index(k1=k1, b=b)
        self._lock = asyncio.Lock()

    def __len__(self) -> int:
        return len(self._lexical)

    def _check_consistency(self) -> None:
        if self._semantic.chunk_ids() != self._lexical.chunk_ids():
            raise RetrievalConsistencyError("Retrieval indexes contain different chunk IDs")

    async def add_chunks(
        self,
        chunks: Sequence[Chunk],
        provider: EmbeddingProvider,
        *,
        batch_size: int = 16,
    ) -> None:
        async with self._lock:
            self._check_consistency()
            prepared = self._lexical._prepare_add(chunks)
            await self._semantic.add_chunks(chunks, provider, batch_size=batch_size)
            self._lexical._commit_prepared(prepared)
            self._check_consistency()

    async def search(
        self,
        query: str,
        provider: EmbeddingProvider | None = None,
        *,
        top_k: int = 5,
        config: HybridSearchConfig | None = None,
    ) -> tuple[HybridResult, ...]:
        if config is None:
            config = HybridSearchConfig()
        if not isinstance(config, HybridSearchConfig):
            raise RetrievalInputError("Retrieval settings are invalid")
        try:
            validate_texts([query])
        except EmbeddingInputError:
            raise RetrievalInputError("Retrieval query is invalid") from None
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= MAX_TOP_K:
            raise RetrievalInputError("Search top_k is invalid")

        semantic_enabled = config.mode != "lexical" and config.semantic_weight > 0
        lexical_enabled = config.mode != "semantic" and config.lexical_weight > 0
        if semantic_enabled and provider is None:
            raise RetrievalInputError("Semantic retrieval requires an embedding provider")
        if semantic_enabled and provider.model_id != self.model_id:
            raise EmbeddingInputError("Embedding model does not match the index")

        async with self._lock:
            self._check_consistency()
            if len(self) == 0:
                return ()
            pool = max(top_k, config.candidate_k)
            semantic = (
                await self._semantic.search(query, provider, top_k=pool)
                if semantic_enabled and provider is not None
                else ()
            )
            lexical = self._lexical.search(query, top_k=pool) if lexical_enabled else ()
            fused: dict[str, HybridResult] = {}
            for rank, result in enumerate(semantic, start=1):
                fused[result.chunk_id] = HybridResult(
                    chunk_id=result.chunk_id,
                    source_id=result.source_id,
                    index=result.index,
                    text=result.text,
                    metadata=result.metadata,
                    fused_score=config.semantic_weight / (config.rrf_k + rank),
                    semantic_score=result.score,
                    semantic_rank=rank,
                )
            for rank, result in enumerate(lexical, start=1):
                contribution = config.lexical_weight / (config.rrf_k + rank)
                existing = fused.get(result.chunk_id)
                if existing is None:
                    fused[result.chunk_id] = HybridResult(
                        chunk_id=result.chunk_id,
                        source_id=result.source_id,
                        index=result.index,
                        text=result.text,
                        metadata=result.metadata,
                        fused_score=contribution,
                        bm25_score=result.score,
                        bm25_rank=rank,
                    )
                else:
                    if (
                        existing.source_id != result.source_id
                        or existing.index != result.index
                        or existing.text != result.text
                        or existing.metadata != result.metadata
                    ):
                        raise RetrievalConsistencyError("Retrieval indexes disagree on a chunk")
                    fused[result.chunk_id] = replace(
                        existing,
                        fused_score=existing.fused_score + contribution,
                        bm25_score=result.score,
                        bm25_rank=rank,
                    )
            ranked = sorted(
                fused.values(),
                key=lambda result: (
                    -result.fused_score,
                    result.source_id,
                    result.index,
                    result.chunk_id,
                ),
            )
            return tuple(ranked[:top_k])
