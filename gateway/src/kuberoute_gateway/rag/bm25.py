"""Deterministic in-memory Okapi BM25 over Phase 9A chunks."""

import math
import re
import threading
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from kuberoute_gateway.rag.embeddings import EmbeddingInputError, validate_texts
from kuberoute_gateway.rag.ingestion import MAX_FILE_BYTES, Chunk, SourceMetadata
from kuberoute_gateway.rag.vector_index import MAX_INDEX_CHUNKS, MAX_TOP_K

_TOKEN = re.compile(r"\w+(?:[.:]\w+)*", re.UNICODE)


class RetrievalInputError(ValueError):
    """Invalid retrieval input, without query text or document details."""


class RetrievalConsistencyError(RuntimeError):
    """The dense and lexical indexes do not contain the same chunks."""


def _validate_text(text: str) -> None:
    try:
        validate_texts([text])
    except EmbeddingInputError:
        raise RetrievalInputError("Retrieval text is invalid") from None


def tokenize(text: str) -> tuple[str, ...]:
    """Case-folded Unicode words; retain dots/colons within versions and identifiers."""
    _validate_text(text)
    return tuple(match.group() for match in _TOKEN.finditer(text.casefold()))


@dataclass(frozen=True)
class BM25Result:
    chunk_id: str
    source_id: str
    text: str = field(repr=False)
    score: float
    index: int
    metadata: SourceMetadata


@dataclass(frozen=True)
class _LexicalDocument:
    chunk: Chunk
    terms: Counter[str] = field(repr=False)
    length: int


class BM25Index:
    """Incremental BM25 with precomputed postings and safe, atomic validation."""

    def __init__(self, *, k1: float = 1.2, b: float = 0.75) -> None:
        if (
            isinstance(k1, bool)
            or not isinstance(k1, (int, float))
            or not math.isfinite(k1)
            or not 0 < k1 <= 10
        ):
            raise RetrievalInputError("BM25 k1 must be finite and between 0 and 10")
        if (
            isinstance(b, bool)
            or not isinstance(b, (int, float))
            or not math.isfinite(b)
            or not 0 <= b <= 1
        ):
            raise RetrievalInputError("BM25 b must be finite and between 0 and 1")
        self.k1 = float(k1)
        self.b = float(b)
        self._documents: dict[str, _LexicalDocument] = {}
        self._postings: dict[str, dict[str, int]] = defaultdict(dict)
        self._document_frequency: Counter[str] = Counter()
        self._total_length = 0
        self._lock = threading.RLock()

    def __len__(self) -> int:
        return len(self._documents)

    def chunk_ids(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._documents)

    def _prepare_add(self, chunks: Sequence[Chunk]) -> tuple[_LexicalDocument, ...]:
        if isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence):
            raise RetrievalInputError("Chunks must be an explicit sequence")
        if not 1 <= len(chunks) <= MAX_INDEX_CHUNKS or len(self) + len(chunks) > MAX_INDEX_CHUNKS:
            raise RetrievalInputError("BM25 index capacity exceeded")
        prepared: list[_LexicalDocument] = []
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
                or chunk.metadata.extension not in (".txt", ".md")
                or isinstance(chunk.metadata.byte_size, bool)
                or not isinstance(chunk.metadata.byte_size, int)
                or not 0 <= chunk.metadata.byte_size <= MAX_FILE_BYTES
            ):
                raise RetrievalInputError("Chunk metadata is invalid")
            if chunk.chunk_id in seen or chunk.chunk_id in self._documents:
                raise RetrievalInputError("Duplicate chunk ID")
            seen.add(chunk.chunk_id)
            terms = Counter(tokenize(chunk.text))
            prepared.append(_LexicalDocument(chunk, terms, terms.total()))
        return tuple(prepared)

    def _commit_prepared(self, prepared: tuple[_LexicalDocument, ...]) -> None:
        """Publish already-validated additions without a network or async boundary."""
        with self._lock:
            for item in prepared:
                chunk_id = item.chunk.chunk_id
                self._documents[chunk_id] = item
                self._total_length += item.length
                for term, frequency in item.terms.items():
                    self._postings[term][chunk_id] = frequency
                    self._document_frequency[term] += 1

    def add_chunks(self, chunks: Sequence[Chunk]) -> None:
        with self._lock:
            self._commit_prepared(self._prepare_add(chunks))

    def search(self, query: str, *, top_k: int = 5) -> tuple[BM25Result, ...]:
        terms = tuple(dict.fromkeys(tokenize(query)))
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= MAX_TOP_K:
            raise RetrievalInputError("Search top_k is invalid")
        with self._lock:
            count = len(self._documents)
            if not terms or count == 0 or self._total_length == 0:
                return ()
            average_length = self._total_length / count
            scores: dict[str, float] = defaultdict(float)
            for term in terms:
                postings = self._postings.get(term)
                if not postings:
                    continue
                frequency = self._document_frequency[term]
                idf = math.log1p((count - frequency + 0.5) / (frequency + 0.5))
                for chunk_id, tf in postings.items():
                    length = self._documents[chunk_id].length
                    denominator = tf + self.k1 * (1 - self.b + self.b * length / average_length)
                    scores[chunk_id] += idf * tf * (self.k1 + 1) / denominator
            ranked = [
                BM25Result(
                    chunk_id=chunk_id,
                    source_id=self._documents[chunk_id].chunk.source_id,
                    text=self._documents[chunk_id].chunk.text,
                    score=score,
                    index=self._documents[chunk_id].chunk.index,
                    metadata=self._documents[chunk_id].chunk.metadata,
                )
                for chunk_id, score in scores.items()
                if score > 0 and math.isfinite(score)
            ]
            ranked.sort(
                key=lambda result: (-result.score, result.source_id, result.index, result.chunk_id)
            )
            return tuple(ranked[:top_k])
