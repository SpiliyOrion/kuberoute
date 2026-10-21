"""Small async embedding contract and local Ollama adapter."""

import math
import re
import unicodedata
from collections.abc import Sequence
from typing import Protocol

import httpx

MAX_BATCH_SIZE = 32
MAX_TEXT_CHARS = 10_000
MAX_EMBEDDING_DIMENSIONS = 8_192
_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}\Z")


class EmbeddingInputError(ValueError):
    """Embedding input or vectors are invalid; messages contain no input data."""


class EmbeddingServiceError(RuntimeError):
    """The embedding service failed; messages contain no upstream details."""


class EmbeddingProvider(Protocol):
    @property
    def model_id(self) -> str: ...

    async def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]: ...


def validate_model_id(model_id: str) -> None:
    if not isinstance(model_id, str) or not _MODEL_ID.fullmatch(model_id):
        raise EmbeddingInputError("Embedding model ID is invalid")


def validate_texts(texts: Sequence[str]) -> None:
    if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
        raise EmbeddingInputError("Embedding input must be a sequence of text")
    if not 1 <= len(texts) <= MAX_BATCH_SIZE:
        raise EmbeddingInputError("Embedding batch size is invalid")
    if any(
        not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_CHARS
        for text in texts
    ):
        raise EmbeddingInputError("Embedding text is invalid")
    if any(
        unicodedata.category(char) == "Cs"
        or (unicodedata.category(char) == "Cc" and char not in "\n\t")
        for text in texts
        for char in text
    ):
        raise EmbeddingInputError("Embedding text is invalid")


def normalize_vector(values: object) -> tuple[float, ...]:
    """Validate and normalize a nonzero vector without leaking its values."""
    if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= MAX_EMBEDDING_DIMENSIONS:
        raise EmbeddingInputError("Embedding vector dimensions are invalid")
    vector: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EmbeddingInputError("Embedding vector values are invalid")
        try:
            number = float(value)
        except (OverflowError, ValueError):
            raise EmbeddingInputError("Embedding vector values are invalid") from None
        if not math.isfinite(number):
            raise EmbeddingInputError("Embedding vector values are invalid")
        vector.append(number)
    scale = max(abs(value) for value in vector)
    if scale == 0:
        raise EmbeddingInputError("Embedding vector must be nonzero")
    scaled = [value / scale for value in vector]
    length = math.sqrt(math.fsum(value * value for value in scaled))
    return tuple(value / length for value in scaled)


class OllamaEmbeddingProvider:
    """Owned clients require start/close or async context management."""

    def __init__(
        self,
        base_url: str,
        model_id: str,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        validate_model_id(model_id)
        if not isinstance(base_url, str) or len(base_url) > 2048:
            raise EmbeddingInputError("Embedding server URL is invalid")
        try:
            url = httpx.URL(base_url)
        except (ValueError, httpx.InvalidURL):
            raise EmbeddingInputError("Embedding server URL is invalid") from None
        if (
            url.scheme not in ("http", "https")
            or not url.host
            or url.userinfo
            or url.query
            or url.fragment
        ):
            raise EmbeddingInputError("Embedding server URL is invalid")
        self._endpoint = f"{str(url).rstrip('/')}/api/embed"
        self._model_id = model_id
        self._client = client
        self._owns_client = client is None
        self._timeout = httpx.Timeout(connect=3.0, read=60.0, write=10.0, pool=3.0)

    @property
    def model_id(self) -> str:
        return self._model_id

    async def start(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient()

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "OllamaEmbeddingProvider":
        await self.start()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    async def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        validate_texts(texts)
        if self._client is None:
            raise EmbeddingServiceError("Embedding provider has not started")
        try:
            response = await self._client.post(
                self._endpoint,
                json={"model": self.model_id, "input": list(texts), "truncate": False},
                timeout=self._timeout,
            )
            response.raise_for_status()
        except httpx.TimeoutException:
            raise EmbeddingServiceError("Embedding server timed out") from None
        except httpx.RequestError:
            raise EmbeddingServiceError("Embedding server connection failed") from None
        except httpx.HTTPStatusError:
            raise EmbeddingServiceError("Embedding server returned an error") from None
        try:
            payload = response.json()
        except ValueError:
            raise EmbeddingServiceError("Embedding server response is invalid") from None
        if not isinstance(payload, dict) or (
            "model" in payload and payload["model"] != self.model_id
        ):
            raise EmbeddingServiceError("Embedding server response is invalid")
        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise EmbeddingServiceError("Embedding server response is invalid")
        try:
            vectors = tuple(normalize_vector(vector) for vector in embeddings)
        except EmbeddingInputError:
            raise EmbeddingServiceError("Embedding server response is invalid") from None
        if len({len(vector) for vector in vectors}) != 1:
            raise EmbeddingServiceError("Embedding server response is invalid")
        return vectors
