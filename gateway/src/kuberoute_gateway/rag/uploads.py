"""Opt-in, memory-only uploads isolated from the curated RAG index."""

import asyncio
import os
import re
import secrets
from dataclasses import dataclass, field

from kuberoute_gateway.rag.embeddings import EmbeddingProvider, validate_model_id
from kuberoute_gateway.rag.hybrid import HybridRetriever
from kuberoute_gateway.rag.ingestion import (
    Document,
    DocumentInputError,
    SourceMetadata,
    chunk_document,
    normalize_text,
)

MAX_UPLOAD_BYTES = 1_048_576
MAX_UPLOAD_DOCUMENTS = 100
MAX_UPLOAD_CHUNKS = 2_000
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._ -]{0,119}\Z")


class UploadInputError(ValueError):
    """Invalid upload, without filename or document contents."""


class UploadTooLargeError(UploadInputError):
    """The document exceeds the upload or collection limit."""


class UploadUnsupportedTypeError(UploadInputError):
    """The document type is unsupported."""


@dataclass(frozen=True)
class UploadConfig:
    embedding_model: str
    embedding_base_url: str = field(repr=False)


def upload_config_from_env() -> UploadConfig | None:
    enabled = os.getenv("KUBEROUTE_UPLOADS_ENABLED", "false").lower()
    if enabled in ("false", "0"):
        return None
    if enabled not in ("true", "1"):
        raise ValueError("Upload enabled setting is invalid")
    model = os.getenv("KUBEROUTE_RAG_EMBEDDING_MODEL", "")
    base_url = os.getenv("KUBEROUTE_RAG_EMBEDDING_BASE_URL", "")
    try:
        validate_model_id(model)
    except ValueError:
        raise ValueError("Upload embedding model configuration is invalid") from None
    if not base_url:
        raise ValueError("Upload embedding server configuration is invalid")
    return UploadConfig(model, base_url)


@dataclass(frozen=True)
class UploadedDocument:
    id: str
    name: str = field(repr=False)
    byte_size: int
    chunk_count: int


def prepare_upload(name: str | None, media_type: str | None, raw: bytes) -> tuple[Document, str]:
    """Validate one bounded UTF-8 document without touching the filesystem."""
    if not isinstance(name, str) or not _SAFE_NAME.fullmatch(name) or ".." in name:
        raise UploadInputError("Document name is invalid")
    extension = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    if extension not in ("txt", "md"):
        raise UploadUnsupportedTypeError("Document type is unsupported")
    if not isinstance(media_type, str):
        raise UploadUnsupportedTypeError("Document type is unsupported")
    parts = [part.strip().lower() for part in media_type.split(";")]
    if parts[0] not in ("text/plain", "text/markdown", "application/octet-stream") or any(
        part != "charset=utf-8" for part in parts[1:]
    ):
        raise UploadUnsupportedTypeError("Document type is unsupported")
    if not isinstance(raw, bytes) or not raw:
        raise UploadInputError("Document is empty or invalid")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise UploadTooLargeError("Document exceeds the upload size limit")
    try:
        text = normalize_text(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, DocumentInputError):
        raise UploadInputError("Document must contain valid UTF-8 text") from None
    if not text.strip():
        raise UploadInputError("Document is empty or invalid")
    source_id = secrets.token_hex(32)
    return Document(source_id, text, SourceMetadata(f".{extension}", len(raw))), name


class UploadCollection:
    """Publish metadata only after the separate hybrid index accepts every chunk."""

    def __init__(self, embedding_model: str, provider: EmbeddingProvider) -> None:
        self.retriever = HybridRetriever(embedding_model)
        self.provider = provider
        self._documents: dict[str, UploadedDocument] = {}
        self._lock = asyncio.Lock()

    async def add(self, document: Document, name: str) -> UploadedDocument:
        chunks = chunk_document(document)
        async with self._lock:
            if (
                len(self._documents) >= MAX_UPLOAD_DOCUMENTS
                or len(self.retriever) + len(chunks) > MAX_UPLOAD_CHUNKS
            ):
                raise UploadTooLargeError("Upload collection limit reached")
            await self.retriever.add_chunks(chunks, self.provider)
            result = UploadedDocument(
                document.source_id, name, document.metadata.byte_size, len(chunks)
            )
            self._documents[result.id] = result
            return result

    async def list_documents(self) -> tuple[UploadedDocument, ...]:
        async with self._lock:
            return tuple(self._documents.values())
