"""Bounded, deterministic ingestion of explicitly named local text files."""

import hashlib
import os
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

DEFAULT_MAX_FILE_BYTES = 1_048_576
MAX_FILE_BYTES = 10_485_760
MAX_FILES = 100
MAX_CHUNK_SIZE = 10_000
MAX_CHUNKS_PER_DOCUMENT = 10_000
SUPPORTED_EXTENSIONS = frozenset({".txt", ".md"})


class DocumentInputError(ValueError):
    """Invalid or unreadable input, without source paths or document text."""


@dataclass(frozen=True)
class SourceMetadata:
    extension: Literal[".txt", ".md"]
    byte_size: int


@dataclass(frozen=True)
class Document:
    source_id: str
    text: str = field(repr=False)
    metadata: SourceMetadata


@dataclass(frozen=True)
class Chunk:
    source_id: str
    chunk_id: str
    index: int
    text: str = field(repr=False)
    metadata: SourceMetadata


@dataclass(frozen=True)
class ChunkingConfig:
    size: int = 800
    overlap: int = 120

    def __post_init__(self) -> None:
        if (
            isinstance(self.size, bool)
            or not isinstance(self.size, int)
            or not 1 <= self.size <= MAX_CHUNK_SIZE
        ):
            raise ValueError("Chunk size must be between 1 and 10000 characters")
        if (
            isinstance(self.overlap, bool)
            or not isinstance(self.overlap, int)
            or not 0 <= self.overlap < self.size
        ):
            raise ValueError("Chunk overlap must be nonnegative and smaller than chunk size")


def _validate_file_limit(max_file_bytes: int) -> None:
    if (
        isinstance(max_file_bytes, bool)
        or not isinstance(max_file_bytes, int)
        or not 1 <= max_file_bytes <= MAX_FILE_BYTES
    ):
        raise DocumentInputError("File-size limit must be between 1 and 10485760 bytes")


def normalize_text(text: str) -> str:
    """Normalize Unicode/newlines and trailing whitespace, preserving layout."""
    if not isinstance(text, str):
        raise DocumentInputError("Document text must be a string")
    normalized = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    if any(
        (unicodedata.category(char) == "Cc" and char not in "\n\t")
        or unicodedata.category(char) == "Cs"
        for char in normalized
    ):
        raise DocumentInputError("Document contains unsupported control characters")
    lines = [line.rstrip(" \t") for line in normalized.split("\n")]
    first, last = 0, len(lines)
    while first < last and not lines[first].strip():
        first += 1
    while last > first and not lines[last - 1].strip():
        last -= 1
    return "\n".join(lines[first:last])


def load_document(path: str | Path, *, max_file_bytes: int = DEFAULT_MAX_FILE_BYTES) -> Document:
    """Load one explicitly supplied UTF-8 text/Markdown file, bounded in bytes."""
    _validate_file_limit(max_file_bytes)
    if not isinstance(path, (str, Path)) or not str(path).strip():
        raise DocumentInputError("Source path must name a file")
    source = Path(path)
    extension = source.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise DocumentInputError("Source must be a .txt or .md file")
    try:
        canonical = os.path.normcase(str(source.resolve(strict=True)))
        if not source.is_file():
            raise DocumentInputError("Source must be a regular file")
        with source.open("rb") as stream:
            raw = stream.read(max_file_bytes + 1)
    except (OSError, RuntimeError, ValueError):
        raise DocumentInputError("Source file could not be read") from None
    if len(raw) > max_file_bytes:
        raise DocumentInputError("Source file exceeds the configured size limit")
    if not raw:
        raise DocumentInputError("Source file is empty")
    try:
        text = normalize_text(raw.decode("utf-8-sig"))
    except UnicodeDecodeError:
        raise DocumentInputError("Source file must be UTF-8 text") from None
    if not text.strip():
        raise DocumentInputError("Source file contains no text")
    source_id = hashlib.sha256(canonical.encode("utf-8", errors="surrogatepass")).hexdigest()
    return Document(source_id, text, SourceMetadata(extension, len(raw)))


def load_documents(
    paths: Sequence[str | Path], *, max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
) -> tuple[Document, ...]:
    """Load a bounded explicit list; directories are never discovered or scanned."""
    if isinstance(paths, (str, bytes, Path)) or not isinstance(paths, Sequence):
        raise DocumentInputError("Sources must be an explicit sequence of file paths")
    if not 1 <= len(paths) <= MAX_FILES:
        raise DocumentInputError("Source count must be between 1 and 100")
    _validate_file_limit(max_file_bytes)
    documents = tuple(load_document(path, max_file_bytes=max_file_bytes) for path in paths)
    if len({document.source_id for document in documents}) != len(documents):
        raise DocumentInputError("Duplicate source path")
    return documents


def _chunk_end(text: str, start: int, size: int) -> int:
    hard_end = min(start + size, len(text))
    if hard_end == len(text):
        return hard_end
    # Do not create a very short chunk just to find a word boundary.
    boundary = next(
        (
            index
            for index in range(hard_end - 1, start + size // 2 - 1, -1)
            if text[index].isspace()
        ),
        None,
    )
    return boundary if boundary is not None and boundary > start else hard_end


def chunk_document(document: Document, config: ChunkingConfig | None = None) -> tuple[Chunk, ...]:
    """Split normalized text in source order with bounded character overlap."""
    if config is None:
        config = ChunkingConfig()
    if not isinstance(document, Document) or not isinstance(config, ChunkingConfig):
        raise DocumentInputError("Document and chunk settings are required")
    text = normalize_text(document.text)
    if not text or not text.strip():
        raise DocumentInputError("Source file contains no text")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        while start < len(text) and text[start].isspace():
            start += 1
        if start == len(text):
            break
        end = _chunk_end(text, start, config.size)
        content = text[start:end].strip()
        if content:
            if len(chunks) >= MAX_CHUNKS_PER_DOCUMENT:
                raise DocumentInputError("Document would produce too many chunks")
            index = len(chunks)
            identity = (
                f"{document.source_id}\0{digest}\0{config.size}\0"
                f"{config.overlap}\0{index}\0{content}"
            )
            chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            chunks.append(Chunk(document.source_id, chunk_id, index, content, document.metadata))
        if end == len(text):
            break
        start = max(start + 1, end - config.overlap)
    return tuple(chunks)


def ingest_paths(
    paths: Sequence[str | Path],
    *,
    config: ChunkingConfig | None = None,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> tuple[Chunk, ...]:
    """Load explicit sources and return their ordered chunks."""
    if config is None:
        config = ChunkingConfig()
    if not isinstance(config, ChunkingConfig):
        raise DocumentInputError("Chunk settings are invalid")
    return tuple(
        chunk
        for document in load_documents(paths, max_file_bytes=max_file_bytes)
        for chunk in chunk_document(document, config)
    )
