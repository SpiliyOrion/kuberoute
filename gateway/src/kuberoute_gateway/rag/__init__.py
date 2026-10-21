"""Local document ingestion primitives for future retrieval work."""

from kuberoute_gateway.rag.ingestion import (
    Chunk,
    ChunkingConfig,
    Document,
    DocumentInputError,
    SourceMetadata,
    chunk_document,
    ingest_paths,
    load_document,
    load_documents,
    normalize_text,
)

__all__ = (
    "Chunk",
    "ChunkingConfig",
    "Document",
    "DocumentInputError",
    "SourceMetadata",
    "chunk_document",
    "ingest_paths",
    "load_document",
    "load_documents",
    "normalize_text",
)
