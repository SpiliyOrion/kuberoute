"""Local document ingestion primitives for future retrieval work."""

from kuberoute_gateway.rag.bm25 import (
    BM25Index,
    BM25Result,
    RetrievalConsistencyError,
    RetrievalInputError,
    tokenize,
)
from kuberoute_gateway.rag.embeddings import (
    EmbeddingInputError,
    EmbeddingProvider,
    EmbeddingServiceError,
    OllamaEmbeddingProvider,
)
from kuberoute_gateway.rag.hybrid import HybridResult, HybridRetriever, HybridSearchConfig
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
from kuberoute_gateway.rag.vector_index import InMemoryVectorIndex, SearchResult

__all__ = (
    "BM25Index",
    "BM25Result",
    "Chunk",
    "ChunkingConfig",
    "Document",
    "DocumentInputError",
    "EmbeddingInputError",
    "EmbeddingProvider",
    "EmbeddingServiceError",
    "HybridResult",
    "HybridRetriever",
    "HybridSearchConfig",
    "InMemoryVectorIndex",
    "OllamaEmbeddingProvider",
    "RetrievalConsistencyError",
    "RetrievalInputError",
    "SearchResult",
    "SourceMetadata",
    "chunk_document",
    "ingest_paths",
    "load_document",
    "load_documents",
    "normalize_text",
    "tokenize",
)
