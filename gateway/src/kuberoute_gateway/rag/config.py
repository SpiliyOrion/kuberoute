"""Opt-in local RAG configuration, without echoing configuration values."""

import json
import os
from dataclasses import dataclass, field

from kuberoute_gateway.rag.embeddings import validate_model_id
from kuberoute_gateway.rag.ingestion import MAX_FILES


@dataclass(frozen=True)
class RagConfig:
    paths: tuple[str, ...] = field(repr=False)
    embedding_model: str
    embedding_base_url: str = field(repr=False)


def rag_config_from_env() -> RagConfig | None:
    enabled = os.getenv("KUBEROUTE_RAG_ENABLED", "false").lower()
    if enabled in ("false", "0"):
        return None
    if enabled not in ("true", "1"):
        raise ValueError("RAG enabled setting is invalid")
    raw_paths = os.getenv("KUBEROUTE_RAG_DOCUMENT_PATHS_JSON", "")
    try:
        paths = json.loads(raw_paths)
    except (ValueError, TypeError):
        raise ValueError("RAG document configuration is invalid") from None
    if (
        not isinstance(paths, list)
        or not 1 <= len(paths) <= MAX_FILES
        or any(not isinstance(path, str) or not path.strip() for path in paths)
        or len(set(paths)) != len(paths)
    ):
        raise ValueError("RAG document configuration is invalid")
    model = os.getenv("KUBEROUTE_RAG_EMBEDDING_MODEL", "")
    base_url = os.getenv("KUBEROUTE_RAG_EMBEDDING_BASE_URL", "")
    try:
        validate_model_id(model)
    except ValueError:
        raise ValueError("RAG embedding model configuration is invalid") from None
    if not base_url:
        raise ValueError("RAG embedding server configuration is invalid")
    return RagConfig(tuple(paths), model, base_url)
