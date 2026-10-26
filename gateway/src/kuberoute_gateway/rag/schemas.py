"""Public, bounded RAG request and response payloads."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from kuberoute_gateway.schemas import ChatCompletionUsage


class RagChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: StrictStr = Field(min_length=1, max_length=128)
    question: StrictStr = Field(min_length=1, max_length=2_000)
    collection: Literal["curated", "uploads"] = "curated"
    retrieval_mode: Literal["hybrid", "semantic", "lexical"] = "hybrid"
    top_k: StrictInt = Field(default=5, ge=1, le=10)
    temperature: float | None = Field(default=None, ge=0, le=2, allow_inf_nan=False)
    seed: StrictInt | None = Field(default=None, ge=0)
    max_tokens: StrictInt = Field(default=256, ge=1, le=4096)
    reasoning_effort: Literal["none"] | None = None
    stream: Literal[False] = False


class RagSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    marker: str
    source_id: str
    chunk_id: str
    chunk_index: int
    catalog_id: str | None = Field(default=None, exclude_if=lambda value: value is None)
    title: str | None = Field(default=None, exclude_if=lambda value: value is None)
    authors: str | None = Field(default=None, exclude_if=lambda value: value is None)
    publication_year: int | None = Field(default=None, exclude_if=lambda value: value is None)
    source_type: str | None = Field(default=None, exclude_if=lambda value: value is None)
    original_url: str | None = Field(default=None, exclude_if=lambda value: value is None)
    uploaded_name: str | None = Field(default=None, exclude_if=lambda value: value is None)


class RagChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    model: str | None
    sources: list[RagSource]
    retrieval_mode: Literal["hybrid", "semantic", "lexical"]
    finish_reason: Literal["stop", "length"] | None = None
    usage: ChatCompletionUsage | None = None
