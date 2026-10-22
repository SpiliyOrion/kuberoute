"""Public, bounded RAG request and response payloads."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from kuberoute_gateway.schemas import ChatCompletionUsage


class RagChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: StrictStr = Field(min_length=1, max_length=128)
    question: StrictStr = Field(min_length=1, max_length=2_000)
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


class RagChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    model: str | None
    sources: list[RagSource]
    retrieval_mode: Literal["hybrid", "semantic", "lexical"]
    finish_reason: Literal["stop", "length"] | None = None
    usage: ChatCompletionUsage | None = None
