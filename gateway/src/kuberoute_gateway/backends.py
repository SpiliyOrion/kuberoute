"""Transport-independent contract for model inference adapters."""

from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True)
class ChatInputMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class NormalizedChatRequest:
    model: str
    messages: tuple[ChatInputMessage, ...]
    temperature: float | None = None
    seed: int | None = None
    max_tokens: int | None = None
    reasoning_effort: Literal["none"] | None = None


@dataclass(frozen=True)
class ModelOutput:
    content: str
    finish_reason: Literal["stop", "length"] = "stop"
    prompt_tokens: int | None = None
    output_tokens: int | None = None


class ChatBackend(Protocol):
    async def complete(self, request: NormalizedChatRequest) -> ModelOutput:
        """Produce one non-streaming assistant response."""
        ...
