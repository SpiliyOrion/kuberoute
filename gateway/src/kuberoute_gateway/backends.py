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


@dataclass(frozen=True)
class ModelOutput:
    content: str


class ChatBackend(Protocol):
    def complete(self, request: NormalizedChatRequest) -> ModelOutput:
        """Produce one non-streaming assistant response."""
        ...
