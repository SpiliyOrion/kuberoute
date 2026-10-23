"""Minimal OpenAI-style chat completion payloads."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

MAX_CHAT_MESSAGES = 32
MAX_CHAT_MESSAGE_CHARS = 8_192
MAX_CHAT_TOTAL_CHARS = 32_768


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: StrictStr = Field(max_length=MAX_CHAT_MESSAGE_CHARS)


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: StrictStr = Field(min_length=1, max_length=128)
    messages: list[ChatMessage] = Field(min_length=1, max_length=MAX_CHAT_MESSAGES)
    temperature: float | None = Field(default=None, ge=0, le=2, allow_inf_nan=False)
    seed: StrictInt | None = Field(default=None, ge=0)
    max_tokens: StrictInt | None = Field(default=None, ge=1, le=4096)
    reasoning_effort: Literal["none"] | None = None
    stream: Literal[False] = False

    @model_validator(mode="after")
    def limit_total_content(self) -> Self:
        if sum(len(message.content) for message in self.messages) > MAX_CHAT_TOTAL_CHARS:
            raise ValueError("Chat content exceeds the total size limit")
        return self


class AssistantMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["assistant"] = "assistant"
    content: StrictStr


class ChatCompletionChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    message: AssistantMessage
    finish_reason: Literal["stop", "length"] = "stop"


class ChatCompletionUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class ChatCompletionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(ge=0)
    model: str = Field(min_length=1)
    choices: list[ChatCompletionChoice] = Field(min_length=1)
    usage: ChatCompletionUsage | None = None
