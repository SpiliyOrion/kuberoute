"""Minimal OpenAI-style chat completion payloads."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: StrictStr


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: StrictStr = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0, le=2, allow_inf_nan=False)
    stream: Literal[False] = False


class AssistantMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["assistant"] = "assistant"
    content: StrictStr


class ChatCompletionChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    message: AssistantMessage
    finish_reason: Literal["stop"] = "stop"


class ChatCompletionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(ge=0)
    model: str = Field(min_length=1)
    choices: list[ChatCompletionChoice] = Field(min_length=1)
