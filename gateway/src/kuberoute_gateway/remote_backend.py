"""Async adapter for an opt-in OpenAI-compatible model server."""

import asyncio
import math
import os
from dataclasses import dataclass
from typing import Literal

import httpx
from pydantic import BaseModel, Field, StrictStr, ValidationError

from kuberoute_gateway.backends import ModelOutput, NormalizedChatRequest


class BackendTimeoutError(Exception):
    """The upstream model did not respond within the configured timeout."""


class BackendConnectionError(Exception):
    """The upstream model could not be reached."""


class BackendHTTPError(Exception):
    """The upstream model returned an unsuccessful HTTP status."""


class BackendResponseError(Exception):
    """The upstream model returned an unsupported completion payload."""


class _UpstreamMessage(BaseModel):
    role: Literal["assistant"]
    content: StrictStr


class _UpstreamChoice(BaseModel):
    index: Literal[0]
    message: _UpstreamMessage
    finish_reason: Literal["stop", "length"]


class _UpstreamUsage(BaseModel):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)


class _UpstreamCompletion(BaseModel):
    choices: list[_UpstreamChoice] = Field(min_length=1, max_length=1)
    usage: _UpstreamUsage | None = None


@dataclass(frozen=True)
class RemoteModelConfig:
    base_url: str
    model_id: str
    api_key: str | None = None
    deadline_seconds: float | None = None


def remote_config_from_env() -> RemoteModelConfig | None:
    """Enable the remote model only when both required environment values are set."""
    base_url = os.getenv("KUBEROUTE_REMOTE_BASE_URL")
    model_id = os.getenv("KUBEROUTE_REMOTE_MODEL_ID")
    if not base_url and not model_id:
        return None
    if not base_url or not model_id:
        raise ValueError("Set both KUBEROUTE_REMOTE_BASE_URL and KUBEROUTE_REMOTE_MODEL_ID")
    raw_deadline = os.getenv("KUBEROUTE_REMOTE_DEADLINE_SECONDS")
    deadline = None
    if raw_deadline is not None:
        try:
            deadline = float(raw_deadline)
        except ValueError as exc:
            raise ValueError("KUBEROUTE_REMOTE_DEADLINE_SECONDS must be positive") from exc
        if not math.isfinite(deadline) or deadline <= 0:
            raise ValueError("KUBEROUTE_REMOTE_DEADLINE_SECONDS must be positive")
    return RemoteModelConfig(
        base_url=base_url,
        model_id=model_id,
        api_key=os.getenv("KUBEROUTE_REMOTE_API_KEY") or None,
        deadline_seconds=deadline,
    )


class OpenAICompatibleBackend:
    def __init__(
        self,
        base_url: str,
        model_id: str,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
        deadline_seconds: float | None = None,
    ) -> None:
        url = httpx.URL(base_url)
        if url.scheme not in ("http", "https") or not url.host:
            raise ValueError("Remote model base URL must be an absolute HTTP(S) URL")
        if not model_id:
            raise ValueError("Remote model ID must be nonempty")
        if deadline_seconds is not None and (
            not math.isfinite(deadline_seconds) or deadline_seconds <= 0
        ):
            raise ValueError("Overall inference deadline must be positive")
        self._endpoint = f"{str(url).rstrip('/')}/v1/chat/completions"
        self._model_id = model_id
        self._api_key = api_key
        self._client = client
        self._owns_client = client is None
        self._deadline_seconds = deadline_seconds
        self._timeout = httpx.Timeout(connect=3.0, read=30.0, write=10.0, pool=3.0)

    async def start(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient()

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def complete(self, request: NormalizedChatRequest) -> ModelOutput:
        if self._client is None:
            raise RuntimeError("Remote backend has not started")
        try:
            async with asyncio.timeout(self._deadline_seconds):
                return await self._complete(request)
        except TimeoutError as exc:
            raise BackendTimeoutError("Upstream model timed out") from exc

    async def _complete(self, request: NormalizedChatRequest) -> ModelOutput:
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        payload: dict[str, object] = {
            "model": self._model_id,
            "messages": [
                {"role": message.role, "content": message.content} for message in request.messages
            ],
            "stream": False,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.reasoning_effort is not None:
            payload["reasoning_effort"] = request.reasoning_effort

        try:
            assert self._client is not None
            response = await self._client.post(
                self._endpoint, json=payload, headers=headers, timeout=self._timeout
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise BackendTimeoutError("Upstream model timed out") from exc
        except httpx.RequestError as exc:
            raise BackendConnectionError("Upstream model connection failed") from exc
        except httpx.HTTPStatusError as exc:
            raise BackendHTTPError("Upstream model returned an error") from exc

        try:
            completion = _UpstreamCompletion.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise BackendResponseError("Upstream model returned an invalid response") from exc
        choice = completion.choices[0]
        usage = completion.usage
        return ModelOutput(
            content=choice.message.content,
            finish_reason=choice.finish_reason,
            prompt_tokens=usage.prompt_tokens if usage is not None else None,
            output_tokens=usage.completion_tokens if usage is not None else None,
        )
