"""Async adapter for an opt-in OpenAI-compatible model server."""

import asyncio
import json
import math
import os
import re
from dataclasses import dataclass, field
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
    base_url: str = field(repr=False)
    model_id: str
    api_key: str | None = field(default=None, repr=False)
    deadline_seconds: float | None = None


_MULTI_CONFIG_ENV = "KUBEROUTE_REMOTE_MODELS_JSON"
_LEGACY_CONFIG_ENVS = (
    "KUBEROUTE_REMOTE_BASE_URL",
    "KUBEROUTE_REMOTE_MODEL_ID",
    "KUBEROUTE_REMOTE_API_KEY",
    "KUBEROUTE_REMOTE_DEADLINE_SECONDS",
)
_MODEL_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]*\Z")
_ENV_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_MAX_REMOTE_MODELS = 8


def remote_configs_from_env() -> tuple[RemoteModelConfig, ...]:
    """Read ordered multi-model config or the unchanged legacy single-model config."""
    raw = os.environ.get(_MULTI_CONFIG_ENV)
    if raw is None:
        legacy = remote_config_from_env()
        return (legacy,) if legacy is not None else ()
    if any(name in os.environ for name in _LEGACY_CONFIG_ENVS):
        raise ValueError("Use either multi-model or legacy remote configuration, not both")
    if len(raw) > 16_384:
        raise ValueError("Multi-model configuration exceeds the size limit")
    try:
        entries = json.loads(raw)
    except (ValueError, RecursionError):
        raise ValueError("Multi-model configuration must be valid JSON") from None
    if not isinstance(entries, list) or not 1 <= len(entries) <= _MAX_REMOTE_MODELS:
        raise ValueError("Multi-model configuration must contain 1 to 8 models")

    configs: list[RemoteModelConfig] = []
    seen_ids: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not {"model_id", "base_url"} <= entry.keys():
            raise ValueError("Each remote model requires model_id and base_url")
        if entry.keys() - {"model_id", "base_url", "deadline_seconds", "api_key_env"}:
            raise ValueError("Remote model configuration contains unsupported fields")

        model_id = entry["model_id"]
        if (
            not isinstance(model_id, str)
            or not 1 <= len(model_id) <= 128
            or not _MODEL_ID_PATTERN.fullmatch(model_id)
        ):
            raise ValueError("Remote model ID must be 1 to 128 safe characters")
        if model_id in ("auto", "mock-small", "mock-large"):
            raise ValueError("Remote model ID is reserved")
        if model_id in seen_ids:
            raise ValueError("Remote model IDs must be unique")
        seen_ids.add(model_id)

        base_url = entry["base_url"]
        if not isinstance(base_url, str) or not base_url or len(base_url) > 2048:
            raise ValueError("Remote model base URL is invalid")
        try:
            url = httpx.URL(base_url)
            valid_url = (
                url.scheme in ("http", "https")
                and bool(url.host)
                and not url.userinfo
                and not url.query
                and not url.fragment
            )
        except (ValueError, httpx.InvalidURL):
            raise ValueError("Remote model base URL is invalid") from None
        if not valid_url:
            raise ValueError("Remote model base URL is invalid")

        deadline = entry.get("deadline_seconds")
        if "deadline_seconds" in entry:
            if (
                isinstance(deadline, bool)
                or not isinstance(deadline, (int, float))
                or not 0 < deadline <= 3600
                or not math.isfinite(deadline)
            ):
                raise ValueError("Remote model deadline must be between 0 and 3600 seconds")
            deadline = float(deadline)

        api_key = None
        if "api_key_env" in entry:
            name = entry["api_key_env"]
            if (
                not isinstance(name, str)
                or not 1 <= len(name) <= 128
                or not _ENV_NAME_PATTERN.fullmatch(name)
                or name in (*_LEGACY_CONFIG_ENVS, _MULTI_CONFIG_ENV)
            ):
                raise ValueError("Remote model API-key environment reference is invalid")
            api_key = os.environ.get(name)
            if not api_key:
                raise ValueError("Referenced remote model API-key environment variable is unset")

        configs.append(RemoteModelConfig(base_url, model_id, api_key, deadline))
    return tuple(configs)


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
        except ValueError:
            raise ValueError("KUBEROUTE_REMOTE_DEADLINE_SECONDS must be positive") from None
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
        try:
            url = httpx.URL(base_url)
        except httpx.InvalidURL:
            raise ValueError("Remote model base URL must be an absolute HTTP(S) URL") from None
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
