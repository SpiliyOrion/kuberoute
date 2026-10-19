"""HTTP entry point for KubeRoute."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator
from uuid import uuid4

from fastapi import FastAPI, HTTPException

from kuberoute_gateway.backends import ChatInputMessage, NormalizedChatRequest
from kuberoute_gateway.mock_backends import MockLargeBackend, MockSmallBackend
from kuberoute_gateway.registry import ModelRegistry, UnknownModelError
from kuberoute_gateway.remote_backend import (
    BackendConnectionError,
    BackendHTTPError,
    BackendResponseError,
    BackendTimeoutError,
    OpenAICompatibleBackend,
    remote_config_from_env,
)
from kuberoute_gateway.schemas import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
)


def create_registry() -> ModelRegistry:
    models = ModelRegistry()
    models.register("mock-small", MockSmallBackend())
    models.register("mock-large", MockLargeBackend())
    config = remote_config_from_env()
    if config is not None:
        models.register(
            config.model_id,
            OpenAICompatibleBackend(
                config.base_url,
                config.model_id,
                config.api_key,
                deadline_seconds=config.deadline_seconds,
            ),
        )
    return models


registry = create_registry()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    remote = None
    for model_id in registry.model_ids():
        backend = registry.get(model_id)
        if isinstance(backend, OpenAICompatibleBackend):
            remote = backend
            break
    if remote is not None:
        await remote.start()
    try:
        yield
    finally:
        if remote is not None:
            await remote.close()


app = FastAPI(title="KubeRoute Gateway", version="0.1.0", lifespan=lifespan)


@app.get("/v1/models")
def list_models() -> dict[str, object]:
    """List model IDs available for explicit routing."""
    return {
        "object": "list",
        "data": [{"id": model_id, "object": "model"} for model_id in registry.model_ids()],
    }


@app.post(
    "/v1/chat/completions", response_model=ChatCompletionResponse, response_model_exclude_none=True
)
async def create_chat_completion(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """Run a non-streaming chat completion on the selected model."""
    try:
        backend = registry.get(request.model)
    except UnknownModelError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown model: {request.model}") from exc

    normalized = NormalizedChatRequest(
        model=request.model,
        messages=tuple(
            ChatInputMessage(role=message.role, content=message.content)
            for message in request.messages
        ),
        temperature=request.temperature,
        seed=request.seed,
        max_tokens=request.max_tokens,
        reasoning_effort=request.reasoning_effort,
    )
    try:
        output = await backend.complete(normalized)
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid4().hex}",
            created=int(datetime.now(timezone.utc).timestamp()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=AssistantMessage(content=output.content),
                    finish_reason=output.finish_reason,
                )
            ],
            usage=(
                ChatCompletionUsage(
                    prompt_tokens=output.prompt_tokens,
                    completion_tokens=output.output_tokens,
                    total_tokens=output.prompt_tokens + output.output_tokens,
                )
                if output.prompt_tokens is not None and output.output_tokens is not None
                else None
            ),
        )
    except BackendTimeoutError as exc:
        raise HTTPException(status_code=504, detail="Upstream model timed out") from exc
    except (BackendConnectionError, BackendHTTPError, BackendResponseError) as exc:
        raise HTTPException(status_code=502, detail="Upstream model failed") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Model inference failed") from exc


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness endpoint for local checks and future orchestration."""
    return {"status": "ok"}


@app.get("/health/live")
def liveness() -> dict[str, str]:
    """Confirm that the gateway HTTP service is responsive."""
    return {"status": "alive"}


@app.get("/health/ready")
def readiness() -> dict[str, str]:
    """Confirm basic gateway readiness."""
    return {"status": "ready"}
