"""HTTP entry point for KubeRoute."""

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import FastAPI, HTTPException

from kuberoute_gateway.backends import ChatInputMessage, NormalizedChatRequest
from kuberoute_gateway.mock_backends import MockLargeBackend, MockSmallBackend
from kuberoute_gateway.registry import ModelRegistry, UnknownModelError
from kuberoute_gateway.schemas import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
)

app = FastAPI(title="KubeRoute Gateway", version="0.1.0")
registry = ModelRegistry()
registry.register("mock-small", MockSmallBackend())
registry.register("mock-large", MockLargeBackend())


@app.get("/v1/models")
def list_models() -> dict[str, object]:
    """List model IDs available for explicit routing."""
    return {
        "object": "list",
        "data": [{"id": model_id, "object": "model"} for model_id in registry.model_ids()],
    }


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def create_chat_completion(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """Run a synchronous chat completion on the selected model."""
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
    )
    try:
        output = backend.complete(normalized)
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid4().hex}",
            created=int(datetime.now(timezone.utc).timestamp()),
            model=request.model,
            choices=[
                ChatCompletionChoice(index=0, message=AssistantMessage(content=output.content))
            ],
        )
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
