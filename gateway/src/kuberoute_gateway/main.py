"""HTTP entry point for KubeRoute."""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter
from typing import AsyncIterator, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from kuberoute_gateway.adaptive_routing import AdaptiveRoutingState, auto_routing_policy_from_env
from kuberoute_gateway.backends import ChatInputMessage, NormalizedChatRequest
from kuberoute_gateway.frontend import router as frontend_router
from kuberoute_gateway.knowledge_catalog import CatalogSource, load_catalog
from kuberoute_gateway.metrics import (
    INFERENCE_DURATION,
    INFERENCE_REQUESTS,
    METRICS_REGISTRY,
    HTTPMetricsMiddleware,
)
from kuberoute_gateway.mock_backends import MockLargeBackend, MockSmallBackend
from kuberoute_gateway.performance_policy import InferenceObservation
from kuberoute_gateway.rag.config import RagConfig, rag_config_from_env
from kuberoute_gateway.rag.embeddings import (
    EmbeddingInputError,
    EmbeddingServiceError,
    OllamaEmbeddingProvider,
)
from kuberoute_gateway.rag.generation import (
    GeneratedAnswer,
    RagContextError,
    RagGenerationError,
    RagInputError,
    answer_question,
)
from kuberoute_gateway.rag.hybrid import HybridRetriever
from kuberoute_gateway.rag.ingestion import ingest_paths, load_documents
from kuberoute_gateway.rag.schemas import RagChatRequest, RagChatResponse, RagSource
from kuberoute_gateway.rag.uploads import (
    MAX_UPLOAD_BYTES,
    UploadCollection,
    UploadInputError,
    UploadTooLargeError,
    UploadUnsupportedTypeError,
    prepare_upload,
    upload_config_from_env,
)
from kuberoute_gateway.registry import ModelRegistry, UnknownModelError
from kuberoute_gateway.remote_backend import (
    BackendConnectionError,
    BackendHTTPError,
    BackendResponseError,
    BackendTimeoutError,
    OpenAICompatibleBackend,
    remote_configs_from_env,
)
from kuberoute_gateway.routing import AUTO_MODEL_ID, NoEligibleModelError, select_model
from kuberoute_gateway.schemas import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
)
from kuberoute_gateway.security import SecurityHeadersMiddleware


def create_registry() -> ModelRegistry:
    models = ModelRegistry()
    models.register("mock-small", MockSmallBackend())
    models.register("mock-large", MockLargeBackend())
    for config in remote_configs_from_env():
        try:
            backend = OpenAICompatibleBackend(
                config.base_url,
                config.model_id,
                config.api_key,
                deadline_seconds=config.deadline_seconds,
            )
        except ValueError:
            raise ValueError("Remote model configuration is invalid") from None
        models.register(
            config.model_id,
            backend,
            auto_eligible=True,
        )
    return models


registry = create_registry()


@dataclass(frozen=True)
class RagRuntime:
    retriever: HybridRetriever
    provider: OllamaEmbeddingProvider
    catalog_by_source_id: dict[str, CatalogSource] = field(default_factory=dict, repr=False)


async def _close_embedding_provider(provider: OllamaEmbeddingProvider) -> None:
    try:
        await provider.close()
    except Exception:
        raise RuntimeError("Embedding provider shutdown failed") from None


async def _initialize_rag(stack: AsyncExitStack, config: RagConfig) -> RagRuntime:
    catalog_by_source_id: dict[str, CatalogSource] = {}
    if config.source_manifest is not None:
        sources = load_catalog(config.source_manifest)
        documents = load_documents([source.document for source in sources])
        catalog_by_source_id = {
            document.source_id: source for document, source in zip(documents, sources, strict=True)
        }
    chunks = ingest_paths(config.paths)
    if len(chunks) > 2_000:
        raise ValueError("RAG document collection is too large")
    provider = OllamaEmbeddingProvider(config.embedding_base_url, config.embedding_model)
    try:
        await provider.start()
        retriever = HybridRetriever(config.embedding_model)
        await retriever.add_chunks(chunks, provider)
    except BaseException:
        await provider.close()
        raise
    stack.push_async_callback(_close_embedding_provider, provider)
    return RagRuntime(retriever, provider, catalog_by_source_id)


async def _initialize_uploads(stack: AsyncExitStack) -> UploadCollection | None:
    config = upload_config_from_env()
    if config is None:
        return None
    provider = OllamaEmbeddingProvider(config.embedding_base_url, config.embedding_model)
    try:
        await provider.start()
    except BaseException:
        await provider.close()
        raise
    stack.push_async_callback(_close_embedding_provider, provider)
    return UploadCollection(config.embedding_model, provider)


async def _close_remote_backend(backend: OpenAICompatibleBackend) -> None:
    try:
        await backend.close()
    except Exception:
        raise RuntimeError("Remote backend shutdown failed") from None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    policy = auto_routing_policy_from_env()
    eligible_remote_ids = tuple(
        model_id
        for model_id in registry.auto_eligible_model_ids()
        if isinstance(registry.get(model_id), OpenAICompatibleBackend)
    )
    app.state.adaptive_routing = (
        AdaptiveRoutingState(eligible_remote_ids) if policy == "performance_v1" else None
    )
    app.state.rag_runtime = None
    app.state.rag_enabled = False
    app.state.upload_runtime = None
    app.state.upload_enabled = False
    try:
        async with AsyncExitStack() as stack:
            for model_id in registry.model_ids():
                backend = registry.get(model_id)
                if isinstance(backend, OpenAICompatibleBackend):
                    stack.push_async_callback(_close_remote_backend, backend)
                    try:
                        await backend.start()
                    except Exception:
                        raise RuntimeError("Remote backend initialization failed") from None
            try:
                rag_config = rag_config_from_env()
                app.state.rag_enabled = rag_config is not None
                if rag_config is not None:
                    app.state.rag_runtime = await _initialize_rag(stack, rag_config)
            except Exception:
                # Other gateway routes remain usable when the optional index is unavailable.
                app.state.rag_enabled = True
                app.state.rag_runtime = None
            try:
                app.state.upload_runtime = await _initialize_uploads(stack)
                app.state.upload_enabled = app.state.upload_runtime is not None
            except Exception:
                # Upload configuration/indexing failures do not affect existing routes.
                app.state.upload_enabled = True
                app.state.upload_runtime = None
            yield
    finally:
        del app.state.adaptive_routing
        del app.state.rag_runtime
        del app.state.rag_enabled
        del app.state.upload_runtime
        del app.state.upload_enabled


app = FastAPI(title="KubeRoute Gateway", version="0.1.0", lifespan=lifespan)
app.add_middleware(HTTPMetricsMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.include_router(frontend_router)


@app.exception_handler(RequestValidationError)
async def safe_validation_error(request: Request, exc: RequestValidationError) -> Response:
    detail = "Request is invalid"
    if request.url.path == "/v1/rag/chat/completions":
        detail = "RAG request is invalid"
    elif request.url.path == "/v1/chat/completions":
        detail = "Chat request is invalid"
    return JSONResponse(status_code=422, content={"detail": detail})


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    """Expose gateway HTTP metrics in Prometheus text format."""
    return Response(generate_latest(METRICS_REGISTRY), media_type=CONTENT_TYPE_LATEST)


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
async def create_chat_completion(
    request: ChatCompletionRequest, http_request: Request
) -> ChatCompletionResponse:
    """Select a model, then run one non-streaming chat completion."""
    try:
        adaptive_state: AdaptiveRoutingState | None = http_request.app.state.adaptive_routing
        if request.model == AUTO_MODEL_ID and adaptive_state is not None:
            decision = await adaptive_state.select()
        else:
            decision = select_model(request.model, registry)
        backend = registry.get(decision.model_id)
    except NoEligibleModelError as exc:
        raise HTTPException(
            status_code=503, detail="No eligible inference model is configured"
        ) from exc
    except UnknownModelError as exc:
        raise HTTPException(status_code=404, detail="Unknown model") from exc

    normalized = NormalizedChatRequest(
        model=decision.model_id,
        messages=tuple(
            ChatInputMessage(role=message.role, content=message.content)
            for message in request.messages
        ),
        temperature=request.temperature,
        seed=request.seed,
        max_tokens=request.max_tokens,
        reasoning_effort=request.reasoning_effort,
    )
    if isinstance(backend, (MockSmallBackend, MockLargeBackend)):
        backend_kind = "mock"
    elif isinstance(backend, OpenAICompatibleBackend):
        backend_kind = "remote"
    else:
        backend_kind = "other"
    outcome: Literal["success", "timeout", "error"] | None = None
    started = perf_counter()
    try:
        try:
            output = await backend.complete(normalized)
            outcome = "success"
        except BackendTimeoutError:
            outcome = "timeout"
            raise
        except asyncio.CancelledError:
            raise
        except Exception:
            outcome = "error"
            raise
        finally:
            if outcome is not None:
                duration = perf_counter() - started
                labels = (decision.model_id, backend_kind, decision.policy, outcome)
                INFERENCE_REQUESTS.labels(*labels).inc()
                INFERENCE_DURATION.labels(*labels).observe(duration)
                if adaptive_state is not None and decision.policy == "performance_v1":
                    observation = InferenceObservation(
                        outcome,
                        max(duration, 1e-9) if outcome == "success" else None,
                    )
                    await adaptive_state.record(decision.model_id, observation)
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid4().hex}",
            created=int(datetime.now(timezone.utc).timestamp()),
            model=decision.model_id,
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


@app.post("/v1/rag/chat/completions", response_model=RagChatResponse)
async def create_rag_chat_completion(
    request: RagChatRequest, http_request: Request
) -> RagChatResponse:
    """Retrieve bounded local evidence and use the existing inference path once."""
    upload_runtime: UploadCollection | None = None
    if request.collection == "uploads":
        upload_runtime = _uploads_or_error(http_request)
        retriever = upload_runtime.retriever
        provider = upload_runtime.provider
        catalog_by_source_id = {}
    else:
        if not http_request.app.state.rag_enabled:
            raise HTTPException(status_code=503, detail="RAG is disabled")
        runtime: RagRuntime | None = http_request.app.state.rag_runtime
        if runtime is None:
            raise HTTPException(status_code=503, detail="RAG index is unavailable")
        retriever = runtime.retriever
        provider = runtime.provider
        catalog_by_source_id = runtime.catalog_by_source_id

    async def generate(normalized: NormalizedChatRequest) -> GeneratedAnswer:
        chat_request = ChatCompletionRequest(
            model=normalized.model,
            messages=[
                ChatMessage(role=message.role, content=message.content)
                for message in normalized.messages
            ],
            temperature=normalized.temperature,
            seed=normalized.seed,
            max_tokens=normalized.max_tokens,
            reasoning_effort=normalized.reasoning_effort,
        )
        completion = await create_chat_completion(chat_request, http_request)
        if len(completion.choices) != 1:
            raise RagGenerationError("Generated answer is invalid")
        choice = completion.choices[0]
        return GeneratedAnswer(
            choice.message.content, completion.model, choice.finish_reason, completion.usage
        )

    try:
        result = await answer_question(
            question=request.question,
            model=request.model,
            retriever=retriever,
            provider=provider,
            generate=generate,
            retrieval_mode=request.retrieval_mode,
            top_k=request.top_k,
            temperature=request.temperature,
            seed=request.seed,
            max_tokens=request.max_tokens,
            reasoning_effort=request.reasoning_effort,
        )
    except HTTPException as exc:
        if exc.status_code == 404:
            raise HTTPException(status_code=404, detail="Unknown model") from None
        raise
    except (RagInputError, EmbeddingInputError) as exc:
        raise HTTPException(status_code=422, detail="RAG request is invalid") from exc
    except RagContextError as exc:
        raise HTTPException(status_code=413, detail="RAG context limit exceeded") from exc
    except EmbeddingServiceError as exc:
        raise HTTPException(status_code=502, detail="Embedding service failed") from exc
    except RagGenerationError as exc:
        raise HTTPException(status_code=502, detail="Generated answer is invalid") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="RAG request failed") from exc
    if upload_runtime is not None and any(
        upload_runtime.name_for_source(source.source_id) is None for source in result.sources
    ):
        raise HTTPException(status_code=502, detail="Uploaded citation is unavailable")
    return RagChatResponse(
        answer=result.answer,
        model=result.model,
        sources=[
            RagSource(
                marker=source.marker,
                source_id=source.source_id,
                chunk_id=source.chunk_id,
                chunk_index=source.chunk_index,
                catalog_id=metadata.id
                if (metadata := catalog_by_source_id.get(source.source_id))
                else None,
                title=metadata.title if metadata else None,
                authors=metadata.authors if metadata else None,
                publication_year=metadata.publication_year if metadata else None,
                source_type=metadata.source_type if metadata else None,
                original_url=metadata.original_url if metadata else None,
                uploaded_name=(
                    upload_runtime.name_for_source(source.source_id) if upload_runtime else None
                ),
            )
            for source in result.sources
        ],
        retrieval_mode=result.retrieval_mode,
        finish_reason=result.finish_reason,
        usage=result.usage,
    )


def _uploads_or_error(http_request: Request) -> UploadCollection:
    if not http_request.app.state.upload_enabled:
        raise HTTPException(status_code=503, detail="Document uploads are disabled")
    runtime: UploadCollection | None = http_request.app.state.upload_runtime
    if runtime is None:
        raise HTTPException(status_code=503, detail="Document upload index is unavailable")
    return runtime


@app.post("/v1/rag/uploads", status_code=201)
async def upload_rag_document(http_request: Request, response: Response) -> dict[str, object]:
    """Index one bounded raw UTF-8 document in an isolated ephemeral collection."""
    runtime = _uploads_or_error(http_request)
    content_length = http_request.headers.get("content-length")
    if (
        content_length is not None
        and content_length.isdecimal()
        and int(content_length) > MAX_UPLOAD_BYTES
    ):
        raise HTTPException(status_code=413, detail="Document exceeds the upload size limit")
    raw = bytearray()
    try:
        async for part in http_request.stream():
            if len(part) > MAX_UPLOAD_BYTES - len(raw):
                raise UploadTooLargeError("Document exceeds the upload size limit")
            raw.extend(part)
        document, name = prepare_upload(
            http_request.headers.get("x-document-name"),
            http_request.headers.get("content-type"),
            bytes(raw),
        )
        indexed = await runtime.add(document, name)
    except UploadTooLargeError as exc:
        raise HTTPException(
            status_code=413, detail="Document exceeds the upload size limit"
        ) from exc
    except UploadUnsupportedTypeError as exc:
        raise HTTPException(status_code=415, detail="Document type is unsupported") from exc
    except UploadInputError as exc:
        raise HTTPException(status_code=422, detail="Document is invalid") from exc
    except (EmbeddingInputError, EmbeddingServiceError) as exc:
        raise HTTPException(status_code=502, detail="Document indexing failed") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Document indexing failed") from exc
    response.headers["Cache-Control"] = "no-store"
    return {
        "id": indexed.id,
        "name": indexed.name,
        "byte_size": indexed.byte_size,
        "chunk_count": indexed.chunk_count,
    }


@app.get("/v1/rag/uploads")
async def list_uploaded_documents(http_request: Request, response: Response) -> dict[str, object]:
    """List only documents fully committed to the uploaded collection."""
    runtime = _uploads_or_error(http_request)
    documents = await runtime.list_documents()
    response.headers["Cache-Control"] = "no-store"
    return {
        "object": "list",
        "data": [
            {
                "id": document.id,
                "name": document.name,
                "byte_size": document.byte_size,
                "chunk_count": document.chunk_count,
            }
            for document in documents
        ],
    }


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
