# KubeRoute

**Adaptive AI inference and retrieval platform — in development.**

KubeRoute aims to route inference requests to the best available model and, when useful, select a retrieval strategy using measured quality, cost, latency, and resource signals. Planned: Kubernetes, cloud deployments, retrieval/RAG, evaluation, and a full-stack control plane.

## Current gateway

The FastAPI gateway has `GET /health`, `GET /health/live`, and `GET /health/ready`
endpoints. It also serves two **simulated** chat models, `mock-small` and `mock-large`.
They return deterministic responses for testing and do not run an LLM.

The gateway package also defines a minimal chat completion request/response schema, a
transport-independent backend contract, and an in-memory model registry. Requests require a
nonempty model ID and at least one system, user, or assistant message with string content.
Temperature is optional and must be between 0 and 2. Streaming is limited to `false`;
unsupported fields (including tools) and message formats return HTTP 422. Unknown model IDs
return HTTP 404. Unexpected backend failures return HTTP 500 with a generic message.

List available models:

```bash
curl http://127.0.0.1:8000/v1/models
```

Request a simulated chat completion:

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mock-small","messages":[{"role":"user","content":"Hello"}],"stream":false}'
```

Set `model` to `mock-large` to select the other simulated backend.

To enable one external OpenAI-compatible model server, set these environment variables before
starting the gateway:

```powershell
$env:KUBEROUTE_REMOTE_BASE_URL = "http://127.0.0.1:8001"
$env:KUBEROUTE_REMOTE_MODEL_ID = "my-model"
# Optional, only if the server requires a bearer token:
$env:KUBEROUTE_REMOTE_API_KEY = "your-token"
# Optional overall deadline in seconds:
$env:KUBEROUTE_REMOTE_DEADLINE_SECONDS = "60"
uvicorn kuberoute_gateway.main:app --reload
```

The base URL is the server root; the adapter appends `/v1/chat/completions`. The configured model
ID appears in `/v1/models` and is sent upstream in each request. Select it with the `model`
field in a chat request. The remote model is absent when both required variables are unset;
setting just one causes startup to fail. Keep API keys out of version control.

The adapter supports asynchronous, non-streaming text completions with a single assistant
message. It uses 3-second connect and pool timeouts, a 10-second write timeout, and a 30-second
read timeout. Upstream timeouts return HTTP 504; connection errors, upstream HTTP errors, and
invalid responses return HTTP 502 without exposing upstream details. Mock models continue to
work without an external server. Streaming, adaptive routing, retrieval, and deployment
integrations are future work.

The gateway shares one HTTP client across remote requests and closes it when the app stops.
Set `KUBEROUTE_REMOTE_DEADLINE_SECONDS` to a positive number for a total request deadline;
it includes the upstream HTTP exchange and response validation. The adapter accepts exactly
one upstream choice with an assistant text message and a `stop` or `length` finish reason.
Other finish reasons and multiple choices return HTTP 502.

## Prerequisites

- Python 3.11+
- Git

## Run locally

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
uvicorn kuberoute_gateway.main:app --reload
```

On Linux/macOS, use `python3 -m venv .venv` and `source .venv/bin/activate` instead. Visit <http://127.0.0.1:8000/health>.

## Run with Docker

On Windows, install [Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/),
enable its WSL 2 backend, and start Docker Desktop. From the repository root in PowerShell:

```powershell
docker build -t kuberoute-gateway:local .
docker run --rm -d --name kuberoute-gateway -p 8000:8000 kuberoute-gateway:local
docker inspect --format '{{.State.Health.Status}}' kuberoute-gateway
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/health/live
Invoke-RestMethod http://127.0.0.1:8000/health/ready
Invoke-RestMethod http://127.0.0.1:8000/v1/models
docker stop kuberoute-gateway
```

The image runs as an unprivileged user, installs runtime dependencies only, and checks `/health`
inside the container. It includes the gateway and its mock models, but no Ollama service or model
files. The same setup is available with `docker compose up --build -d` and
`docker compose ps`; stop it with `docker compose down`.

For an inference server running on the Windows host, set the existing environment variables
before starting Compose. Use Docker Desktop's `host.docker.internal` address because
`127.0.0.1` inside the container refers to the container itself:

```powershell
$env:KUBEROUTE_REMOTE_BASE_URL = "http://host.docker.internal:11434"
$env:KUBEROUTE_REMOTE_MODEL_ID = "qwen3:0.6b"
docker compose up --build -d
```

Compose forwards these variables only when set in the host environment, along with optional
`KUBEROUTE_REMOTE_API_KEY` and `KUBEROUTE_REMOTE_DEADLINE_SECONDS`. The container does not
start or download an external model server.

## Test

```bash
ruff check .
ruff format --check .
pytest -q
```

Optional: `pre-commit install` to automatically lint/format on each commit.

## Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md) and [docs/DECISIONS.md](docs/DECISIONS.md).

## Contribution workflow

Create a feature branch, implement and test a small change, push it, and open a pull request. Merge into `main` only after the CI checks pass. Keep credentials, patient/private data, and `.env` files out of Git.
