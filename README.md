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

Set `model` to `mock-large` to select the other simulated backend. Inference is synchronous
and non-streaming; adaptive routing, retrieval, real model downloads, and deployment integrations
are future work.

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
