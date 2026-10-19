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
Temperature is optional and must be between 0 and 2. Optional `seed` fixes sampling and
`max_tokens` caps generated tokens. Streaming is limited to `false`;
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

## Compare inference latency

The benchmark CLI compares direct Ollama `/api/chat` with KubeRoute
`/v1/chat/completions`. Start Ollama and the gateway with the same model, then run each mode
from the repository root in PowerShell:

```powershell
python -m kuberoute_gateway.benchmark --mode direct --model qwen3:0.6b --warmup 2 --requests 10 --concurrency 1
python -m kuberoute_gateway.benchmark --mode gateway --model qwen3:0.6b --warmup 2 --requests 10 --concurrency 1
```

For Qwen3 runs where the token limit may be spent on reasoning, request no thinking in
**both** modes:

```powershell
python -m kuberoute_gateway.benchmark --mode direct --model qwen3:0.6b --disable-thinking --warmup 2 --requests 10 --concurrency 1
python -m kuberoute_gateway.benchmark --mode gateway --model qwen3:0.6b --disable-thinking --warmup 2 --requests 10 --concurrency 1
```

The direct request sends Ollama's native `think: false`. The gateway accepts the optional
`reasoning_effort: "none"` field and forwards it to the remote OpenAI-compatible chat endpoint.
Without `--disable-thinking`, neither request includes a thinking control, preserving each
endpoint's default. Use this flag only with a model and Ollama version that support both
controls; confirm parity with a small trial before interpreting benchmark comparisons.
The gateway container must also be configured with `KUBEROUTE_REMOTE_BASE_URL` and
`KUBEROUTE_REMOTE_MODEL_ID` so `/v1/models` lists the selected model.

To reproduce the Docker gateway path on Windows with Ollama running on the host, start the
container and verify model registration before the paired commands above:

```powershell
$env:KUBEROUTE_REMOTE_BASE_URL = "http://host.docker.internal:11434"
$env:KUBEROUTE_REMOTE_MODEL_ID = "qwen3:0.6b"
docker compose up --build -d
Invoke-RestMethod http://127.0.0.1:8000/v1/models
```

In the first repeated local experiment, four runs in direct–gateway–gateway–direct order
each used two warm-ups and ten measured requests. Qwen3:0.6b ran with concurrency 1,
thinking disabled, temperature 0, seed 42, and a 32-token output limit. All 40 measured
requests succeeded. Mean request-completion latency across 20 measured requests per path was
70.54 ms for direct Ollama and 116.97 ms for the Docker gateway, a 46.43 ms difference.
This is a **path-level comparison**, not an isolated measure of gateway overhead: the paths
use different Ollama endpoints, and the gateway path also crosses the published container
port and performs request handling and remote forwarding. Hardware, model load state,
background work, and run order can change the result; repeat both modes under the same
conditions and inspect output-token counts and finish reasons before comparing latency.

The defaults are one warm-up, five measured requests, 120 seconds per request, and concurrency
one. Use `--prompt`, `--temperature`, `--seed`, `--max-output-tokens`, `--timeout-seconds`,
`--ollama-url`, `--gateway-url`, and `--output-dir` to configure a run. The CLI limits
concurrency to four, measured requests to 100, and output tokens to 128. It does not start
either server or run automatically during tests.

Each invocation creates a timestamped directory under the ignored `benchmark-results/` root.
`metadata.json` records the configuration and applied options, `requests.csv` is flushed after
every request so interrupted runs retain completed rows, and `summary.json` contains p50, p95,
mean successful-request latency, successful requests per second, timeouts, and error rate.
Warm-ups are recorded but excluded from the summary. Failed requests count toward the error
rate but not latency percentiles.
HTTP errors retain their status; HTTP 200 responses with empty content are labeled
`empty_content`, or `length_exhausted` when the finish reason is `length`. Results record
finish reason, token counts when available, and whether native Ollama returned a separate
thinking field. They do not store generated text or reasoning content.

Both modes use the same model, prompt, zero-temperature default, fixed seed (42), 32-token
output limit, and non-streaming requests. The direct Ollama API names these controls
`options.temperature`, `options.seed`, and `options.num_predict`; Ollama's OpenAI-compatible
endpoint names them `temperature`, `seed`, and `max_tokens`. KubeRoute forwards the latter
unchanged. Run metadata records both the requested values and wire parameter names. The
gateway response includes upstream token usage when supplied, so per-request results record
output-token counts in both modes when available.

This aligns the requested controls, but the two Ollama endpoints may still differ in response
serialization, internal handling, and token accounting, and KubeRoute adds an HTTP hop.
Keep the model loaded, use the same hardware and background workload, run the modes in both
orders, and compare recorded output-token counts and finish reasons before interpreting latency.

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
