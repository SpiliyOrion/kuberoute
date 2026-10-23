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

The gateway reuses one HTTP client per configured remote backend and closes owned clients when
the app stops.
Set `KUBEROUTE_REMOTE_DEADLINE_SECONDS` to a positive number for a total request deadline;
it includes the upstream HTTP exchange and response validation. The adapter accepts exactly
one upstream choice with an assistant text message and a `stop` or `length` finish reason.
Other finish reasons and multiple choices return HTTP 502.

To configure multiple real models, use `KUBEROUTE_REMOTE_MODELS_JSON` instead of the four
legacy variables above. It is an ordered JSON array with 1 to 8 entries. Each entry requires
`model_id` (1 to 128 safe characters) and an HTTP(S) `base_url`; it may include a numeric
`deadline_seconds` greater than 0 and at most 3600, and `api_key_env` naming an already-set
environment variable containing an API key. The key itself must not appear in the JSON.
Legacy configuration variable names are not valid key references in this mode. For Compose,
pass any referenced key variables into the container separately through an appropriate local
secret or environment configuration.
Credential-bearing URLs and URL query strings are rejected. Duplicate IDs, `auto`, and the
built-in mock IDs are also rejected. Setting any legacy remote variable alongside the JSON
causes startup to fail with a sanitized error.

For example, after confirming both model names appear in `ollama list`, run this in a fresh
PowerShell window to use two models served by the same local Ollama instance. The second model
name is an example; replace it with a model already installed on your machine:

```powershell
$models = @(
    @{ model_id = "qwen3:0.6b"; base_url = "http://127.0.0.1:11434" }
    @{ model_id = "qwen2.5:0.5b"; base_url = "http://127.0.0.1:11434"; deadline_seconds = 60 }
)
$env:KUBEROUTE_REMOTE_MODELS_JSON = ConvertTo-Json -InputObject $models -Compress -Depth 4
uvicorn kuberoute_gateway.main:app --host 127.0.0.1 --port 8000
```

For Docker Compose, use `http://host.docker.internal:11434` as both base URLs and set the
same JSON variable before `docker compose up --build -d`; Compose forwards it without changing
the localhost-only published port. Different model entries may also use different server URLs.
`/v1/models` lists all configured IDs after the mock IDs. Explicit requests use the named
backend; `model: "auto"` selects the first configured real backend. No scoring, retries, or
fallback are performed. Each backend has its own reusable HTTP client, even when two models
share one server. The existing single-model variables still work when JSON is unset.
Multi-model routing has been tested with mocked upstream responses and with sequential live
requests to `qwen3:0.6b`, `llama3.2:3b`, and `gemma3:4b` through a temporary local gateway.

## Automatic model selection

Set `model` to `auto` in the existing chat completion endpoint to select an
eligible configured inference backend. For example, after configuring the
remote backend above, run this from PowerShell:

```powershell
$body = @{ model = "auto"; messages = @(@{ role = "user"; content = "Hello" }); stream = $false } | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri http://127.0.0.1:8000/v1/chat/completions -Method Post -ContentType application/json -Body $body
```

The response's `model` field contains the selected model ID, not `auto`.
Explicit model IDs still select exactly that model and keep their existing
HTTP errors. `auto` is a reserved selector, so it is not listed by `/v1/models`
and cannot be used as a registered model ID. The configured remote backend is
eligible; mock backends are not. If no real backend is configured, `auto`
returns HTTP 503 instead of using a mock. With multiple eligible backends,
the default policy selects the first registered one without retries or fallback.
By default this is deterministic first-eligible routing, not latency-adaptive,
load-aware, quality-aware, or prompt-classifying routing. It does not query
Prometheus on the request path.

### Experimental performance-aware automatic routing

The default `KUBEROUTE_AUTO_ROUTING_POLICY=first_eligible` keeps the existing
behavior. To opt in, set the variable before starting the gateway:

```powershell
$env:KUBEROUTE_AUTO_ROUTING_POLICY = "performance_v1"
uvicorn kuberoute_gateway.main:app --host 127.0.0.1 --port 8000
```

Only `model: "auto"` requests use this policy; explicit IDs still select
exactly the requested backend. An invalid policy value fails startup with a
sanitized configuration error. The policy receives ordered IDs from the
registry's auto-eligible **remote** backends, never mock backends. Without an
eligible remote backend, `auto` returns HTTP 503. There are no retries or
fallbacks. `performance_policy.py` remains a pure selector: it accepts
immutable snapshots, fixed configuration, and a decision index, and performs
no inference or network I/O.

For each model, a rolling window retains at most 32 completed attempts by
default (configurable from 1 to 128). Each attempt stores only a fixed outcome
(`success`, `error`, or `timeout`) and, for successes, backend latency in
seconds. Snapshots expose completed, successful, failed (including timed-out),
and timed-out attempt counts and recent successful latencies. At most eight
eligible models are accepted, matching the current multi-remote configuration
limit. No prompts, responses, credentials, URLs, or request identifiers enter
the policy state or decision.

A model is scoreable after three completed attempts and at least one success
by default. Its score is `median(successful latency) / success_rate ** failure_penalty`,
where `success_rate = successes / completed attempts` and the default penalty is
1. Lower scores win; ties use eligible registration order. When no healthy
candidate is scored, an under-sampled model with the fewest attempts is chosen.
Otherwise, every fourth decision can sample an under-sampled model. Once all
candidates are scored, every sixteenth decision probes a currently losing,
non-failure-heavy model in registration-order rotation. This lets slower models
receive fresh observations and recover if their performance improves.

After the minimum sample count, a model with at least 50% failed attempts in
its recent window is failure-heavy. While a healthier scored model exists, it
is excluded from normal ranking and receives at most one round-robin probe
every 64 decisions. With no healthier scored candidate, the policy balances
unsampled models or periodically probes other failure-heavy models; with only
one model it selects that model. In these cases it cannot avoid choosing a
potentially failing backend. It does not retry or fall back. The exploration
intervals and failure threshold are fixed at their policy defaults in this
integration; they are configurable through `PerformancePolicyConfig` in code.
The gateway supplies a monotonically increasing index for automatic adaptive
decisions and records one outcome after each completed selected backend call.
Successful calls also record their backend execution duration; failures and
timeouts record only their outcome. Explicit calls still appear in Prometheus
metrics but do not train this policy. Cancelled calls add no completed-attempt
observation. An async lock coordinates selection and feedback, but is never
held during inference. Concurrent requests may select from the same last
completed snapshot and initially choose the same model.

Raw latency depends on whether a model was loaded, prompt and output length,
GPU contention, and other local conditions. The policy cannot identify a cold
Ollama model from latency alone. It does not assess response quality or handle
prompt-specific suitability, and a short rolling window is not a statistically
reliable model ranking. State exists only for one gateway process and resets
on restart; separate Kubernetes Pods make independent decisions without shared
observations. The policy is experimental until exploration, quality, and
failure behavior are validated with controlled workloads. A poor scored
candidate is probed only periodically, so improvement may take many requests
to become visible. Do not infer a global ranking from the initial live
smoke-test timings.

## HTTP metrics

The gateway serves Prometheus text metrics at `GET /metrics`. From PowerShell,
query a locally running gateway without changing its network exposure:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/metrics | Select-Object -ExpandProperty Content
```

For the local Kubernetes Service, first run
`kubectl port-forward --address 127.0.0.1 service/kuberoute-gateway 18000:8000`
in another PowerShell window, then query `http://127.0.0.1:18000/metrics`.
Stop the port-forward with Ctrl+C afterward.

- `kuberoute_http_requests_total` counts completed HTTP requests by `method`,
  normalized `route`, and `status_code`.
- `kuberoute_http_request_duration_seconds` is a Histogram with the same labels;
  its `_bucket`, `_count`, and `_sum` samples describe request duration.
- `kuberoute_http_requests_in_progress` is an unlabeled Gauge for active HTTP requests.

Unknown URLs use the fixed `unmatched` route label. `/metrics` requests are not
instrumented. Metrics contain no URLs with user-supplied path segments, prompts,
responses, API keys, or request bodies. The endpoint has no authentication, so
keep gateway access private as in the local Docker and Kubernetes examples.
Metrics are local to each gateway process or Pod. The [local Prometheus setup](docs/kubernetes_prometheus.md)
discovers and scrapes each gateway Pod independently without changing the gateway API.

### Per-model inference metrics

The same `/metrics` endpoint also exposes `kuberoute_inference_requests_total`
and the `kuberoute_inference_duration_seconds` Histogram (`_bucket`, `_count`,
and `_sum`). Both use `model` (the selected registered ID, never `auto`),
`backend_kind` (`mock`, `remote`, or `other`), `policy` (`explicit`,
`first_eligible`, or opt-in `performance_v1`), and `outcome` (`success`,
`timeout`, or `error`). The built-in mock models are labeled `mock`; exclude
them from real inference analysis.
`other` covers additional backend implementations. A selected backend call
produces one count and one duration observation, including when it fails.
Selection failures (HTTP 404 or 503) do not execute a backend and do not create
inference samples. These labels never contain prompts, responses, exceptions,
request IDs, or upstream URLs.

Prometheus scrapes each gateway Pod separately. These PromQL examples combine
replicas and consider only successful remote calls for latency:

```promql
sum by (model) (rate(kuberoute_inference_requests_total{backend_kind="remote"}[5m]))
sum by (model, outcome) (rate(kuberoute_inference_requests_total{backend_kind="remote"}[5m]))
sum by (model) (rate(kuberoute_inference_requests_total{backend_kind="remote",outcome=~"timeout|error"}[5m])) / sum by (model) (rate(kuberoute_inference_requests_total{backend_kind="remote"}[5m]))
sum by (model) (rate(kuberoute_inference_duration_seconds_sum{backend_kind="remote",outcome="success"}[5m])) / sum by (model) (rate(kuberoute_inference_duration_seconds_count{backend_kind="remote",outcome="success"}[5m]))
histogram_quantile(0.95, sum by (le, model) (rate(kuberoute_inference_duration_seconds_bucket{backend_kind="remote",outcome="success"}[5m])))
```

The queries show requests per second, outcome rates, failure rate, mean
successful-call duration, and p95 successful-call duration, respectively. A
short observation window with no calls may produce no result. Backend duration
starts immediately before `backend.complete` and includes upstream transport
and response processing; it excludes gateway routing, request validation, and
response serialization. The HTTP duration metric covers the whole gateway
request and therefore measures a different interval. Neither metric isolates
the model server's internal generation time.

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
docker run --rm -d --name kuberoute-gateway -p 127.0.0.1:8000:8000 kuberoute-gateway:local
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

## Run on local Docker Desktop Kubernetes

The [local manifest](k8s/gateway.yaml) deploys one gateway replica and an internal
`ClusterIP` Service on port 8000. It has no Ingress, NodePort, or LoadBalancer. Liveness
and readiness probes use `/health/live` and `/health/ready`; CPU and memory requests are
100m/128Mi, with limits of 500m/512Mi. Readiness checks the gateway HTTP service, not
the availability of optional Ollama inference.

From the repository root in PowerShell, confirm the Docker Desktop context and build the
current source as the manifest's local image tag. The single-node Docker Desktop **kubeadm**
cluster uses the Docker image store, so no public registry is needed. The manifest sets
`imagePullPolicy: Never` to prevent a registry pull. If Docker Desktop is configured with
the **kind** provisioner instead, its node does not share that image store; load the image
into the kind node before deploying or use the kubeadm provisioner.

```powershell
kubectl config current-context  # expect docker-desktop
kubectl get nodes
docker build -t kuberoute-gateway:latest .
docker image inspect kuberoute-gateway:latest
kubectl apply --dry-run=server --validate=strict -f k8s/gateway.yaml
```

After reviewing the dry-run result, deploy and check the rollout:

```powershell
kubectl apply -f k8s/gateway.yaml
kubectl rollout status deployment/kuberoute-gateway
kubectl get deployment,pods,service -l app=kuberoute-gateway
```

The Service is private inside the cluster. For local access, run this in a separate
PowerShell window; `--address 127.0.0.1` keeps the forwarded port on the host loopback
interface. Port 18000 avoids a conflict with a Compose gateway already using port 8000.

```powershell
kubectl port-forward --address 127.0.0.1 service/kuberoute-gateway 18000:8000
```

Then check `http://127.0.0.1:18000/health`, `/health/live`, `/health/ready`, and
`/v1/models` with `Invoke-RestMethod` in another window. Stop port-forward with Ctrl+C.
Rebuild the same image tag and run `kubectl rollout restart deployment/kuberoute-gateway`
when updating an existing local deployment.

Remote Ollama is opt-in. For Ollama on the Windows host, set the existing environment
variables on the Deployment, then wait for its new pod. No host address or model ID is
embedded in application code or the manifest:

```powershell
kubectl set env deployment/kuberoute-gateway KUBEROUTE_REMOTE_BASE_URL=http://host.docker.internal:11434 KUBEROUTE_REMOTE_MODEL_ID=qwen3:0.6b
kubectl rollout status deployment/kuberoute-gateway
Invoke-RestMethod http://127.0.0.1:18000/v1/models
```

To verify remote inference through the forwarded Kubernetes Service, send a short
non-streaming request. The returned `choices[0].message.content` should be nonempty,
with model `qwen3:0.6b` and finish reason `stop` or `length`:

```powershell
$body = @{ model = "qwen3:0.6b"; messages = @(@{ role = "user"; content = "Say hello briefly." }); stream = $false; temperature = 0; seed = 42; max_tokens = 64; reasoning_effort = "none" } | ConvertTo-Json -Depth 5
$result = Invoke-RestMethod -Uri http://127.0.0.1:18000/v1/chat/completions -Method Post -ContentType application/json -Body $body
$result.choices[0]
```

The host Ollama service must be reachable from the pod at that address. For a server
requiring an API key, provide `KUBEROUTE_REMOTE_API_KEY` through a Kubernetes Secret;
keep credentials out of the manifest and Git. An optional
`KUBEROUTE_REMOTE_DEADLINE_SECONDS` can also be set on the Deployment. With both required
remote variables omitted, the gateway registers only its mock models; setting just one
causes startup to fail.

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

## Evaluate fixed and adaptive routing

The separate evaluation CLI compares five **gateway** strategies: `fixed_qwen3`,
`fixed_llama3_2`, `fixed_gemma3`, `performance_v1`, and the optional
`first_eligible` baseline. `--task-set-version 1` is the default and preserves
the original [five synthetic tasks](gateway/src/kuberoute_gateway/evaluation_tasks.py),
their IDs, keys, order, and modulo repetition. Four request an exact single-letter
answer; one open-ended task has no automatic quality score. Select
`--task-set-version 2` for [60 distinct A/B/C tasks](gateway/src/kuberoute_gateway/evaluation_tasks_v2.py)
in arithmetic, numerical comparison, logical reasoning, string operations, and
reading comprehension. Each category has 12 tasks with four answers in each
letter position. The fixed category-interleaved order gives the first 15 tasks
three per category and five each of A/B/C. V2 uses each task at most once;
`--requests` above 60 fails before creating an output directory. Neither task
set measures general model quality or suitability for real users.

Evaluation output schema version 2 adds a separate conservative choice grade.
The existing `correct` field and `objective_accuracy` still mean **exact-format**
matching: the entire response must equal the expected letter after trimming and
uppercasing. The new `choice_status` is `correct` or `incorrect` only when one
unambiguous A/B/C choice appears in a bounded answer format, such as `B`, `B.`,
`(B)`, `Answer: B`, `Option B`, or `The correct answer is B.` The entire
response must match one supported format. Explanations, conflicting choices,
alternatives, unsupported text, and responses over 512 characters are
`ambiguous`. It does not infer answers from numbers or free-form explanations.
Warm-ups, ungraded tasks, and failed requests (including empty content) are
`not_applicable`.

Configure the gateway's three remote models as described above, then start a
single gateway process bound to localhost. Set
`KUBEROUTE_AUTO_ROUTING_POLICY=performance_v1` **before** starting it for the
adaptive run. The fixed strategies explicitly name their model and work under
either policy. Run each strategy separately from PowerShell, for example:

```powershell
python -m kuberoute_gateway.evaluation --strategy fixed_qwen3 --gateway-url http://127.0.0.1:8000 --warmup 6 --requests 20 --max-output-tokens 64 --timeout-seconds 180
python -m kuberoute_gateway.evaluation --strategy fixed_llama3_2 --gateway-url http://127.0.0.1:8000 --warmup 6 --requests 20 --max-output-tokens 64 --timeout-seconds 180
python -m kuberoute_gateway.evaluation --strategy fixed_gemma3 --gateway-url http://127.0.0.1:8000 --warmup 6 --requests 20 --max-output-tokens 64 --timeout-seconds 180
python -m kuberoute_gateway.evaluation --strategy performance_v1 --gateway-url http://127.0.0.1:8000 --warmup 6 --requests 20 --max-output-tokens 64 --timeout-seconds 180
```

These examples retain the v1 default. For a small v2 pilot, add
`--task-set-version 2 --requests 15` to **each** strategy command; for the
entire v2 set, use `--task-set-version 2 --requests 60`. Keep strategy settings
and task count identical within a comparison. For example:

```powershell
python -m kuberoute_gateway.evaluation --strategy fixed_qwen3 --task-set-version 2 --gateway-url http://127.0.0.1:8000 --warmup 6 --requests 15 --max-output-tokens 64 --timeout-seconds 180
python -m kuberoute_gateway.evaluation --strategy performance_v1 --task-set-version 2 --gateway-url http://127.0.0.1:8000 --warmup 6 --requests 15 --max-output-tokens 64 --timeout-seconds 180
```

For the optional baseline, stop that gateway, set
`$env:KUBEROUTE_AUTO_ROUTING_POLICY = "first_eligible"`, restart the **same**
three-model configuration, then run:

```powershell
python -m kuberoute_gateway.evaluation --strategy first_eligible --gateway-url http://127.0.0.1:8000 --warmup 6 --requests 20 --max-output-tokens 64 --timeout-seconds 180
```

The chat API does not report the active automatic policy. For either automatic
strategy, the CLI reads `/metrics` before and after the run and requires remote
inference counter increments to match **every** successful warm-up and measured
request's selected model and declared policy. A missing or malformed metrics
endpoint, observed counter reset, extra inference traffic, or policy mismatch
invalidates the run: it exits nonzero, marks metadata `invalid`, and writes no comparison
summary. If any request fails, completed rows and a diagnostic summary remain,
but the run is marked `completed_with_failures` and automatic policy verification
is `unverified_due_to_request_failures`; the CLI exits nonzero. Use one gateway
process and avoid unrelated inference traffic. Aggregate metrics cannot prove
per-request attribution across multiple processes. **Restart the gateway before
every independent adaptive trial** so its in-memory rolling window and decision
index reset. Separate Pods do not share state, so a load-balanced Service is
unsuitable for this first evaluation.

The default run uses two warm-ups, ten measured requests, 64 output tokens,
and a 180-second timeout. Runs are sequential and capped at 20 warm-ups and
100 measured requests for v1, or 60 distinct measured requests for v2. All paths request non-streaming generation with
temperature 0, seed 42, and `reasoning_effort="none"`; confirm these controls
work for the configured model server. The gateway forwards `max_tokens` as the
output-token limit. Use the same settings and task-set version across trials.
Adaptive warm-up requests are real inference calls and update the policy's
rolling history and decision index; measured adaptive decisions therefore
begin from the warm-up state.

Each invocation writes `metadata.json`, checkpointed `requests.csv`, and
`summary.json` under ignored `evaluation-results/`. You may choose another
directory with `--output-dir`; keep it private and Git-ignored. Metadata stores
the task-set version, ordered full-set task IDs, ordered measured task IDs,
a SHA-256 fingerprint of the full task definitions, requested strategy, counts,
and generation settings, but omits the gateway URL and prompt text. The fingerprint
helps identify task-definition changes; it is not a privacy guarantee or a
substitute for a source revision. Each request row stores
phase, sequence, task ID/category, requested and actual model, HTTP status,
bounded error category, end-to-end latency, optional reported token counts,
finish reason, nonempty-content flag, exact-format correctness, and the bounded
choice status where gradable. Metadata and summaries record schema version `2`
and grading version `exact-plus-choice-v2` for both task sets. The task-set
version is `1` or `2` according to the selected source tasks; scoring version
is independent of task-set version. Earlier schema-1 runs
remain untouched and cannot be rescored without their deliberately unsaved
response text. Keep grading versions separate when comparing results.
Generated text, exception details, and credentials are never written. Warm-ups
are recorded but excluded from measured latency and accuracy aggregates. The
metadata starts as `in_progress` and becomes `completed`,
`completed_with_failures`, `invalid`, or `interrupted`. Interrupted runs retain
flushed CSV rows but have no completed comparison summary.

The summary reports success/failure rates, mean/p50/p95 latency of successful
measured requests, selected-model counts, bounded error counts, and exact-letter
accuracy where applicable. Nonempty responses ending at the generation limit
are marked `generation_length` and excluded from success and correctness; empty
responses distinguish `empty_content` from `length_exhausted`.

The separate choice summary reports explicitly correct, explicitly incorrect,
and ambiguous counts. Identification coverage is the fraction of successful
graded requests with one identified choice; identification accuracy is the
correct fraction **among identified choices only**. Read accuracy alongside
coverage: low coverage can make identified-choice accuracy misleading. Neither
score establishes semantic reasoning quality or a reliable model ranking.
Adaptive routing optimizes observed performance, not answer quality. Repeating
v1's four graded questions is not independent evidence of model quality. V2
adds task variety without making the questions a comprehensive reasoning or
semantic-quality benchmark. Logic and reading keys have source-level rationales;
arithmetic, comparison, and string keys are checked with deterministic reference
operations in automated tests. Model responses are still graded only by the
existing exact-letter and conservative explicit-choice rules.
Compare model-selection distributions alongside grades: an adaptive score is a
mixture of the models it chose for those tasks, not a fixed model's score. Each
strategy needs the same balanced task prefix and generation controls to make
that comparison interpretable. Pilot results from 15 tasks are especially
sensitive to individual questions and should not be extrapolated to all 60.

Token-adjusted throughput is the sum of **reported** output tokens divided by
the summed end-to-end latency of successful requests with usage data. The summary reports
successful-request count, usage-covered count, and usage-coverage fraction; a
partial-coverage rate applies only to that subset. Missing token counts are
never estimated. This throughput is not isolated model token-generation speed.
Latency percentiles use only successful measured requests and linear
interpolation over their sorted latencies; the latency sample count is explicit.
Failures and timeouts appear separately in counts and bounded error categories.
In particular, p95 from 20 or fewer requests is unstable.

For an initial controlled run, start with six warm-ups and 20 measured requests
per strategy, then repeat in a rotated strategy order. Confirm all three models
are resident before comparing steady-state latency, or report first-load costs
separately. Compare output-token counts and finish reasons: a shorter response
can make a path appear faster. Adaptive warm-up and ongoing exploration add
cost, so report them separately from later selections. Fixed versus automatic
requests share the gateway path, but changing model choice, GPU contention, and
response length still confound a pure routing-overhead estimate. More repetitions
and more representative tasks are needed before making performance or quality
claims. V2's 60 distinct tasks reduce within-run repetition but remain a small,
synthetic set with possible wording and format bias. Results from different task
sets or grading versions must be reported separately; historical v1 artifacts
are never reinterpreted as v2 results.

## Local document ingestion (RAG foundation)

The `kuberoute_gateway.rag` module loads explicitly named local UTF-8 `.txt` and
`.md` files and splits their normalized text into ordered, overlapping chunks.
It does not scan directories, call a model, or change the gateway API.

```python
from pathlib import Path

from kuberoute_gateway.rag import ChunkingConfig, ingest_paths

chunks = ingest_paths(
    [Path("synthetic-note.txt"), Path("synthetic-guide.md")],
    config=ChunkingConfig(size=800, overlap=120),
)
for chunk in chunks:
    print(chunk.source_id, chunk.chunk_id, chunk.index)
```

The source ID is a SHA-256 digest of the resolved local path, so it stays stable
for the same file location but changes if the file moves. Chunk IDs also depend
on normalized document content, chunk settings, and order. Metadata contains
only the file extension and byte count, not the path. Newlines, Unicode, and
trailing whitespace are normalized deterministically; the splitter prefers
word boundaries but may split a long word. Overlap is bounded by the configured
character count. The defaults are 800 characters per chunk and 120 characters
of overlap. At most 100 files can be supplied per call; each file is limited to
1 MiB by default (configurable up to 10 MiB), and each document is limited to
10,000 chunks. Empty, malformed, and unsupported inputs raise sanitized errors.
Keep source documents private; this module does not log or save their text.

### Local embeddings and vector search

Phase 9B adds an optional async Ollama `/api/embed` adapter and an in-memory,
exact cosine-similarity index. Install an embedding model yourself before using
it; `nomic-embed-text` is a suggested starting point (`ollama list` to check,
`ollama pull nomic-embed-text` only if you choose to download it). The model and
server URL are supplied by the caller, not fixed in the application.

```python
import asyncio
from pathlib import Path

from kuberoute_gateway.rag import (
    InMemoryVectorIndex,
    OllamaEmbeddingProvider,
    ingest_paths,
)


async def main():
    chunks = ingest_paths([Path("synthetic-note.txt")])
    async with OllamaEmbeddingProvider("http://127.0.0.1:11434", "nomic-embed-text") as provider:
        index = InMemoryVectorIndex(provider.model_id)
        await index.add_chunks(chunks, provider, batch_size=16)
        results = await index.search("What does the note describe?", provider, top_k=3)
        for result in results:
            print(result.chunk_id, result.score)


asyncio.run(main())
```

The adapter sends batches of up to 32 texts with `truncate: false`; oversized
inputs fail rather than being silently truncated. Document and query embeddings
must use the same configured model and dimension. Results include the chunk
text, source ID, index, and minimal metadata for downstream retrieval, but text
is excluded from object representations and error messages. The index holds at
most 10,000 chunks in process memory, has deterministic tie-breaking, and loses
its contents when the process exits. Search is exact and linear in index size;
there is no persistence, API endpoint, or reranking yet. Keep
source documents and search results private, and do not log them.

### Hybrid retrieval: BM25, cosine search, and RRF

`BM25Index` matches exact technical terms such as `HTTP 504` and model or
version identifiers. `HybridRetriever` indexes the same chunks with BM25 and
the Phase 9B cosine index, then combines their ranked lists using reciprocal
rank fusion (RRF): `sum(weight / (60 + rank))`, with ranks starting at 1. This
combines rankings without adding incomparable BM25 and cosine scores. Semantic
retrieval can find similar wording when exact terms differ; hybrid retrieval
can use both signals. It is not guaranteed to outperform either method.

This runnable example uses **synthetic files and test-only embeddings**; it
does not call Ollama or require a downloaded model:

```python
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from kuberoute_gateway.rag import HybridRetriever, HybridSearchConfig, ingest_paths


class DemoEmbeddings:
    model_id = "demo-embeddings"

    async def embed(self, texts):
        return tuple(
            (1.0 + float("timeout" in text.casefold()), 1.0 + float("metrics" in text.casefold()))
            for text in texts
        )


async def main():
    with TemporaryDirectory() as directory:
        folder = Path(directory)
        (folder / "timeout.md").write_text("HTTP 504 means gateway timeout.", encoding="utf-8")
        (folder / "metrics.txt").write_text("Prometheus tracks failures.", encoding="utf-8")
        chunks = ingest_paths([folder / "timeout.md", folder / "metrics.txt"])
        provider = DemoEmbeddings()
        retriever = HybridRetriever(provider.model_id)
        await retriever.add_chunks(chunks, provider)
        for mode in ("semantic", "lexical", "hybrid"):
            results = await retriever.search(
                "How do I monitor timeout failures?",
                provider if mode != "lexical" else None,
                config=HybridSearchConfig(mode=mode),
            )
            print(mode, [(result.chunk_id, result.fused_score) for result in results])


asyncio.run(main())
```

For real semantic retrieval, replace `DemoEmbeddings` with the
`OllamaEmbeddingProvider` shown above and install the chosen embedding model
yourself first. Lexical-only queries do not call the embedding provider. BM25
uses case-folded Unicode word tokens, preserving dots and colons inside version
strings; it splits hyphenated words and is not a full tokenizer for every
language. Its defaults are `k1=1.2` and `b=0.75`; repeated query terms count
once. Each index holds at most 10,000 chunks. Hybrid candidate lists are
bounded to 100 per method, and index additions and searches are serialized so
queries never see a partial update. Retrieval remains in memory; there is no
reranking or quality guarantee. Keep returned passage text private.

### Opt-in RAG chat generation

`POST /v1/rag/chat/completions` joins the local document index, hybrid retrieval,
and the gateway's existing model routing and inference path. It is **disabled by
default**. Supply only explicitly named UTF-8 `.txt` or `.md` files. The index
is built at gateway startup and held in memory; changing a file requires a
gateway restart. The embedding model must already be installed in Ollama. For
example, `ollama pull nomic-embed-text` is an **optional manual setup command**;
KubeRoute never downloads it. Configure an existing generation backend with
`KUBEROUTE_REMOTE_MODELS_JSON` or the legacy single-model variables as described
above. The following PowerShell example creates a harmless temporary document
and runs the gateway locally:

```powershell
$demoDir = Join-Path $env:TEMP 'kuberoute-rag-demo'
New-Item -ItemType Directory -Force -Path $demoDir | Out-Null
$demoDoc = Join-Path $demoDir 'synthetic-guide.md'
@'
Kubernetes Deployments replace failed Pods to maintain replica counts.
HTTP 504 means an upstream inference response exceeded its timeout.
Prometheus counters track requests and histograms measure duration.
'@ | Set-Content -Path $demoDoc -Encoding utf8
$env:KUBEROUTE_RAG_ENABLED = 'true'
$env:KUBEROUTE_RAG_DOCUMENT_PATHS_JSON = ConvertTo-Json -InputObject @($demoDoc) -Compress
$env:KUBEROUTE_RAG_EMBEDDING_MODEL = 'nomic-embed-text'
$env:KUBEROUTE_RAG_EMBEDDING_BASE_URL = 'http://127.0.0.1:11434'
$env:KUBEROUTE_REMOTE_BASE_URL = 'http://127.0.0.1:11434'
$env:KUBEROUTE_REMOTE_MODEL_ID = 'qwen3:0.6b'
uvicorn kuberoute_gateway.main:app --host 127.0.0.1 --port 8000
```

The paths JSON is an explicit list, not a directory scan. Use the corresponding
container-reachable paths and embedding URL if running in Docker. Do not mix
legacy and JSON remote-model configuration. A local request can select an
explicit model or use `"auto"` to invoke the same first-eligible or opt-in
adaptive routing policy used by ordinary chat completions:

```powershell
$body = @{
  model = 'auto'
  question = 'What does the synthetic guide say about timeouts?'
  retrieval_mode = 'hybrid'
  top_k = 5
  temperature = 0
  max_tokens = 128
  stream = $false
} | ConvertTo-Json
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/v1/rag/chat/completions' -Method Post -ContentType 'application/json' -Body $body
```

`retrieval_mode` can be `hybrid` (default), `semantic`, or `lexical`. A response
has `answer`, the **actual** selected `model`, `retrieval_mode`, optional
`finish_reason` and `usage`, and cited `sources` containing only marker, hashed
source ID, hashed chunk ID, and chunk index. For example:

```json
{
  "answer": "The guide recommends a timeout [1].",
  "model": "qwen3:0.6b",
  "retrieval_mode": "hybrid",
  "finish_reason": "stop",
  "sources": [{"marker": "[1]", "source_id": "<source hash>", "chunk_id": "<chunk hash>", "chunk_index": 0}]
}
```

At most ten passages are requested and at most 6,000 evidence characters are
sent to generation; the default output limit is 256 tokens. Duplicate passage
text is omitted. If retrieval finds no
evidence, the endpoint abstains without calling a model (`model` and
`finish_reason` are null, `sources` is empty). If the answer has no valid
citations, `sources` is empty; invented numeric markers cause a 502. Source
references identify passages supplied to the model, **not** proof that a claim
is factually correct. Prompt instructions separate untrusted document text
from system rules, but prompting does not guarantee resistance to prompt
injection. Generation can still be inaccurate or incomplete, especially when
`finish_reason` is `length`. Only local inference should be used with private
documents: configuring an externally hosted inference server would send
retrieved passage text to that server. No document upload or persistent vector
store is provided. Set `KUBEROUTE_RAG_ENABLED=false` (or remove it) to disable
RAG. Missing files, unavailable embedding models, and indexing failures leave
the RAG endpoint unavailable with sanitized errors while ordinary gateway
routes remain usable.

A small localhost-only smoke test on October 23, 2026 used five temporary
synthetic documents, `nomic-embed-text:latest`, and `llama3.2:3b`. The index
initialized with five chunks. Two supported-question requests returned HTTP 200,
the expected model, nonempty answers, and citations that resolved to supplied
chunks. One unsupported-question request also returned HTTP 200 with a nonempty
answer but no citation; that does **not** establish that the model abstained.
The existing chat, health, models, and metrics routes returned HTTP 200. No
generated text or local source paths were saved. In a separate five-question
test using deliberately assigned fake embeddings, Recall@1 / MRR@3 was 1.0 / 1.0
for semantic, 0.6 / 0.6 for lexical, and 0.8 / 0.9 for hybrid retrieval. Those
figures verify the test fixture and ranking behavior, not real-world retrieval
quality or a general advantage for any mode. Returned markers prove only that
a passage was supplied: the model can cite an irrelevant passage, and absence
of citations does not by itself prove that the answer is unsupported or safe.

## Web dashboard (Phase 10A)

The React dashboard provides local, non-streaming multi-turn chat through the
existing gateway. It discovers models from `/v1/models` and adds **Automatic
routing** (`model: "auto"`) as a separate selector. Mock models are labeled
*simulated*; the response shows the backend that actually replied. Automatic
routing follows the gateway's configured policy and does not promise the
fastest or highest-quality answer. No chat text is saved between page loads.

**Managed Windows PC (no Node.js required):** GitHub Actions runs the frontend
lint, tests, and build with Node.js 22 on its runner, then uploads a
`kuberoute-dashboard-<commit SHA>` build artifact. After a successful workflow
run for a trusted commit, open that run under the repository's **Actions** tab
and download its dashboard artifact. GitHub provides it as a ZIP archive;
extract its contents into the ignored `frontend/dist` directory. From the
repository root in PowerShell:

```powershell
$dashboardZip = Join-Path $HOME 'Downloads\kuberoute-dashboard-<commit SHA>.zip'
New-Item -ItemType Directory -Force frontend\dist | Out-Null
Expand-Archive -LiteralPath $dashboardZip -DestinationPath frontend\dist -Force
Test-Path frontend\dist\index.html  # must return True
```

Download only the artifact from the intended repository and commit. The
compiled files are static HTML, JavaScript, and CSS; Node.js is needed only on
the GitHub Actions build runner, not on this PC or at runtime. The generated
build and `node_modules` are Git-ignored. A fresh workflow build is needed
after frontend source changes. Artifact availability is limited by the
workflow's retention period.

Start the gateway from this source checkout in a PowerShell window. The two
mock backends work without Ollama or model downloads:

```powershell
python -m pip install -e ".[dev]"
uvicorn kuberoute_gateway.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/`, choose `mock-small` or `mock-large`, and send a
message. FastAPI serves the compiled frontend from the same localhost origin
as `/v1/models` and `/v1/chat/completions`, so no permissive CORS setting is
needed. If `frontend/dist/index.html` is absent, `/` remains HTTP 404 and all
existing API routes remain available. An alternative build location can be
supplied through `KUBEROUTE_FRONTEND_DIST_DIR`; point it only at a trusted
extracted dashboard build. Do not bind Uvicorn to a public interface for this
local workflow. If a real backend is configured through the existing gateway
environment variables, it also appears in the selector. An unavailable
gateway shows a discovery error with a retry action.

On an unrestricted development machine, Node.js 22.12 or newer supports an
optional hot-reload workflow: run `npm ci` and `npm run dev` from `frontend`,
then open `http://127.0.0.1:5173`. Vite binds to localhost and proxies `/v1`
to `http://127.0.0.1:8000`. `npm run lint`, `npm run test`, and `npm run build`
are also available there and run in the frontend CI job. The managed-PC
workflow above does not use npm locally.

Enter sends and Shift+Enter inserts a newline. New Chat clears browser-memory
history and cancels an outstanding request. Retry resends the failed request
without duplicating the user message. Inputs are limited to 4,000 characters;
each request sends at most the latest 16 messages and 16,000 total characters.
Older conversation content may therefore be omitted from later inference
requests. The dashboard has no accounts, persistent history, streaming, RAG
controls, citations, or public deployment configuration.

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
