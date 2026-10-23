# KubeRoute security baseline

KubeRoute is a **local-development project**, not an internet-facing service.
The FastAPI gateway, dashboard, and optional Ollama server are intended for a
trusted workstation or a private Docker Desktop Kubernetes cluster. Docker
Compose publishes the gateway on `127.0.0.1` only; the Kubernetes Services are
ClusterIP and the documented port-forwards bind to `127.0.0.1`. The gateway has
**no authentication or rate limiting**. Do not expose it or Prometheus through
an Ingress, NodePort, LoadBalancer, public proxy, or non-local port binding.

## Current safeguards

- Chat requests allow at most 32 messages, 8,192 characters per message, and
  32,768 message-content characters in total. Model selectors are limited to
  128 characters. Invalid chat and RAG requests return generic HTTP 422
  details; unknown chat model IDs are not reflected in the HTTP 404 body.
  These checks limit parsed request content, but they are **not** a transport
  byte limit or denial-of-service defense.
- Handled HTTP responses receive `X-Content-Type-Options: nosniff`,
  `Referrer-Policy: no-referrer`, and `X-Frame-Options: DENY`. The optional
  compiled dashboard also has a same-origin Content Security Policy. This
  does not replace authentication or browser security review.
- Gateway metrics use bounded labels and do not contain prompts, responses,
  request bodies, or API keys. Application code does not intentionally log
  those values. Operators should not put secrets in URL paths or query strings:
  ordinary server or proxy access logs may record request URLs. Run the
  gateway with debug logging disabled when using private data.
- RAG source paths, retrieved text, vectors, and generated answers are kept
  out of default dataclass representations where applicable. RAG evidence is
  still sent to the **configured** inference server. Only configure trusted
  local backends for private documents.
- `.gitignore` excludes environment files, private keys, local data folders,
  benchmark/evaluation outputs, logs, databases, and generated frontend files.
  Git ignore rules do not remove files already tracked or prevent manual
  `git add -f`; review staged diffs before pushing.
- CI runs Python and npm dependency advisories and a redacted full-history
  secret scan. The secret scanner withholds detailed output from CI logs. A
  failed scan requires private, redacted triage; do not paste potential secret
  values into issues or comments. Scanners detect known patterns and published
  advisories, not every vulnerability or credential.

## Dependency review

The `python_dependencies` CI job audits the dependencies installed from
`.[dev]` (including the audit tool's own environment) with `pip-audit`. The
`npm_dependencies` job audits the committed lockfile after `npm ci`, including
development dependencies; high and critical npm advisories fail that job.
Lower-severity npm advisories remain visible for review. Neither job upgrades
packages automatically. Python's broad version ranges and the Docker build
can resolve a different dependency set at another time; a reproducible Python
lock/SBOM and image scan are future improvements. Audit results reflect the
advisory databases available at run time, and require applicability review.

## Before any external deployment

Add authentication and authorization, request rate limiting and a hard
request-body byte limit, TLS termination, explicit CORS and network policy,
managed secret storage/rotation, centralized log redaction and retention,
dependency and container-image pinning/scanning, and an operational incident
response plan. Review RAG document provenance and prompt-injection risks.
Do not rely on local-only assumptions once any component is reachable by
untrusted clients.

## Responsible reporting

Do not file a public issue containing exploit details, credentials, private
documents, or generated sensitive text. Use GitHub's **Report a vulnerability**
private reporting option for this repository if available. If it is not
available, request a private contact channel from the repository maintainer
without disclosing the sensitive details publicly. If a real credential was
exposed, rotate or revoke it immediately; simply deleting it from a new commit
does not erase Git history.
