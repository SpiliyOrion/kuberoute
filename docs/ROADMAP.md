# KubeRoute milestones

- [ ] **M0 — Repository foundation:** repo, branch rules, CI, tests, documentation (starter scaffold included)
- [ ] **M1 — Multi-model gateway:** adapters, request validation, streaming, retries, timeouts, telemetry
- [ ] **M2 — Adaptive model router:** baselines, quality models, calibrated routing, reproducible comparisons
- [ ] **M3 — Retrieval:** document ingestion, dense/BM25/hybrid search, reranking, RAG tests
- [ ] **M4 — Joint adaptive policy:** choose both model and retrieval policy under quality/cost constraints
- [ ] **M5 — Local Kubernetes:** images, manifests/Helm, health probes, service discovery
- [ ] **M6 — Observability + dashboard:** OpenTelemetry, Prometheus, full-stack control UI
- [ ] **M7 — Cloud and controller:** Terraform, managed cloud infrastructure, Kubernetes operator, security
- [ ] **M8 — Public release:** benchmark, load tests, failure injection, docs, demos

## Definition of done for each feature

1. Contract and acceptance criteria are documented.
2. Tests added and passing locally.
3. Lint and format checks pass.
4. PR opens against `main` and CI passes.
5. README/docs updated for changed behavior.
6. Changes reviewed, then merged.
