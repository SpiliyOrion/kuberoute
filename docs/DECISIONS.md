# Architecture decision log

## ADR-0001: Monorepo with isolated components

**Status:** Accepted (initial proposal)

**Decision:** Begin with a Python API in `gateway/` and add ML routing, retrieval, frontend, and Kubernetes components as their milestones start. Shared CI and docs live at the repository root.

**Reason:** Easy integration and reproducible development while the platform evolves. Avoid premature microservices.

## ADR-0002: Ship a small working vertical slice first

**Status:** Accepted (initial proposal)

**Decision:** Begin with `GET /health` and contract tests, not simultaneous LLM serving, RAG, and Kubernetes.

**Reason:** Isolate errors and build familiarity with testing, packaging, and pull-request workflow.

## Open decisions

- Initial local inference backend and hardware budget
- Cloud provider and spending limits
- Quality evaluation datasets and acceptable tradeoffs
- Timing of production secrets/authentication design
