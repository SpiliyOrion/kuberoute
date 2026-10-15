"""HTTP entry point for KubeRoute."""

from fastapi import FastAPI

app = FastAPI(title="KubeRoute Gateway", version="0.1.0")


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
