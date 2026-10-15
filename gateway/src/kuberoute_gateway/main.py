"""HTTP entry point for KubeRoute."""

from fastapi import FastAPI

app = FastAPI(title="KubeRoute Gateway", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness endpoint for local checks and future orchestration."""
    return {"status": "ok"}
