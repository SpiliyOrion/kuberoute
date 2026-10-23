"""Optional, same-origin serving for a prebuilt local dashboard."""

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter(include_in_schema=False)
_DEFAULT_DIST = Path(__file__).resolve().parents[3] / "frontend" / "dist"
_INDEX_HEADERS = {
    "Cache-Control": "no-store",
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "connect-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
}


def _dist_directory() -> Path | None:
    configured = os.getenv("KUBEROUTE_FRONTEND_DIST_DIR")
    root = Path(configured) if configured else _DEFAULT_DIST
    try:
        resolved = root.resolve()
        index = (resolved / "index.html").resolve()
        if resolved.is_dir() and index.is_relative_to(resolved) and index.is_file():
            return resolved
    except OSError:
        pass
    return None


@router.get("/")
def dashboard() -> FileResponse:
    """Serve the compiled dashboard only when its index is present."""
    root = _dist_directory()
    if root is None:
        raise HTTPException(status_code=404)
    return FileResponse(root / "index.html", media_type="text/html", headers=_INDEX_HEADERS)


@router.get("/assets/{asset_path:path}")
def dashboard_asset(asset_path: str) -> FileResponse:
    """Serve only files inside the compiled asset directory."""
    root = _dist_directory()
    if root is None:
        raise HTTPException(status_code=404)
    assets = (root / "assets").resolve()
    if not assets.is_relative_to(root) or not assets.is_dir():
        raise HTTPException(status_code=404)
    candidate = (assets / asset_path).resolve()
    if not candidate.is_relative_to(assets) or not candidate.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(candidate, headers={"X-Content-Type-Options": "nosniff"})
