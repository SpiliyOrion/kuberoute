"""Bounded-cardinality HTTP metrics for the gateway."""

from time import perf_counter

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from starlette.types import ASGIApp, Message, Receive, Scope, Send

METRICS_REGISTRY = CollectorRegistry()

REQUESTS = Counter(
    "kuberoute_http_requests_total",
    "Completed gateway HTTP requests.",
    ("method", "route", "status_code"),
    registry=METRICS_REGISTRY,
)
DURATION = Histogram(
    "kuberoute_http_request_duration_seconds",
    "Gateway HTTP request duration in seconds.",
    ("method", "route", "status_code"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120),
    registry=METRICS_REGISTRY,
)
IN_PROGRESS = Gauge(
    "kuberoute_http_requests_in_progress",
    "Gateway HTTP requests currently being processed.",
    registry=METRICS_REGISTRY,
)

_KNOWN_METHODS = frozenset({"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"})


class HTTPMetricsMiddleware:
    """Observe requests after routing, using only fixed route templates."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") == "/metrics":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "")
        method = method if method in _KNOWN_METHODS else "OTHER"
        status_code = 500
        started = perf_counter()
        IN_PROGRESS.inc()

        async def send_with_status(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_with_status)
        finally:
            try:
                route = scope.get("route")
                route_template = route.path if route is not None else "unmatched"
                labels = (method, route_template, str(status_code))
                REQUESTS.labels(*labels).inc()
                DURATION.labels(*labels).observe(perf_counter() - started)
            finally:
                IN_PROGRESS.dec()
