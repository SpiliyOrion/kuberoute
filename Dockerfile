FROM python:3.12-slim-bookworm AS builder

WORKDIR /build
COPY pyproject.toml README.md ./
COPY gateway/src ./gateway/src
RUN python -m pip wheel --no-cache-dir --wheel-dir /wheels .

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY --from=builder /wheels /wheels
RUN python -m pip install --no-cache-dir --no-index --find-links=/wheels kuberoute-gateway==0.1.0 \
    && rm -rf /wheels \
    && groupadd --system --gid 10001 gateway \
    && useradd --system --uid 10001 --gid gateway --no-create-home gateway

USER 10001:10001
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health', timeout=3).close()" || exit 1

CMD ["uvicorn", "kuberoute_gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
