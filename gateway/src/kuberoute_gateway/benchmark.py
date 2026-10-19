"""Small, bounded benchmark for direct Ollama and KubeRoute chat inference."""

import argparse
import asyncio
import csv
import json
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

import httpx
from pydantic import ValidationError

from kuberoute_gateway.schemas import ChatCompletionResponse

Mode = Literal["direct", "gateway"]
DEFAULT_PROMPT = "What is 2 + 2? Answer in one short sentence."
FIELDS = (
    "phase",
    "sequence",
    "status",
    "success",
    "timeout",
    "latency_seconds",
    "error_type",
    "response_metadata",
)


@dataclass(frozen=True)
class BenchmarkConfig:
    mode: Mode
    model: str = "qwen3:0.6b"
    prompt: str = DEFAULT_PROMPT
    temperature: float = 0.0
    seed: int = 42
    max_output_tokens: int = 32
    disable_thinking: bool = False
    warmup: int = 1
    requests: int = 5
    timeout_seconds: float = 120.0
    concurrency: int = 1
    ollama_url: str = "http://127.0.0.1:11434"
    gateway_url: str = "http://127.0.0.1:8000"
    output_dir: Path = Path("benchmark-results")

    def __post_init__(self) -> None:
        if not self.model or not self.prompt:
            raise ValueError("Model and prompt must be nonempty")
        if not 0 <= self.temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2")
        if self.seed < 0:
            raise ValueError("Seed must be nonnegative")
        if not 1 <= self.max_output_tokens <= 128:
            raise ValueError("Output-token limit must be between 1 and 128")
        if not 0 <= self.warmup <= 20 or not 1 <= self.requests <= 100:
            raise ValueError("Warm-up must be 0–20 and measured requests must be 1–100")
        if not 1 <= self.concurrency <= 4:
            raise ValueError("Concurrency must be between 1 and 4")
        if not 0 < self.timeout_seconds <= 600:
            raise ValueError("Request timeout must be greater than 0 and at most 600 seconds")
        for url in (self.ollama_url, self.gateway_url):
            parsed = httpx.URL(url)
            if parsed.scheme not in ("http", "https") or not parsed.host:
                raise ValueError("Base URLs must be absolute HTTP(S) URLs")

    @property
    def endpoint(self) -> str:
        base = self.ollama_url if self.mode == "direct" else self.gateway_url
        path = "/api/chat" if self.mode == "direct" else "/v1/chat/completions"
        return f"{base.rstrip('/')}{path}"

    @property
    def payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [{"role": "user", "content": self.prompt}],
            "stream": False,
        }
        if self.mode == "direct":
            payload["options"] = {
                "temperature": self.temperature,
                "seed": self.seed,
                "num_predict": self.max_output_tokens,
            }
            if self.disable_thinking:
                payload["think"] = False
        else:
            payload["temperature"] = self.temperature
            payload["seed"] = self.seed
            payload["max_tokens"] = self.max_output_tokens
            if self.disable_thinking:
                payload["reasoning_effort"] = "none"
        return payload


@dataclass(frozen=True)
class RequestResult:
    phase: str
    sequence: int
    status: int | None
    success: bool
    timeout: bool
    latency_seconds: float
    error_type: str | None
    response_metadata: dict[str, object]

    def csv_row(self) -> dict[str, object]:
        return {
            **asdict(self),
            "response_metadata": json.dumps(self.response_metadata, sort_keys=True),
        }


class BenchmarkResponseError(ValueError):
    def __init__(self, error_type: str, metadata: dict[str, object]) -> None:
        super().__init__(error_type)
        self.error_type = error_type
        self.metadata = metadata


def _response_metadata(mode: Mode, response: httpx.Response) -> dict[str, object]:
    data = response.json()
    if mode == "gateway":
        if (
            not isinstance(data, dict)
            or not isinstance(data.get("choices"), list)
            or len(data["choices"]) != 1
            or not isinstance(data["choices"][0], dict)
            or "finish_reason" not in data["choices"][0]
        ):
            raise ValueError("Invalid gateway completion")
        completion = ChatCompletionResponse.model_validate(data)
        choice = completion.choices[0]
        metadata = {
            "model": completion.model,
            "finish_reason": choice.finish_reason,
            "output_chars": len(choice.message.content),
            "prompt_tokens": completion.usage.prompt_tokens if completion.usage else None,
            "output_tokens": completion.usage.completion_tokens if completion.usage else None,
        }
        if not choice.message.content.strip():
            reason = "length_exhausted" if choice.finish_reason == "length" else "empty_content"
            raise BenchmarkResponseError(reason, metadata)
        return metadata
    if (
        not isinstance(data, dict)
        or data.get("done") is not True
        or data.get("done_reason") not in ("stop", "length")
    ):
        raise ValueError("Invalid Ollama completion")
    message = data.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ValueError("Invalid Ollama message")
    metadata = {
        "model": data.get("model"),
        "finish_reason": data.get("done_reason"),
        "output_chars": len(message["content"]),
        "prompt_tokens": data.get("prompt_eval_count"),
        "output_tokens": data.get("eval_count"),
        "server_duration_ns": data.get("total_duration"),
        "thinking_present": bool(message.get("thinking")),
    }
    if not message["content"].strip():
        reason = "length_exhausted" if data["done_reason"] == "length" else "empty_content"
        raise BenchmarkResponseError(reason, metadata)
    return metadata


async def measure_request(
    client: httpx.AsyncClient, config: BenchmarkConfig, phase: str, sequence: int
) -> RequestResult:
    start = time.perf_counter()
    status = None
    metadata: dict[str, object] = {}
    timeout = False
    error_type = None
    try:
        async with asyncio.timeout(config.timeout_seconds):
            response = await client.post(config.endpoint, json=config.payload)
            status = response.status_code
            if not 200 <= status < 300:
                timeout = status in (408, 504)
                error_type = "timeout" if timeout else "http_error"
            else:
                metadata = _response_metadata(config.mode, response)
    except (TimeoutError, httpx.TimeoutException):
        timeout = True
        error_type = "timeout"
    except httpx.RequestError:
        error_type = "network_error"
    except BenchmarkResponseError as exc:
        error_type = exc.error_type
        metadata = exc.metadata
    except (ValueError, ValidationError, KeyError, IndexError, TypeError):
        error_type = "invalid_response"
    return RequestResult(
        phase=phase,
        sequence=sequence,
        status=status,
        success=error_type is None,
        timeout=timeout,
        latency_seconds=time.perf_counter() - start,
        error_type=error_type,
        response_metadata=metadata,
    )


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(results: list[RequestResult], elapsed_seconds: float) -> dict[str, object]:
    successful = [result.latency_seconds for result in results if result.success]
    count = len(results)
    return {
        "requests": count,
        "successful_requests": len(successful),
        "timeouts": sum(result.timeout for result in results),
        "error_rate": (count - len(successful)) / count if count else 0.0,
        "mean_latency_seconds": statistics.mean(successful) if successful else None,
        "p50_latency_seconds": _percentile(successful, 0.5),
        "p95_latency_seconds": _percentile(successful, 0.95),
        "measurement_seconds": elapsed_seconds,
        "throughput_successes_per_second": len(successful) / elapsed_seconds
        if elapsed_seconds > 0
        else 0.0,
    }


async def run_benchmark(
    config: BenchmarkConfig, client: httpx.AsyncClient | None = None
) -> tuple[Path, dict[str, object]]:
    run_dir = config.output_dir / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": config.endpoint,
        "generation_controls": {
            "temperature": config.temperature,
            "seed": config.seed,
            "max_output_tokens": config.max_output_tokens,
            "disable_thinking": config.disable_thinking,
        },
        "wire_parameter_names": (
            {
                "temperature": "options.temperature",
                "seed": "options.seed",
                "max_output_tokens": "options.num_predict",
            }
            if config.mode == "direct"
            else {"temperature": "temperature", "seed": "seed", "max_output_tokens": "max_tokens"}
        ),
        "thinking_control": (
            "think=false" if config.mode == "direct" else 'reasoning_effort="none"'
        )
        if config.disable_thinking
        else None,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=config.timeout_seconds)
    try:
        with (run_dir / "requests.csv").open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=FIELDS)
            writer.writeheader()
            output.flush()
            for sequence in range(config.warmup):
                result = await measure_request(client, config, "warmup", sequence)
                writer.writerow(result.csv_row())
                output.flush()

            semaphore = asyncio.Semaphore(config.concurrency)

            async def measured(sequence: int) -> RequestResult:
                async with semaphore:
                    return await measure_request(client, config, "measurement", sequence)

            started = time.perf_counter()
            tasks = [asyncio.create_task(measured(sequence)) for sequence in range(config.requests)]
            results = []
            try:
                for task in asyncio.as_completed(tasks):
                    result = await task
                    results.append(result)
                    writer.writerow(result.csv_row())
                    output.flush()
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            summary = summarize(results, time.perf_counter() - started)
            (run_dir / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            return run_dir, summary
    finally:
        if owns_client:
            await client.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark direct Ollama or KubeRoute inference")
    parser.add_argument("--mode", choices=("direct", "gateway"), required=True)
    parser.add_argument("--model", default="qwen3:0.6b")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-output-tokens", type=int, default=32)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--gateway-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark-results"))
    args = parser.parse_args()
    try:
        config = BenchmarkConfig(**vars(args))
    except ValueError as exc:
        parser.error(str(exc))
    run_dir, summary = asyncio.run(run_benchmark(config))
    print(json.dumps({"run_dir": str(run_dir), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
