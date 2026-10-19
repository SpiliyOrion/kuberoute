"""Benchmark tests use mocked HTTP transports and never call an inference server."""

import asyncio
import csv
import json
from pathlib import Path

import httpx
import pytest
from kuberoute_gateway.benchmark import (
    BenchmarkConfig,
    RequestResult,
    measure_request,
    run_benchmark,
    summarize,
)


def test_modes_translate_generation_options() -> None:
    direct = BenchmarkConfig(mode="direct", model="qwen3:0.6b", prompt="Hello")
    gateway = BenchmarkConfig(mode="gateway", model="qwen3:0.6b", prompt="Hello")
    assert direct.endpoint == "http://127.0.0.1:11434/api/chat"
    assert gateway.endpoint == "http://127.0.0.1:8000/v1/chat/completions"
    assert direct.payload["messages"] == gateway.payload["messages"]
    assert direct.payload["options"] == {
        "temperature": 0.0,
        "seed": 42,
        "num_predict": 32,
    }
    assert gateway.payload["temperature"] == 0.0
    assert gateway.payload["seed"] == 42
    assert gateway.payload["max_tokens"] == 32
    assert "options" not in gateway.payload
    assert "think" not in direct.payload
    assert "reasoning_effort" not in gateway.payload


def test_disable_thinking_uses_endpoint_specific_controls() -> None:
    direct = BenchmarkConfig(mode="direct", disable_thinking=True)
    gateway = BenchmarkConfig(mode="gateway", disable_thinking=True)
    assert direct.payload["think"] is False
    assert gateway.payload["reasoning_effort"] == "none"
    assert direct.payload["options"]["num_predict"] == gateway.payload["max_tokens"]


@pytest.mark.parametrize("mode", ["direct", "gateway"])
def test_benchmark_writes_checkpoints_and_summary(mode: str, tmp_path: Path) -> None:
    seen = 0
    config = BenchmarkConfig(mode=mode, warmup=1, requests=3, output_dir=tmp_path)

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal seen
        seen += 1
        body = json.loads(request.content)
        assert body["model"] == "qwen3:0.6b"
        assert body["stream"] is False
        assert body["messages"] == [{"role": "user", "content": config.prompt}]
        if mode == "direct":
            assert request.url.path == "/api/chat"
            assert body["options"] == {
                "temperature": config.temperature,
                "seed": config.seed,
                "num_predict": config.max_output_tokens,
            }
            return httpx.Response(
                200,
                json={
                    "model": "qwen3:0.6b",
                    "message": {"role": "assistant", "content": "Four"},
                    "done": True,
                    "done_reason": "stop",
                    "eval_count": 2,
                },
            )
        assert request.url.path == "/v1/chat/completions"
        assert body["temperature"] == config.temperature
        assert body["seed"] == config.seed
        assert body["max_tokens"] == config.max_output_tokens
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "created": 1,
                "model": "qwen3:0.6b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Four"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 6, "completion_tokens": 2, "total_tokens": 8},
            },
        )

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            run_dir, summary = await run_benchmark(config, client)
            assert seen == 4
            assert summary["requests"] == 3
            assert summary["successful_requests"] == 3
            assert summary["error_rate"] == 0
            assert summary["p95_latency_seconds"] is not None
            with (run_dir / "requests.csv").open(newline="", encoding="utf-8") as output:
                rows = list(csv.DictReader(output))
            assert len(rows) == 4
            assert rows[0]["phase"] == "warmup"
            assert {row["phase"] for row in rows[1:]} == {"measurement"}
            assert all(row["success"] == "True" for row in rows)
            assert json.loads(rows[1]["response_metadata"])["output_chars"] == 4
            assert json.loads(rows[1]["response_metadata"])["output_tokens"] == 2
            metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
            assert metadata["model"] == "qwen3:0.6b"
            assert metadata["generation_controls"] == {
                "temperature": 0.0,
                "seed": 42,
                "max_output_tokens": 32,
                "disable_thinking": False,
            }
            assert metadata["wire_parameter_names"]["max_output_tokens"] == (
                "max_tokens" if mode == "gateway" else "options.num_predict"
            )
            assert json.loads((run_dir / "summary.json").read_text(encoding="utf-8")) == summary

    asyncio.run(run())


def test_failures_record_status_and_timeout_without_response_text() -> None:
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, text="private upstream detail")
        if calls == 2:
            raise httpx.ReadTimeout("private timeout")
        if calls == 3:
            return httpx.Response(504, text="private gateway timeout")
        return httpx.Response(200, json={"choices": []})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            config = BenchmarkConfig(mode="gateway")
            results = [await measure_request(client, config, "measurement", i) for i in range(4)]
            assert [result.status for result in results] == [503, None, 504, 200]
            assert [result.error_type for result in results] == [
                "http_error",
                "timeout",
                "timeout",
                "invalid_response",
            ]
            assert results[1].timeout is True
            assert results[2].timeout is True
            assert all(not result.success for result in results)
            assert all("private" not in str(result.csv_row()) for result in results)
            summary = summarize(results, 3.0)
            assert summary["error_rate"] == 1.0
            assert summary["p50_latency_seconds"] is None
            assert summary["throughput_successes_per_second"] == 0

    asyncio.run(run())


def test_concurrency_is_bounded(tmp_path: Path) -> None:
    active = 0
    peak = 0

    async def respond(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return httpx.Response(
            200,
            json={
                "model": "qwen3:0.6b",
                "message": {"role": "assistant", "content": "Four"},
                "done": True,
                "done_reason": "stop",
            },
        )

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            config = BenchmarkConfig(
                mode="direct", warmup=0, requests=6, concurrency=2, output_dir=tmp_path
            )
            _, summary = await run_benchmark(config, client)
            assert summary["successful_requests"] == 6

    asyncio.run(run())
    assert peak == 2


def test_total_timeout_and_network_failure_are_recorded() -> None:
    calls = 0

    async def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.sleep(0.05)
            return httpx.Response(200, json={})
        raise httpx.ConnectError("private connection detail")

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            config = BenchmarkConfig(mode="direct", timeout_seconds=0.01)
            timeout_result = await measure_request(client, config, "measurement", 0)
            network_result = await measure_request(client, config, "measurement", 1)
            assert timeout_result.timeout is True
            assert timeout_result.error_type == "timeout"
            assert network_result.error_type == "network_error"
            assert network_result.status is None
            assert "private" not in str(network_result.csv_row())

    asyncio.run(run())


@pytest.mark.parametrize(
    ("mode", "body"),
    [
        ("direct", {"model": "qwen3:0.6b", "message": {"content": "Four"}, "done": False}),
        (
            "direct",
            {
                "model": "qwen3:0.6b",
                "message": {"content": "Four"},
                "done": True,
                "done_reason": "error",
            },
        ),
        (
            "gateway",
            {
                "id": "chatcmpl-1",
                "created": 1,
                "model": "qwen3:0.6b",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Four"}}],
            },
        ),
    ],
)
def test_http_200_requires_valid_nonempty_completed_output(
    mode: str, body: dict[str, object]
) -> None:
    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body))
        ) as client:
            result = await measure_request(client, BenchmarkConfig(mode=mode), "measurement", 0)
            assert result.status == 200
            assert result.success is False
            assert result.error_type == "invalid_response"

    asyncio.run(run())


@pytest.mark.parametrize(
    ("mode", "body", "expected_error"),
    [
        (
            "direct",
            {
                "model": "qwen3:0.6b",
                "message": {"content": "", "thinking": "private reasoning"},
                "done": True,
                "done_reason": "length",
                "eval_count": 32,
            },
            "length_exhausted",
        ),
        (
            "direct",
            {
                "model": "qwen3:0.6b",
                "message": {"content": ""},
                "done": True,
                "done_reason": "stop",
            },
            "empty_content",
        ),
        (
            "gateway",
            {
                "id": "chatcmpl-1",
                "created": 1,
                "model": "qwen3:0.6b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": ""},
                        "finish_reason": "length",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 32, "total_tokens": 37},
            },
            "length_exhausted",
        ),
    ],
)
def test_empty_output_diagnostics_omit_generated_text(
    mode: str, body: dict[str, object], expected_error: str
) -> None:
    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body))
        ) as client:
            result = await measure_request(client, BenchmarkConfig(mode=mode), "measurement", 0)
            assert result.status == 200
            assert result.success is False
            assert result.error_type == expected_error
            assert result.response_metadata["finish_reason"] in ("stop", "length")
            assert "private reasoning" not in str(result.csv_row())
            if mode == "direct" and expected_error == "length_exhausted":
                assert result.response_metadata["thinking_present"] is True
                assert result.response_metadata["output_tokens"] == 32

    asyncio.run(run())


@pytest.mark.parametrize(
    "change",
    [
        {"concurrency": 0},
        {"concurrency": 5},
        {"requests": 101},
        {"warmup": 21},
        {"max_output_tokens": 129},
        {"seed": -1},
        {"timeout_seconds": 0},
    ],
)
def test_configuration_limits(change: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        BenchmarkConfig(mode="direct", **change)


def test_summary_percentiles_and_success_throughput() -> None:
    results = [
        RequestResult("measurement", i, 200, True, False, value, None, {})
        for i, value in enumerate((1.0, 2.0, 3.0, 4.0, 5.0))
    ]
    summary = summarize(results, 10.0)
    assert summary["mean_latency_seconds"] == 3.0
    assert summary["p50_latency_seconds"] == 3.0
    assert summary["p95_latency_seconds"] == pytest.approx(4.8)
    assert summary["throughput_successes_per_second"] == 0.5
