"""Routing evaluation tests use only mocked HTTP and synthetic task text."""

import asyncio
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import httpx
import kuberoute_gateway.evaluation as evaluation
import pytest
from kuberoute_gateway.evaluation import (
    MODEL_IDS,
    RESULT_FIELDS,
    EvaluationConfig,
    EvaluationResult,
    EvaluationVerificationError,
    measure_evaluation_request,
    run_evaluation,
    summarize_evaluation,
)
from kuberoute_gateway.evaluation_tasks import TASK_SET_VERSION, TASKS, WARMUP_TASK


def completion(
    model: str = "qwen3:0.6b",
    content: str = "B",
    finish_reason: str = "stop",
    *,
    usage: bool = True,
) -> dict[str, object]:
    response: dict[str, object] = {
        "id": "chatcmpl-synthetic",
        "created": 1,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage:
        response["usage"] = {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}
    return response


def inference_metrics(counts: dict[tuple[str, str, str], int]) -> str:
    lines = [
        "# HELP kuberoute_inference_requests_total Completed backend inference calls.",
        "# TYPE kuberoute_inference_requests_total counter",
    ]
    for (policy, model, outcome), count in counts.items():
        lines.append(
            'kuberoute_inference_requests_total{backend_kind="remote",'
            f'model="{model}",outcome="{outcome}",policy="{policy}"}} {count}.0'
        )
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize(
    ("strategy", "requested_model"),
    [
        ("fixed_qwen3", "qwen3:0.6b"),
        ("fixed_llama3_2", "llama3.2:3b"),
        ("fixed_gemma3", "gemma3:4b"),
        ("performance_v1", "auto"),
        ("first_eligible", "auto"),
    ],
)
def test_strategy_payload_reuses_gateway_generation_controls(
    strategy: str, requested_model: str
) -> None:
    config = EvaluationConfig(strategy=strategy)
    payload = config.payload(TASKS[0])
    assert config.requested_model == requested_model
    assert payload == {
        "model": requested_model,
        "messages": [{"role": "user", "content": TASKS[0].prompt}],
        "stream": False,
        "temperature": 0.0,
        "seed": 42,
        "max_tokens": 64,
        "reasoning_effort": "none",
    }
    assert config.endpoint == "http://127.0.0.1:8000/v1/chat/completions"


def test_tasks_are_versioned_unique_and_scored_only_when_objective() -> None:
    assert TASK_SET_VERSION == "1"
    assert len({task.task_id for task in TASKS}) == len(TASKS)
    assert len({task.prompt for task in TASKS}) == len(TASKS)
    assert any(task.expected_choice is None for task in TASKS)
    assert all(task.expected_choice in ("A", "B", "C", None) for task in TASKS)


def test_run_writes_separate_warmups_and_safe_checkpointed_outputs(tmp_path: Path) -> None:
    config = EvaluationConfig(strategy="fixed_qwen3", warmup=2, requests=5, output_dir=tmp_path)
    seen: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path == "/v1/chat/completions"
        assert body["model"] == "qwen3:0.6b"
        assert body["stream"] is False
        prompt = body["messages"][0]["content"]
        seen.append(prompt)
        task = next((task for task in TASKS if task.prompt == prompt), WARMUP_TASK)
        content = task.expected_choice or "private generated output"
        return httpx.Response(200, json=completion(content=content))

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            run_dir, summary = await run_evaluation(config, client)
        assert len(seen) == 7
        assert seen[:2] == [WARMUP_TASK.prompt] * 2
        assert seen[2:] == [task.prompt for task in TASKS]
        with (run_dir / "requests.csv").open(newline="", encoding="utf-8") as output:
            rows = list(csv.DictReader(output))
        assert tuple(rows[0]) == RESULT_FIELDS
        assert [row["phase"] for row in rows] == ["warmup"] * 2 + ["measurement"] * 5
        assert [row["task_id"] for row in rows[2:]] == [task.task_id for task in TASKS]
        assert all(row["success"] == "True" for row in rows)
        assert all(row["correct"] == "" for row in rows[:2])
        assert [row["correct"] for row in rows[2:]] == ["True"] * 4 + [""]
        assert summary["requests"] == 5
        assert summary["successful_requests"] == 5
        assert summary["objectively_scored_requests"] == 4
        assert summary["objective_accuracy"] == 1.0
        assert summary["reported_output_tokens"] == 10
        assert summary["requests_with_output_usage"] == 5
        assert summary["output_usage_coverage"] == 1.0
        assert summary["model_selection_distribution"] == {"qwen3:0.6b": 5}
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["strategy"] == "fixed_qwen3"
        assert metadata["generation_controls"]["reasoning_effort"] == "none"
        assert metadata["task_set_version"] == TASK_SET_VERSION
        assert json.loads((run_dir / "summary.json").read_text(encoding="utf-8")) == summary
        artifacts = "".join(path.read_text(encoding="utf-8") for path in run_dir.iterdir())
        assert "private generated output" not in artifacts
        assert all(task.prompt not in artifacts for task in TASKS)
        assert WARMUP_TASK.prompt not in artifacts
        assert config.gateway_url not in artifacts

    asyncio.run(run())


def test_auto_strategy_records_actual_selected_model_and_missing_usage() -> None:
    config = EvaluationConfig(strategy="performance_v1")

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, json=completion(model="llama3.2:3b", usage=False)
                )
            )
        ) as client:
            result = await measure_evaluation_request(client, config, TASKS[0], "measurement", 0)
        assert result.success is True
        assert result.requested_model == "auto"
        assert result.selected_model == "llama3.2:3b"
        assert result.correct is True
        assert result.output_tokens is None
        summary = summarize_evaluation(config.strategy, [result])
        assert summary["model_selection_distribution"] == {"llama3.2:3b": 1}
        assert summary["requests_with_output_usage"] == 0
        assert summary["output_usage_coverage"] == 0.0
        assert summary["token_adjusted_throughput_tokens_per_second"] is None

    asyncio.run(run())


@pytest.mark.parametrize("strategy", ["performance_v1", "first_eligible"])
def test_automatic_run_verifies_actual_policy_and_selected_models(
    tmp_path: Path, strategy: str
) -> None:
    config = EvaluationConfig(strategy=strategy, warmup=1, requests=3, output_dir=tmp_path)
    counts: dict[tuple[str, str, str], int] = {}
    sent_models: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metrics":
            return httpx.Response(200, text=inference_metrics(counts))
        body = json.loads(request.content)
        assert body["model"] == "auto"
        selected = MODEL_IDS[len(sent_models) % 2] if strategy == "performance_v1" else MODEL_IDS[0]
        sent_models.append(selected)
        key = (strategy, selected, "success")
        counts[key] = counts.get(key, 0) + 1
        return httpx.Response(200, json=completion(model=selected))

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            run_dir, summary = await run_evaluation(config, client)
        assert len(sent_models) == 4
        assert summary["run_status"] == "completed"
        assert summary["policy_verification"] == "verified"
        assert summary["warmup_requests"] == 1
        assert summary["requests"] == 3
        assert sum(summary["model_selection_distribution"].values()) == 3
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["policy_verification"] == "verified"

    asyncio.run(run())


@pytest.mark.parametrize(
    ("declared", "actual"),
    [("performance_v1", "first_eligible"), ("first_eligible", "performance_v1")],
)
def test_policy_mismatch_preserves_rows_but_cannot_publish_a_summary(
    tmp_path: Path, declared: str, actual: str
) -> None:
    config = EvaluationConfig(strategy=declared, warmup=1, requests=1, output_dir=tmp_path)
    counts: dict[tuple[str, str, str], int] = {}

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metrics":
            return httpx.Response(200, text=inference_metrics(counts))
        key = (actual, MODEL_IDS[0], "success")
        counts[key] = counts.get(key, 0) + 1
        return httpx.Response(200, json=completion())

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            with pytest.raises(EvaluationVerificationError, match="metric deltas"):
                await run_evaluation(config, client)
        run_dir = next(tmp_path.iterdir())
        with (run_dir / "requests.csv").open(newline="", encoding="utf-8") as output:
            assert len(list(csv.DictReader(output))) == 2
        assert not (run_dir / "summary.json").exists()
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["run_status"] == "invalid"
        assert metadata["policy_verification"] == "unverified"

    asyncio.run(run())


@pytest.mark.parametrize("metrics_response", [httpx.Response(404), httpx.Response(200, text="bad")])
def test_missing_or_malformed_metrics_fail_before_inference(
    tmp_path: Path, metrics_response: httpx.Response
) -> None:
    sent = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal sent
        if request.url.path == "/metrics":
            return metrics_response
        sent += 1
        return httpx.Response(200, json=completion())

    async def run() -> None:
        config = EvaluationConfig(strategy="performance_v1", output_dir=tmp_path)
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            with pytest.raises(EvaluationVerificationError, match="unavailable or invalid"):
                await run_evaluation(config, client)
        assert sent == 0
        run_dir = next(tmp_path.iterdir())
        assert not (run_dir / "summary.json").exists()
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["run_status"] == "invalid"

    asyncio.run(run())


def test_counter_reset_invalidates_automatic_run(tmp_path: Path) -> None:
    metrics_reads = 0
    key = ("performance_v1", MODEL_IDS[0], "success")

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal metrics_reads
        if request.url.path == "/metrics":
            metrics_reads += 1
            count = 5 if metrics_reads == 1 else 1
            return httpx.Response(200, text=inference_metrics({key: count}))
        return httpx.Response(200, json=completion())

    async def run() -> None:
        config = EvaluationConfig(
            strategy="performance_v1", warmup=0, requests=1, output_dir=tmp_path
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            with pytest.raises(EvaluationVerificationError, match="metric deltas"):
                await run_evaluation(config, client)
        run_dir = next(tmp_path.iterdir())
        assert not (run_dir / "summary.json").exists()
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["run_status"] == "invalid"

    asyncio.run(run())


def test_automatic_run_with_request_failure_is_explicitly_unverified(tmp_path: Path) -> None:
    counts: dict[tuple[str, str, str], int] = {}
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        if request.url.path == "/metrics":
            return httpx.Response(200, text=inference_metrics(counts))
        calls += 1
        if calls == 2:
            return httpx.Response(504, text="private backend detail")
        key = ("performance_v1", MODEL_IDS[0], "success")
        counts[key] = counts.get(key, 0) + 1
        return httpx.Response(200, json=completion())

    async def run() -> None:
        config = EvaluationConfig(
            strategy="performance_v1", warmup=0, requests=3, output_dir=tmp_path
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            run_dir, summary = await run_evaluation(config, client)
        assert summary["run_status"] == "completed_with_failures"
        assert summary["policy_verification"] == "unverified_due_to_request_failures"
        assert summary["requests"] == 3
        assert summary["successful_requests"] == 2
        assert summary["error_categories"] == {"timeout": 1}
        assert summary["timeout_requests"] == 1
        assert summary["latency_sample_count"] == 2
        assert summary["output_usage_coverage"] == 1.0
        assert summary["requests_with_output_usage"] == 2
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["run_status"] == "completed_with_failures"
        assert "private" not in (run_dir / "requests.csv").read_text(encoding="utf-8")

    asyncio.run(run())


def test_strict_objective_grading_and_token_throughput() -> None:
    base = EvaluationResult(
        phase="measurement",
        sequence=0,
        strategy="fixed_qwen3",
        task_id=TASKS[0].task_id,
        category=TASKS[0].category,
        requested_model=MODEL_IDS[0],
        selected_model=MODEL_IDS[0],
        http_status=200,
        success=True,
        error_category=None,
        latency_seconds=1.0,
        prompt_tokens=5,
        output_tokens=10,
        finish_reason="stop",
        content_nonempty=True,
        correct=True,
    )
    results = [
        base,
        replace(base, sequence=1, latency_seconds=3.0, output_tokens=30, correct=False),
        replace(base, sequence=2, latency_seconds=2.0, output_tokens=None, correct=None),
        replace(base, sequence=3, success=False, error_category="timeout", selected_model=None),
    ]
    summary = summarize_evaluation("fixed_qwen3", results)
    assert summary["requests"] == 4
    assert summary["successful_requests"] == 3
    assert summary["success_rate"] == 0.75
    assert summary["mean_latency_seconds"] == 2.0
    assert summary["p50_latency_seconds"] == 2.0
    assert summary["p95_latency_seconds"] == pytest.approx(2.9)
    assert summary["objective_accuracy"] == 0.5
    assert summary["requests_with_output_usage"] == 2
    assert summary["reported_output_tokens"] == 40
    assert summary["token_adjusted_throughput_tokens_per_second"] == 10.0
    assert summary["error_categories"] == {"timeout": 1}
    assert summary["timeout_requests"] == 1
    assert summary["latency_sample_count"] == 3
    assert summary["output_usage_coverage"] == pytest.approx(2 / 3)


def test_empty_summary_has_no_invented_latency_accuracy_or_usage() -> None:
    summary = summarize_evaluation("performance_v1", [])
    assert summary["requests"] == 0
    assert summary["success_rate"] == 0.0
    assert summary["p50_latency_seconds"] is None
    assert summary["p95_latency_seconds"] is None
    assert summary["objective_accuracy"] is None
    assert summary["token_adjusted_throughput_tokens_per_second"] is None
    assert summary["output_usage_coverage"] is None


@pytest.mark.parametrize(
    ("response", "expected_error"),
    [
        (httpx.Response(500, text="private upstream detail"), "http_error"),
        (httpx.Response(504, text="private timeout detail"), "timeout"),
        (httpx.Response(200, text="not json"), "invalid_response"),
        (httpx.Response(200, json={"choices": []}), "invalid_response"),
        (
            httpx.Response(200, json={**completion(), "choices": completion()["choices"] * 2}),
            "invalid_response",
        ),
        (
            httpx.Response(
                200,
                json={
                    **completion(),
                    "usage": {"prompt_tokens": 7, "completion_tokens": -1, "total_tokens": 6},
                },
            ),
            "invalid_response",
        ),
        (httpx.Response(200, json=completion(model="unregistered-model")), "invalid_response"),
        (httpx.Response(200, json=completion(model="gemma3:4b")), "wrong_model"),
        (httpx.Response(200, json=completion(content="")), "empty_content"),
        (
            httpx.Response(200, json=completion(content="", finish_reason="length")),
            "length_exhausted",
        ),
        (
            httpx.Response(200, json=completion(content="B", finish_reason="length")),
            "generation_length",
        ),
        (httpx.Response(200, json=completion(finish_reason="tool_calls")), "invalid_response"),
    ],
)
def test_failures_and_malformed_responses_are_bounded(
    response: httpx.Response, expected_error: str
) -> None:
    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: response)
        ) as client:
            result = await measure_evaluation_request(
                client, EvaluationConfig(strategy="fixed_qwen3"), TASKS[0], "measurement", 0
            )
        assert result.success is False
        assert result.error_category == expected_error
        assert result.correct is None
        assert "private" not in str(result.csv_row())
        assert "unregistered-model" not in str(result.csv_row())
        assert "tool_calls" not in str(result.csv_row())

    asyncio.run(run())


@pytest.mark.parametrize(
    ("failure", "expected_error"),
    [
        (httpx.ReadTimeout("private timeout"), "timeout"),
        (httpx.ConnectError("private connection"), "network_error"),
    ],
)
def test_network_failures_do_not_store_exception_text(
    failure: Exception, expected_error: str
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        raise failure

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            result = await measure_evaluation_request(
                client, EvaluationConfig(strategy="performance_v1"), TASKS[0], "measurement", 0
            )
        assert result.http_status is None
        assert result.error_category == expected_error
        assert "private" not in str(result.csv_row())

    asyncio.run(run())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"strategy": "unknown"},
        {"strategy": "fixed_qwen3", "gateway_url": "http://user:secret@localhost:8000"},
        {"strategy": "fixed_qwen3", "gateway_url": "http://localhost:8000/?token=secret"},
        {"strategy": "fixed_qwen3", "warmup": -1},
        {"strategy": "fixed_qwen3", "requests": 0},
        {"strategy": "fixed_qwen3", "max_output_tokens": 129},
        {"strategy": "fixed_qwen3", "timeout_seconds": float("inf")},
    ],
)
def test_invalid_config_does_not_echo_sensitive_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError) as error:
        EvaluationConfig(**kwargs)
    assert "secret" not in str(error.value)


def test_interruption_keeps_completed_csv_rows(tmp_path: Path) -> None:
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return httpx.Response(200, json=completion())

    async def run() -> None:
        config = EvaluationConfig(strategy="fixed_qwen3", warmup=0, requests=3, output_dir=tmp_path)
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            with pytest.raises(RuntimeError, match="simulated interruption"):
                await run_evaluation(config, client)
        run_dir = next(tmp_path.iterdir())
        with (run_dir / "requests.csv").open(newline="", encoding="utf-8") as output:
            rows = list(csv.DictReader(output))
        assert len(rows) == 1
        assert rows[0]["sequence"] == "0"
        assert not (run_dir / "summary.json").exists()
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["run_status"] == "interrupted"

    asyncio.run(run())


@pytest.mark.parametrize("invalid_policy", [False, True])
def test_cli_exits_nonzero_for_failed_or_unverified_runs(
    monkeypatch: pytest.MonkeyPatch, invalid_policy: bool
) -> None:
    async def fake_run(config: EvaluationConfig) -> tuple[Path, dict[str, object]]:
        if invalid_policy:
            raise EvaluationVerificationError(
                "Automatic policy metric deltas do not match requests"
            )
        return Path("synthetic-run"), {"run_status": "completed_with_failures"}

    monkeypatch.setattr(evaluation, "run_evaluation", fake_run)
    monkeypatch.setattr(sys, "argv", ["evaluation", "--strategy", "performance_v1"])
    with pytest.raises(SystemExit) as error:
        evaluation.main()
    assert error.value.code == 1
