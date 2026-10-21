"""Sequential, checkpointed evaluation of fixed and automatic gateway routing."""

import argparse
import asyncio
import csv
import hashlib
import json
import math
import re
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

import httpx
from prometheus_client.parser import text_string_to_metric_families
from pydantic import ValidationError

from kuberoute_gateway.benchmark import BenchmarkConfig, _percentile
from kuberoute_gateway.evaluation_tasks import (
    TASK_SET_VERSION,
    TASKS,
    WARMUP_TASK,
    EvaluationTask,
    task_for_sequence,
)
from kuberoute_gateway.evaluation_tasks_v2 import TASK_SET_V2_VERSION, TASKS_V2
from kuberoute_gateway.schemas import ChatCompletionResponse

Strategy = Literal[
    "fixed_qwen3", "fixed_llama3_2", "fixed_gemma3", "performance_v1", "first_eligible"
]
ChoiceStatus = Literal["correct", "incorrect", "ambiguous", "not_applicable"]
OUTPUT_SCHEMA_VERSION = 2
GRADING_VERSION = "exact-plus-choice-v2"
MODEL_IDS = ("qwen3:0.6b", "llama3.2:3b", "gemma3:4b")
STRATEGY_MODELS: dict[Strategy, str] = {
    "fixed_qwen3": MODEL_IDS[0],
    "fixed_llama3_2": MODEL_IDS[1],
    "fixed_gemma3": MODEL_IDS[2],
    "performance_v1": "auto",
    "first_eligible": "auto",
}

_MAX_CHOICE_TEXT_LENGTH = 512
_CHOICE_FORMATS = (
    re.compile(r"([ABC])[.!?]?", re.IGNORECASE),
    re.compile(r"\(([ABC])\)[.!?]?", re.IGNORECASE),
    re.compile(
        r"(?:Answer\s*:\s*|Option\s+|The\s+(?:correct\s+)?answer\s+is\s+)"
        r"([ABC])[.!?]?",
        re.IGNORECASE,
    ),
)


def classify_answer_choice(content: str, expected_choice: str) -> ChoiceStatus:
    """Identify only bounded, explicitly selected A/B/C answers."""
    if expected_choice not in {"A", "B", "C"} or len(content) > _MAX_CHOICE_TEXT_LENGTH:
        return "ambiguous"
    response = content.strip()
    if not response:
        return "ambiguous"
    for pattern in _CHOICE_FORMATS:
        if match := pattern.fullmatch(response):
            return "correct" if match.group(1).upper() == expected_choice else "incorrect"
    return "ambiguous"


@dataclass(frozen=True)
class EvaluationConfig:
    strategy: Strategy
    gateway_url: str = "http://127.0.0.1:8000"
    warmup: int = 2
    requests: int = 10
    max_output_tokens: int = 64
    timeout_seconds: float = 180.0
    output_dir: Path = Path("evaluation-results")
    task_set_version: Literal["1", "2"] = "1"

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGY_MODELS:
            raise ValueError("Evaluation strategy is invalid")
        if self.task_set_version not in (TASK_SET_VERSION, TASK_SET_V2_VERSION):
            raise ValueError("Task-set version must be 1 or 2")
        if (
            isinstance(self.warmup, bool)
            or not isinstance(self.warmup, int)
            or not 0 <= self.warmup <= 20
            or isinstance(self.requests, bool)
            or not isinstance(self.requests, int)
            or not 1 <= self.requests <= 100
        ):
            raise ValueError("Warm-up must be 0 to 20 and measured requests 1 to 100")
        if self.task_set_version == TASK_SET_V2_VERSION and self.requests > len(TASKS_V2):
            raise ValueError("Task-set version 2 allows at most 60 measured requests")
        if (
            isinstance(self.max_output_tokens, bool)
            or not isinstance(self.max_output_tokens, int)
            or not 1 <= self.max_output_tokens <= 128
        ):
            raise ValueError("Output-token limit must be 1 to 128")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or not 0 < self.timeout_seconds <= 600
        ):
            raise ValueError("Request timeout must be greater than 0 and at most 600 seconds")
        try:
            url = httpx.URL(self.gateway_url)
            valid = (
                url.scheme in ("http", "https")
                and bool(url.host)
                and not url.userinfo
                and not url.query
                and not url.fragment
                and url.path == "/"
            )
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError("Gateway URL must be an HTTP(S) origin without credentials")
        if not isinstance(self.output_dir, Path):
            raise ValueError("Output directory must be a path")

    @property
    def requested_model(self) -> str:
        return STRATEGY_MODELS[self.strategy]

    @property
    def endpoint(self) -> str:
        return f"{self.gateway_url.rstrip('/')}/v1/chat/completions"

    def payload(self, task: EvaluationTask) -> dict[str, object]:
        """Reuse the existing gateway benchmark's generation-control translation."""
        return BenchmarkConfig(
            mode="gateway",
            model=self.requested_model,
            prompt=task.prompt,
            temperature=0.0,
            seed=42,
            max_output_tokens=self.max_output_tokens,
            disable_thinking=True,
            timeout_seconds=self.timeout_seconds,
            gateway_url=self.gateway_url,
        ).payload


@dataclass(frozen=True)
class EvaluationResult:
    phase: Literal["warmup", "measurement"]
    sequence: int
    strategy: Strategy
    task_id: str
    category: str
    requested_model: str
    selected_model: str | None
    http_status: int | None
    success: bool
    error_category: str | None
    latency_seconds: float
    prompt_tokens: int | None
    output_tokens: int | None
    finish_reason: str | None
    content_nonempty: bool
    correct: bool | None
    choice_status: ChoiceStatus

    def csv_row(self) -> dict[str, object]:
        return asdict(self)


RESULT_FIELDS = tuple(item.name for item in fields(EvaluationResult))


class EvaluationVerificationError(RuntimeError):
    """The declared automatic strategy could not be verified safely."""


async def _inference_counters(
    client: httpx.AsyncClient, config: EvaluationConfig
) -> dict[tuple[str, str, str], float]:
    """Read bounded remote-call counters without retaining metric exposition."""
    try:
        async with asyncio.timeout(min(config.timeout_seconds, 10.0)):
            response = await client.get(f"{config.gateway_url.rstrip('/')}/metrics")
            if response.status_code != 200:
                raise ValueError("Metrics endpoint unavailable")
            families = text_string_to_metric_families(response.text)
            counters: dict[tuple[str, str, str], float] = {}
            found = False
            for family in families:
                if family.name != "kuberoute_inference_requests":
                    continue
                found = True
                for sample in family.samples:
                    if sample.name != "kuberoute_inference_requests_total":
                        continue
                    labels = sample.labels
                    if set(labels) != {"model", "backend_kind", "policy", "outcome"}:
                        raise ValueError("Inference metric labels are invalid")
                    if labels["backend_kind"] != "remote":
                        continue
                    key = (labels["policy"], labels["model"], labels["outcome"])
                    value = float(sample.value)
                    if key in counters or not math.isfinite(value) or value < 0:
                        raise ValueError("Inference metric counter is invalid")
                    counters[key] = value
            if not found:
                raise ValueError("Inference metric family is missing")
            return counters
    except Exception:
        raise EvaluationVerificationError(
            "Automatic policy metrics are unavailable or invalid"
        ) from None


def _verify_automatic_policy(
    strategy: Strategy,
    before: dict[tuple[str, str, str], float],
    after: dict[tuple[str, str, str], float],
    results: list[EvaluationResult],
) -> None:
    """Require the exact selected-model and policy counter deltas for this run."""
    expected = Counter((strategy, item.selected_model, "success") for item in results)
    for key in before.keys() | after.keys() | expected.keys():
        if after.get(key, 0.0) - before.get(key, 0.0) != expected.get(key, 0):
            raise EvaluationVerificationError(
                "Automatic policy metric deltas do not match requests"
            )


async def measure_evaluation_request(
    client: httpx.AsyncClient,
    config: EvaluationConfig,
    task: EvaluationTask,
    phase: Literal["warmup", "measurement"],
    sequence: int,
) -> EvaluationResult:
    """Retain response text only long enough for both bounded grading methods."""
    started = time.perf_counter()
    status: int | None = None
    selected: str | None = None
    error: str | None = None
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    content_nonempty = False
    correct: bool | None = None
    choice_status: ChoiceStatus = "not_applicable"
    try:
        async with asyncio.timeout(config.timeout_seconds):
            response = await client.post(config.endpoint, json=config.payload(task))
            status = response.status_code
            if status != 200:
                error = "timeout" if status in (408, 504) else "http_error"
            else:
                data = response.json()
                if not isinstance(data, dict) or not isinstance(data.get("choices"), list):
                    raise ValueError("Invalid completion")
                if len(data["choices"]) != 1:
                    raise ValueError("Exactly one choice is required")
                completion = ChatCompletionResponse.model_validate(data)
                choice = completion.choices[0]
                if choice.index != 0 or completion.model not in MODEL_IDS:
                    raise ValueError("Unsupported completion")
                selected = completion.model
                finish_reason = choice.finish_reason
                if completion.usage is not None:
                    prompt_tokens = completion.usage.prompt_tokens
                    output_tokens = completion.usage.completion_tokens
                content = choice.message.content
                content_nonempty = bool(content.strip())
                if config.requested_model != "auto" and selected != config.requested_model:
                    error = "wrong_model"
                elif not content_nonempty:
                    error = "length_exhausted" if finish_reason == "length" else "empty_content"
                elif finish_reason == "length":
                    error = "generation_length"
                elif task.expected_choice is not None and phase == "measurement":
                    correct = content.strip().upper() == task.expected_choice
                    choice_status = classify_answer_choice(content, task.expected_choice)
    except (TimeoutError, httpx.TimeoutException):
        error = "timeout"
    except httpx.RequestError:
        error = "network_error"
    except (ValueError, ValidationError, KeyError, IndexError, TypeError):
        error = "invalid_response"
    return EvaluationResult(
        phase=phase,
        sequence=sequence,
        strategy=config.strategy,
        task_id=task.task_id,
        category=task.category,
        requested_model=config.requested_model,
        selected_model=selected,
        http_status=status,
        success=error is None,
        error_category=error,
        latency_seconds=time.perf_counter() - started,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        finish_reason=finish_reason,
        content_nonempty=content_nonempty,
        correct=correct,
        choice_status=choice_status,
    )


def summarize_evaluation(strategy: Strategy, results: list[EvaluationResult]) -> dict[str, object]:
    """Aggregate measured requests only; missing usage is never imputed."""
    successful = [item for item in results if item.success]
    latencies = [item.latency_seconds for item in successful]
    with_usage = [item for item in successful if item.output_tokens is not None]
    known_latency = sum(item.latency_seconds for item in with_usage)
    graded = [item for item in successful if item.correct is not None]
    choice_eligible = graded
    if any(
        item.choice_status not in {"correct", "incorrect", "ambiguous"} for item in choice_eligible
    ):
        raise ValueError("Successful graded requests require a valid choice status")
    choice_counts = Counter(item.choice_status for item in choice_eligible)
    identified_choices = choice_counts["correct"] + choice_counts["incorrect"]
    count = len(results)
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "grading_version": GRADING_VERSION,
        "strategy": strategy,
        "requests": count,
        "successful_requests": len(successful),
        "failed_requests": count - len(successful),
        "timeout_requests": sum(item.error_category == "timeout" for item in results),
        "success_rate": len(successful) / count if count else 0.0,
        "failure_rate": (count - len(successful)) / count if count else 0.0,
        "latency_sample_count": len(latencies),
        "mean_latency_seconds": statistics.mean(latencies) if latencies else None,
        "p50_latency_seconds": _percentile(latencies, 0.5),
        "p95_latency_seconds": _percentile(latencies, 0.95),
        "model_selection_distribution": dict(
            sorted(Counter(item.selected_model for item in results if item.selected_model).items())
        ),
        "error_categories": dict(
            sorted(Counter(item.error_category for item in results if item.error_category).items())
        ),
        "objectively_scored_requests": len(graded),
        "objectively_correct_requests": sum(item.correct is True for item in graded),
        "objective_accuracy": sum(item.correct is True for item in graded) / len(graded)
        if graded
        else None,
        "choice_eligible_requests": len(choice_eligible),
        "choice_correct_requests": choice_counts["correct"],
        "choice_incorrect_requests": choice_counts["incorrect"],
        "choice_ambiguous_requests": choice_counts["ambiguous"],
        "choice_identified_requests": identified_choices,
        "choice_identification_coverage": identified_choices / len(choice_eligible)
        if choice_eligible
        else None,
        "choice_identification_accuracy": choice_counts["correct"] / identified_choices
        if identified_choices
        else None,
        "requests_with_output_usage": len(with_usage),
        "output_usage_coverage": len(with_usage) / len(successful) if successful else None,
        "reported_output_tokens": sum(item.output_tokens or 0 for item in with_usage),
        "token_adjusted_throughput_tokens_per_second": sum(
            item.output_tokens or 0 for item in with_usage
        )
        / known_latency
        if with_usage and known_latency > 0
        else None,
    }


async def run_evaluation(
    config: EvaluationConfig, client: httpx.AsyncClient | None = None
) -> tuple[Path, dict[str, object]]:
    """Write each completed row immediately so interrupted runs keep evidence."""
    task_set = TASKS if config.task_set_version == TASK_SET_VERSION else TASKS_V2
    measured_tasks = (
        [task_for_sequence(sequence) for sequence in range(config.requests)]
        if config.task_set_version == TASK_SET_VERSION
        else list(task_set[: config.requests])
    )
    task_set_bytes = json.dumps(
        [asdict(task) for task in task_set], sort_keys=True, ensure_ascii=True
    ).encode("utf-8")
    run_dir = config.output_dir / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "grading_version": GRADING_VERSION,
        "task_set_version": config.task_set_version,
        "task_ids": [task.task_id for task in task_set],
        "measured_task_ids": [task.task_id for task in measured_tasks],
        "task_set_sha256": hashlib.sha256(task_set_bytes).hexdigest(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "strategy": config.strategy,
        "requested_model": config.requested_model,
        "warmup": config.warmup,
        "requests": config.requests,
        "max_output_tokens": config.max_output_tokens,
        "timeout_seconds": config.timeout_seconds,
        "generation_controls": {
            "temperature": 0.0,
            "seed": 42,
            "max_tokens": config.max_output_tokens,
            "reasoning_effort": "none",
            "stream": False,
        },
        "gateway_url_recorded": False,
        "run_status": "in_progress",
        "policy_verification": "pending" if config.requested_model == "auto" else "not_applicable",
    }

    def write_metadata() -> None:
        (run_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

    write_metadata()
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=config.timeout_seconds)
    try:
        before = (
            await _inference_counters(client, config) if config.requested_model == "auto" else {}
        )
        with (run_dir / "requests.csv").open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=RESULT_FIELDS)
            writer.writeheader()
            output.flush()
            warmups: list[EvaluationResult] = []
            for sequence in range(config.warmup):
                result = await measure_evaluation_request(
                    client, config, WARMUP_TASK, "warmup", sequence
                )
                warmups.append(result)
                writer.writerow(result.csv_row())
                output.flush()
            measured: list[EvaluationResult] = []
            for sequence, task in enumerate(measured_tasks):
                result = await measure_evaluation_request(
                    client, config, task, "measurement", sequence
                )
                measured.append(result)
                writer.writerow(result.csv_row())
                output.flush()
            all_results = warmups + measured
            if config.requested_model == "auto":
                if all(item.success for item in all_results):
                    after = await _inference_counters(client, config)
                    _verify_automatic_policy(config.strategy, before, after, all_results)
                    verification = "verified"
                else:
                    verification = "unverified_due_to_request_failures"
            else:
                verification = "not_applicable"
            run_status = (
                "completed"
                if all(item.success for item in all_results)
                else "completed_with_failures"
            )
            summary = summarize_evaluation(config.strategy, measured)
            summary["warmup_requests"] = len(warmups)
            summary["warmup_failed_requests"] = sum(not item.success for item in warmups)
            summary["run_status"] = run_status
            summary["policy_verification"] = verification
            metadata["run_status"] = run_status
            metadata["policy_verification"] = verification
            write_metadata()
            (run_dir / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            return run_dir, summary
    except EvaluationVerificationError:
        metadata["run_status"] = "invalid"
        metadata["policy_verification"] = "unverified"
        write_metadata()
        raise
    except BaseException:
        metadata["run_status"] = "interrupted"
        write_metadata()
        raise
    finally:
        if owns_client:
            await client.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fixed and automatic gateway routing")
    parser.add_argument("--strategy", choices=tuple(STRATEGY_MODELS), required=True)
    parser.add_argument("--task-set-version", choices=("1", "2"), default="1")
    parser.add_argument("--gateway-url", default="http://127.0.0.1:8000")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--max-output-tokens", type=int, default=64)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation-results"))
    args = parser.parse_args()
    try:
        config = EvaluationConfig(**vars(args))
    except ValueError as exc:
        parser.error(str(exc))
    try:
        run_dir, summary = asyncio.run(run_evaluation(config))
    except EvaluationVerificationError as exc:
        parser.exit(1, f"Evaluation invalid: {exc}\n")
    print(json.dumps({"run_dir": str(run_dir), "summary": summary}, indent=2))
    if summary["run_status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
