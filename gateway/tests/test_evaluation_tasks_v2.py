"""Independent task-key checks and mocked v2 evaluation runs."""

import asyncio
import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import httpx
import pytest
from kuberoute_gateway.evaluation import (
    GRADING_VERSION,
    OUTPUT_SCHEMA_VERSION,
    EvaluationConfig,
    run_evaluation,
)
from kuberoute_gateway.evaluation_tasks import TASK_SET_VERSION, TASKS, task_for_sequence
from kuberoute_gateway.evaluation_tasks_v2 import (
    CATEGORIES_V2,
    SPECS_V2,
    TASK_SET_V2_VERSION,
    TASKS_V2,
)


def task_hash(tasks: tuple) -> str:
    return hashlib.sha256(
        json.dumps([asdict(task) for task in tasks], sort_keys=True, ensure_ascii=True).encode(
            "utf-8"
        )
    ).hexdigest()


def test_task_distribution_pilot_balance_and_unique_ids() -> None:
    assert TASK_SET_VERSION == "1"
    assert TASK_SET_V2_VERSION == "2"
    assert len(TASKS_V2) == 60
    assert len({task.task_id for task in TASKS_V2}) == 60
    assert len({task.prompt for task in TASKS_V2}) == 60
    assert not {task.task_id for task in TASKS_V2} & {task.task_id for task in TASKS}
    assert Counter(task.category for task in TASKS_V2) == dict.fromkeys(CATEGORIES_V2, 12)
    assert Counter(task.expected_choice for task in TASKS_V2) == dict.fromkeys("ABC", 20)
    for category in CATEGORIES_V2:
        category_tasks = [task for task in TASKS_V2 if task.category == category]
        assert Counter(task.expected_choice for task in category_tasks) == dict.fromkeys("ABC", 4)
    assert Counter(task.category for task in TASKS_V2[:15]) == dict.fromkeys(CATEGORIES_V2, 3)
    assert Counter(task.expected_choice for task in TASKS_V2[:15]) == dict.fromkeys("ABC", 5)
    assert [task.category for task in TASKS_V2[:5]] == list(CATEGORIES_V2)
    assert task_hash(TASKS) == "9a617827b7176cad3f9d4b2a0482bfedd7e2858fad1f22ed1d82711361f47de0"
    assert task_hash(TASKS_V2) == "5ffc2035d026dd417011d2f983cc860c48169680b6f1bc25db495e0732ecfa37"


def test_task_fingerprint_covers_prompts_keys_and_order() -> None:
    original = task_hash(TASKS_V2)
    assert original == task_hash(tuple(TASKS_V2))
    assert task_hash((replace(TASKS_V2[0], prompt="Changed prompt"), *TASKS_V2[1:])) != original
    assert task_hash((replace(TASKS_V2[0], expected_choice="B"), *TASKS_V2[1:])) != original
    assert task_hash((TASKS_V2[1], TASKS_V2[0], *TASKS_V2[2:])) != original


def test_task_prompts_have_three_distinct_options_and_source_audit_notes() -> None:
    for round_index in range(12):
        for category_index, category in enumerate(CATEGORIES_V2):
            spec = SPECS_V2[category][round_index]
            task = TASKS_V2[round_index * 5 + category_index]
            options = list(spec.distractors)
            options.insert((round_index + category_index) % 3, spec.answer)
            assert len(set(options)) == 3
            assert task.expected_choice == "ABC"[options.index(spec.answer)]
            assert task.prompt.startswith(spec.question)
            assert all(
                f"{letter}: {answer}." in task.prompt
                for letter, answer in zip("ABC", options, strict=True)
            )
            assert all(task.prompt.count(f" {letter}: ") == 1 for letter in "ABC")
            assert task.prompt.endswith("Reply with only the single answer letter.")
            assert len(task.prompt) <= 330
            if category in {"logical_reasoning", "reading_comprehension"}:
                assert spec.audit_note
    assert len({spec.question for group in SPECS_V2.values() for spec in group}) == 60


def test_arithmetic_keys_against_independent_calculations() -> None:
    from datetime import datetime, timedelta

    clock = (datetime(2020, 1, 1, 14, 35) + timedelta(minutes=50)).strftime("%H:%M")
    reference = {
        "v2_ar_add": str(17 + 8),
        "v2_ar_subtract": str(50 - 19),
        "v2_ar_multiply": str(6 * 7),
        "v2_ar_divide": str(81 // 9),
        "v2_ar_remainder": str(23 % 5),
        "v2_ar_parentheses": str((4 + 6) * 3),
        "v2_ar_average": str(sum((8, 12, 16)) // 3),
        "v2_ar_percent": str(80 * 25 // 100),
        "v2_ar_fraction": str(28 * 3 // 4),
        "v2_ar_negative": str(-7 + 12),
        "v2_ar_precedence": str(2 + 3 * 4),
        "v2_ar_clock": clock,
    }
    assert set(reference) == {spec.task_id for spec in SPECS_V2["arithmetic"]}
    for spec in SPECS_V2["arithmetic"]:
        assert spec.answer == reference[spec.task_id]


def test_numerical_keys_against_independent_comparisons() -> None:
    from fractions import Fraction

    reference = {
        "v2_num_decimal": str(max((0.705, 0.7, 0.65))),
        "v2_num_negative": str(min((-8, -3, -1))),
        "v2_num_fraction": str(max((Fraction(4, 5), Fraction(3, 4), Fraction(2, 3)))),
        "v2_num_length": f"{max((1.5, 1.4, 1.6))} m",
        "v2_num_time": max(("09:45", "10:05", "09:55")),
        "v2_num_absolute": f"|-{max((9, 7, 8))}|",
        "v2_num_powers": "2 to the fifth power" if 2**5 > max(3**3, 5**2) else "invalid",
        "v2_num_closest": str(min((1.85, 1.7, 2.4), key=lambda x: abs(2 - x))),
        "v2_num_duration": f"{min((45, 60, 48))} seconds",
        "v2_num_equivalent": "19/20" if Fraction(19, 20) == Fraction(95, 100) else "invalid",
        "v2_num_median": str(sorted((408, 480, 418))[1]),
        "v2_num_gap": "7 and 10" if abs(7 - 10) < min(abs(12 - 18), abs(20 - 24)) else "invalid",
    }
    assert set(reference) == {spec.task_id for spec in SPECS_V2["numerical_comparison"]}
    for spec in SPECS_V2["numerical_comparison"]:
        assert spec.answer == reference[spec.task_id]


def test_string_keys_against_independent_operations() -> None:
    reference = {
        "v2_str_upper": str(sum(char.isupper() for char in "aBcDEf")),
        "v2_str_reverse": "staple"[::-1],
        "v2_str_sort": sorted("pear")[0],
        "v2_str_spaces": "red fox".replace(" ", ""),
        "v2_str_vowels": str(sum(char in "aeiou" for char in "cooperate")),
        "v2_str_suffix": "lantern"[-3:],
        "v2_str_join": "-".join(("north", "star")),
        "v2_str_index": str("coconut".index("o")),
        "v2_str_replace": "banana".replace("a", "o"),
        "v2_str_distinct": str(len(set("level"))),
        "v2_str_lexical": min(("kitten", "kitchen", "zebra")),
        "v2_str_alternate": "abcdefg"[1::2],
    }
    assert set(reference) == {spec.task_id for spec in SPECS_V2["string_operations"]}
    for spec in SPECS_V2["string_operations"]:
        assert spec.answer == reference[spec.task_id]


def test_v1_cycle_is_unchanged_and_v2_limit_is_validated_before_output(tmp_path: Path) -> None:
    assert [task_for_sequence(i).task_id for i in range(10)] == [task.task_id for task in TASKS] * 2
    assert EvaluationConfig(strategy="fixed_qwen3").task_set_version == "1"
    old_positional = EvaluationConfig(
        "fixed_qwen3", "http://127.0.0.1:8000", 0, 5, 64, 180.0, tmp_path
    )
    assert old_positional.task_set_version == "1"
    assert old_positional.requests == 5
    assert old_positional.output_dir == tmp_path
    EvaluationConfig(strategy="fixed_qwen3", task_set_version="2", requests=60)
    with pytest.raises(ValueError, match="at most 60"):
        EvaluationConfig(
            strategy="fixed_qwen3", task_set_version="2", requests=61, output_dir=tmp_path
        )
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError, match="version must be 1 or 2"):
        EvaluationConfig(strategy="fixed_qwen3", task_set_version="3")


@pytest.mark.parametrize("request_count", [1, 15, 60])
def test_v2_run_records_no_repeats_private_metadata_and_checkpointed_rows(
    tmp_path: Path, request_count: int
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        payload = json.loads(request.content)
        assert payload["model"] == "qwen3:0.6b"
        assert payload["temperature"] == 0.0
        assert payload["seed"] == 42
        assert payload["stream"] is False
        assert payload["reasoning_effort"] == "none"
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "created": 1,
                "model": "qwen3:0.6b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Answer: A"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    async def run() -> None:
        config = EvaluationConfig(
            strategy="fixed_qwen3",
            task_set_version="2",
            requests=request_count,
            warmup=2,
            output_dir=tmp_path,
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            run_dir, summary = await run_evaluation(config, client)
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        with (run_dir / "requests.csv").open(newline="", encoding="utf-8") as output:
            rows = list(csv.DictReader(output))
        assert metadata["task_set_version"] == "2"
        assert metadata["schema_version"] == OUTPUT_SCHEMA_VERSION
        assert metadata["grading_version"] == GRADING_VERSION
        assert metadata["measured_task_ids"] == [task.task_id for task in TASKS_V2[:request_count]]
        assert len(metadata["task_ids"]) == 60
        assert len(metadata["task_set_sha256"]) == 64
        assert metadata["task_set_sha256"] == task_hash(TASKS_V2)
        assert [row["task_id"] for row in rows[:2]] == ["warmup_ack"] * 2
        assert [row["task_id"] for row in rows[2:]] == metadata["measured_task_ids"]
        assert len(set(row["task_id"] for row in rows[2:])) == request_count
        assert all(row["phase"] == "measurement" for row in rows[2:])
        assert summary["requests"] == request_count
        assert summary["run_status"] == "completed"
        assert summary["objective_accuracy"] == 0.0
        assert summary["choice_correct_requests"] == sum(
            task.expected_choice == "A" for task in TASKS_V2[:request_count]
        )
        assert summary["choice_identification_coverage"] == 1.0
        saved = "".join(path.read_text(encoding="utf-8") for path in run_dir.iterdir())
        assert "Answer: A" not in saved
        assert all(task.prompt not in saved for task in TASKS_V2)
        assert "http://127.0.0.1" not in saved

    asyncio.run(run())


def test_interrupted_v2_run_keeps_rows_and_marks_metadata(tmp_path: Path) -> None:
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("transient private diagnostic")
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "created": 1,
                "model": "qwen3:0.6b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "B"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    async def run() -> None:
        config = EvaluationConfig(
            strategy="fixed_qwen3", task_set_version="2", warmup=0, requests=4, output_dir=tmp_path
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            with pytest.raises(RuntimeError, match="transient private diagnostic"):
                await run_evaluation(config, client)
        run_dir = next(tmp_path.iterdir())
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        with (run_dir / "requests.csv").open(newline="", encoding="utf-8") as output:
            rows = list(csv.DictReader(output))
        assert metadata["run_status"] == "interrupted"
        assert metadata["measured_task_ids"] == [task.task_id for task in TASKS_V2[:4]]
        assert len(rows) == 2
        assert not (run_dir / "summary.json").exists()
        assert "transient private diagnostic" not in (run_dir / "metadata.json").read_text(
            encoding="utf-8"
        )

    asyncio.run(run())
