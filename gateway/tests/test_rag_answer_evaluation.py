"""Check review denominators and local-only evaluation requests."""

import argparse
import asyncio
import json
from pathlib import Path

import httpx
import pytest
from kuberoute_gateway.rag_answer_evaluation import _local_url, collect, summarize

ROOT = Path(__file__).resolve().parents[2]
COLLECTION = ROOT / "knowledge" / "exercise_science"


def test_review_rates_require_complete_human_labels() -> None:
    run = {
        "schema_version": 1,
        "questions": [
            {
                "id": "es01",
                "sufficient_evidence": True,
                "review": {
                    "answer_correct": "pass",
                    "citation_supported": "fail",
                    "appropriate_abstention": None,
                },
            },
            {
                "id": "es02",
                "sufficient_evidence": True,
                "review": {
                    "answer_correct": None,
                    "citation_supported": None,
                    "appropriate_abstention": None,
                },
            },
            {
                "id": "es17",
                "sufficient_evidence": False,
                "review": {
                    "answer_correct": None,
                    "citation_supported": None,
                    "appropriate_abstention": "pass",
                },
            },
        ],
    }
    summary = summarize(run)
    assert (summary["total"], summary["answerable"], summary["unanswerable"]) == (3, 2, 1)
    assert summary["review"]["answer_correct"] == {
        "pass": 1,
        "fail": 0,
        "pending": 1,
        "pass_rate": None,
    }
    assert summary["review"]["citation_supported"]["pass_rate"] is None
    assert summary["review"]["appropriate_abstention"]["pass_rate"] == 1.0
    run["questions"][1]["review"]["answer_correct"] = "fail"
    run["questions"][1]["review"]["citation_supported"] = "pass"
    summary = summarize(run)
    assert summary["review"]["answer_correct"]["pass_rate"] == 0.5
    assert summary["review"]["citation_supported"]["pass_rate"] == 0.5
    run["questions"][2]["review"]["citation_supported"] = "pass"
    with pytest.raises(ValueError, match="label"):
        summarize(run)


def test_evaluation_restricts_gateway_to_loopback() -> None:
    assert _local_url("http://127.0.0.1:8001/") == "http://127.0.0.1:8001"
    for url in (
        "https://127.0.0.1:8001",
        "http://example.org",
        "http://user:password@localhost:8001",
        "http://localhost:8001/other",
    ):
        with pytest.raises(argparse.ArgumentTypeError):
            _local_url(url)


def test_collector_sends_curated_requests_and_records_review_fields(monkeypatch) -> None:
    requested = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requested.append(body)
        return httpx.Response(
            200,
            json={
                "answer": "Synthetic answer [1]",
                "model": "llama3.2:3b",
                "sources": [
                    {
                        "marker": "[1]",
                        "source_id": "a" * 64,
                        "chunk_id": "b" * 64,
                        "chunk_index": 0,
                        "catalog_id": "schoenfeld_2017_volume",
                    }
                ],
                "retrieval_mode": "hybrid",
                "finish_reason": "stop",
            },
        )

    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original_client(transport=httpx.MockTransport(respond), **kwargs),
    )
    run = asyncio.run(
        collect(
            COLLECTION / "manifest.json",
            COLLECTION / "evaluation" / "questions.json",
            base_url="http://127.0.0.1:8001",
            model="llama3.2:3b",
            top_k=5,
            max_tokens=384,
            seed=42,
        )
    )
    assert len(requested) == len(run["questions"]) == 18
    assert all(
        request["collection"] == "curated"
        and request["temperature"] == 0
        and request["retrieval_mode"] == "hybrid"
        for request in requested
    )
    assert run["questions"][0]["review"]["answer_correct"] is None
    assert run["questions"][0]["cited_catalog_ids"] == ["schoenfeld_2017_volume"]
    assert run["questions"][0]["answer_points"]
