"""Local, human-reviewed generated-answer evaluation for the curated collection."""

import argparse
import asyncio
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from kuberoute_gateway.knowledge_catalog import load_catalog
from kuberoute_gateway.knowledge_evaluation import load_questions
from kuberoute_gateway.rag.schemas import RagChatResponse

LABELS = ("pass", "fail")
REVIEW_KEYS = ("answer_correct", "citation_supported", "appropriate_abstention")


def _local_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "[::1]", "::1"}
        or parsed.username
        or parsed.password
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise argparse.ArgumentTypeError("Gateway URL must be a local HTTP origin")
    return value.rstrip("/")


async def collect(
    manifest: Path,
    questions_path: Path,
    *,
    base_url: str,
    model: str,
    top_k: int,
    max_tokens: int,
    seed: int | None,
) -> dict:
    sources = load_catalog(manifest)
    questions = load_questions(questions_path, {source.id for source in sources})
    rubric = json.loads(questions_path.read_text(encoding="utf-8"))["questions"]
    details = {record["id"]: record for record in rubric}
    rows = []
    async with httpx.AsyncClient(
        base_url=base_url, timeout=180.0, follow_redirects=False
    ) as client:
        for question in questions:
            response = await client.post(
                "/v1/rag/chat/completions",
                json={
                    "model": model,
                    "question": question.text,
                    "collection": "curated",
                    "retrieval_mode": "hybrid",
                    "top_k": top_k,
                    "temperature": 0,
                    "seed": seed,
                    "max_tokens": max_tokens,
                    "stream": False,
                },
            )
            response.raise_for_status()
            result = RagChatResponse.model_validate(response.json())
            rows.append(
                {
                    "id": question.id,
                    "question": question.text,
                    "expected_source_ids": list(question.expected_source_ids),
                    "answer_points": _question_detail(details[question.id], "answer_points"),
                    "caveats": _question_detail(details[question.id], "caveats"),
                    "sufficient_evidence": question.sufficient_evidence,
                    "answer": result.answer,
                    "model": result.model,
                    "finish_reason": result.finish_reason,
                    "cited_catalog_ids": [
                        source.catalog_id for source in result.sources if source.catalog_id
                    ],
                    "sources": [source.model_dump(exclude_none=True) for source in result.sources],
                    "review": {key: None for key in REVIEW_KEYS},
                }
            )
            print(f"Collected {question.id} ({len(rows)}/{len(questions)})", flush=True)
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_requested": model,
        "retrieval_mode": "hybrid",
        "top_k": top_k,
        "temperature": 0,
        "seed": seed,
        "max_tokens": max_tokens,
        "questions": rows,
    }


def _question_detail(record: dict, key: str) -> list[str]:
    # The checked-in rubric is public, but the existing loader intentionally only
    # retains fields needed for retrieval evaluation.
    detail = record[key]
    if (
        not isinstance(detail, list)
        or not detail
        or not all(isinstance(value, str) and value.strip() for value in detail)
    ):
        raise ValueError("Invalid review rubric")
    return detail


def summarize(run: dict) -> dict:
    if (
        not isinstance(run, dict)
        or run.get("schema_version") != 1
        or not isinstance(run.get("questions"), list)
    ):
        raise ValueError("Invalid answer evaluation run")
    rows = run["questions"]
    if (
        not rows
        or len(rows) > 100
        or any(not isinstance(row, dict) or not isinstance(row.get("id"), str) for row in rows)
        or len({row["id"] for row in rows}) != len(rows)
    ):
        raise ValueError("Invalid answer evaluation questions")
    result = {"total": len(rows), "answerable": 0, "unanswerable": 0}
    counts: dict[str, Counter] = {key: Counter() for key in REVIEW_KEYS}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("review"), dict):
            raise ValueError("Invalid answer evaluation row")
        sufficient = row.get("sufficient_evidence")
        if not isinstance(sufficient, bool) or set(row["review"]) != set(REVIEW_KEYS):
            raise ValueError("Invalid answer evaluation review")
        result["answerable" if sufficient else "unanswerable"] += 1
        for key in REVIEW_KEYS:
            label = row["review"][key]
            applicable = (key != "appropriate_abstention") == sufficient
            if label not in (*LABELS, None) or (not applicable and label is not None):
                raise ValueError("Invalid answer evaluation label")
            if applicable:
                counts[key][label if label is not None else "pending"] += 1
    result["review"] = {
        key: {
            "pass": value["pass"],
            "fail": value["fail"],
            "pending": value["pending"],
            "pass_rate": round(value["pass"] / (value["pass"] + value["fail"]), 4)
            if value["pass"] + value["fail"] and not value["pending"]
            else None,
        }
        for key, value in counts.items()
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect", help="Save local answers for manual review")
    collect_parser.add_argument("manifest", type=Path)
    collect_parser.add_argument("questions", type=Path)
    collect_parser.add_argument("--output", type=Path, required=True)
    collect_parser.add_argument("--base-url", type=_local_url, default="http://127.0.0.1:8001")
    collect_parser.add_argument("--model", default="llama3.2:3b")
    collect_parser.add_argument("--top-k", type=int, choices=range(1, 11), default=5)
    collect_parser.add_argument("--max-tokens", type=int, choices=range(1, 4097), default=384)
    collect_parser.add_argument("--seed", type=int, default=42)
    summary_parser = sub.add_parser("summarize", help="Count manually reviewed labels")
    summary_parser.add_argument("run", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "collect":
            if args.seed is not None and args.seed < 0:
                raise ValueError("Seed is invalid")
            if args.output.exists() or args.output.is_symlink():
                raise ValueError("Output already exists")
            run = asyncio.run(
                collect(
                    args.manifest,
                    args.questions,
                    base_url=args.base_url,
                    model=args.model,
                    top_k=args.top_k,
                    max_tokens=args.max_tokens,
                    seed=args.seed,
                )
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(run, stream, indent=2, ensure_ascii=False)
                stream.write("\n")
            print(f"Saved {len(run['questions'])} responses to {args.output}")
        else:
            run = json.loads(args.run.read_text(encoding="utf-8"))
            print(json.dumps(summarize(run), indent=2, sort_keys=True))
    except (OSError, ValueError, KeyError, TypeError, httpx.HTTPError) as exc:
        parser.exit(1, f"Answer evaluation failed: {type(exc).__name__}.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
