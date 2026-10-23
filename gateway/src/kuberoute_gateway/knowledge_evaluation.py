"""Small, local retrieval-only evaluation for an explicit source catalog."""

import argparse
import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from kuberoute_gateway.knowledge_catalog import CatalogError, load_catalog
from kuberoute_gateway.rag.embeddings import OllamaEmbeddingProvider
from kuberoute_gateway.rag.hybrid import HybridRetriever, HybridSearchConfig, RetrievalMode
from kuberoute_gateway.rag.ingestion import ingest_paths, load_documents

_LABEL = re.compile(r"[a-z][a-z0-9_]{2,63}\Z")


@dataclass(frozen=True)
class EvaluationQuestion:
    id: str
    text: str = field(repr=False)
    expected_source_ids: tuple[str, ...] = ()
    sufficient_evidence: bool = False


def load_questions(path: Path, catalog_ids: set[str]) -> tuple[EvaluationQuestion, ...]:
    """Validate a bounded synthetic question set without echoing its content."""
    try:
        with path.open("rb") as stream:
            raw = stream.read(131_073)
        if len(raw) > 131_072:
            raise CatalogError("Evaluation questions exceed the size limit")
        data = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise CatalogError("Evaluation questions are unavailable or invalid") from None
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise CatalogError("Evaluation questions are invalid")
    records = data.get("questions")
    if not isinstance(records, list) or not 1 <= len(records) <= 100:
        raise CatalogError("Evaluation question count is invalid")
    questions: list[EvaluationQuestion] = []
    for record in records:
        if not isinstance(record, dict):
            raise CatalogError("Evaluation question is invalid")
        question_id = record.get("id")
        text = record.get("question")
        expected = record.get("expected_source_ids")
        sufficient = record.get("sufficient_evidence")
        if (
            not isinstance(question_id, str)
            or not _LABEL.fullmatch(question_id)
            or not isinstance(text, str)
            or not text.strip()
            or len(text) > 2_000
            or not isinstance(expected, list)
            or not all(isinstance(item, str) for item in expected)
            or len(set(expected)) != len(expected)
            or not set(expected).issubset(catalog_ids)
            or not isinstance(sufficient, bool)
            or (sufficient and not expected)
            or (not sufficient and expected)
        ):
            raise CatalogError("Evaluation question is invalid")
        questions.append(EvaluationQuestion(question_id, text, tuple(expected), sufficient))
    if len({question.id for question in questions}) != len(questions):
        raise CatalogError("Evaluation question IDs are duplicated")
    return tuple(questions)


@dataclass(frozen=True)
class RetrievalSummary:
    mode: RetrievalMode
    questions: int
    answerable: int
    hit_at_k: int
    mean_source_recall: float
    mean_reciprocal_rank: float
    unanswerable_with_results: int


def summarize_retrieval(
    mode: RetrievalMode,
    observations: list[tuple[EvaluationQuestion, tuple[str, ...]]],
) -> RetrievalSummary:
    answerable = 0
    hits = 0
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    unanswerable_with_results = 0
    for question, retrieved in observations:
        if not question.sufficient_evidence:
            unanswerable_with_results += bool(retrieved)
            continue
        answerable += 1
        expected = set(question.expected_source_ids)
        found = expected.intersection(retrieved)
        hits += bool(found)
        recalls.append(len(found) / len(expected))
        reciprocal_ranks.append(
            next(
                (1 / rank for rank, source_id in enumerate(retrieved, 1) if source_id in expected),
                0.0,
            )
        )
    return RetrievalSummary(
        mode=mode,
        questions=len(observations),
        answerable=answerable,
        hit_at_k=hits,
        mean_source_recall=sum(recalls) / answerable if answerable else 0.0,
        mean_reciprocal_rank=sum(reciprocal_ranks) / answerable if answerable else 0.0,
        unanswerable_with_results=unanswerable_with_results,
    )


async def evaluate(
    manifest: Path,
    questions_path: Path,
    *,
    embedding_url: str,
    embedding_model: str,
    top_k: int = 5,
) -> tuple[RetrievalSummary, ...]:
    sources = load_catalog(manifest)
    questions = load_questions(questions_path, {source.id for source in sources})
    documents = load_documents([source.document for source in sources])
    catalog_by_hash = {
        document.source_id: source.id for document, source in zip(documents, sources, strict=True)
    }
    chunks = ingest_paths([source.document for source in sources])
    retriever = HybridRetriever(embedding_model)
    async with OllamaEmbeddingProvider(embedding_url, embedding_model) as provider:
        await retriever.add_chunks(chunks, provider)
        summaries: list[RetrievalSummary] = []
        for mode in ("hybrid", "semantic", "lexical"):
            observations: list[tuple[EvaluationQuestion, tuple[str, ...]]] = []
            for question in questions:
                results = await retriever.search(
                    question.text, provider, top_k=top_k, config=HybridSearchConfig(mode=mode)
                )
                selected = tuple(
                    dict.fromkeys(catalog_by_hash[result.source_id] for result in results)
                )
                observations.append((question, selected))
            summaries.append(summarize_retrieval(mode, observations))
    return tuple(summaries)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate local retrieval without generating answers"
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("questions", type=Path)
    parser.add_argument("--embedding-url", default="http://127.0.0.1:11434")
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument("--top-k", type=int, default=5, choices=range(1, 11), metavar="1..10")
    args = parser.parse_args()
    try:
        summaries = asyncio.run(
            evaluate(
                args.manifest,
                args.questions,
                embedding_url=args.embedding_url,
                embedding_model=args.embedding_model,
                top_k=args.top_k,
            )
        )
    except Exception:
        parser.exit(1, "Retrieval evaluation failed; check local catalog and embedding server.\n")
    for summary in summaries:
        print(json.dumps(summary.__dict__, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
