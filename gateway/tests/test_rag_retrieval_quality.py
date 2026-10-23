"""Small deterministic retrieval checks over harmless technical documents."""

import asyncio

from kuberoute_gateway.rag import Chunk, HybridRetriever, HybridSearchConfig, SourceMetadata

DOCUMENTS = (
    ("deployments", "Kubernetes Deployments replace failed Pods to maintain replicas."),
    ("timeouts", "HTTP 504 means upstream inference exceeded its timeout deadline."),
    ("metrics", "Prometheus counters track requests and histograms measure latency."),
    ("routing", "Adaptive routing selects models using latency and failure observations."),
    ("embeddings", "Embedding models map text into dense vectors for similarity search."),
)
QUESTIONS = (
    ("How are crashed workloads restored?", "deployments"),
    ("What does HTTP 504 indicate?", "timeouts"),
    ("Which system tracks request counts and latency?", "metrics"),
    ("What chooses models using failure observations?", "routing"),
    ("How are sentences mapped to numerical features?", "embeddings"),
)


class DeterministicEmbeddings:
    model_id = "synthetic-embeddings"

    def __init__(self, vectors: dict[str, tuple[float, ...]]):
        self.vectors = vectors

    async def embed(self, texts):
        return tuple(self.vectors[text] for text in texts)


def test_small_synthetic_retrieval_recall_and_mrr() -> None:
    async def run() -> None:
        chunks = tuple(
            Chunk(name, name, 0, content, SourceMetadata(".md", len(content)))
            for name, content in DOCUMENTS
        )
        expected_ids = {name: name for name, _ in DOCUMENTS}
        vectors: dict[str, tuple[float, ...]] = {}
        for index, ((_, content), (question, _)) in enumerate(
            zip(DOCUMENTS, QUESTIONS, strict=True)
        ):
            vector = tuple(float(position == index) for position in range(len(DOCUMENTS)))
            vectors[content] = vector
            vectors[question] = vector
        provider = DeterministicEmbeddings(vectors)
        retriever = HybridRetriever(provider.model_id)
        await retriever.add_chunks(chunks, provider)

        observations = {}
        for mode in ("semantic", "lexical", "hybrid"):
            reciprocal_ranks = []
            first_hits = 0
            for question, expected in QUESTIONS:
                results = await retriever.search(
                    question,
                    provider if mode != "lexical" else None,
                    top_k=3,
                    config=HybridSearchConfig(mode=mode),
                )
                ranked_ids = [result.chunk_id for result in results]
                target = expected_ids[expected]
                rank = ranked_ids.index(target) + 1 if target in ranked_ids else None
                first_hits += rank == 1
                reciprocal_ranks.append(1 / rank if rank is not None else 0)
            observations[mode] = (first_hits / len(QUESTIONS), sum(reciprocal_ranks) / 5)

        assert observations["semantic"] == (1.0, 1.0)
        assert observations["hybrid"] == (0.8, 0.9)
        assert observations["lexical"] == (0.6, 0.6)

    asyncio.run(run())
