# Exercise science v1 retrieval check — 2026-10-23

The 12-document collection was indexed using the existing ingestion/chunking, Ollama `nomic-embed-text`, dense vector, BM25, and hybrid retriever components. The 18 synthetic questions in `questions.json` were submitted sequentially to each retrieval mode with Top-K 5. Sixteen questions have one or more expected catalog source IDs; two ask for individualized information not established by the collection. This is a single local functional check, not a benchmark of scientific accuracy.

| Retrieval mode | Questions processed | Answerable questions with ≥1 expected source in Top-K | Mean expected-source recall@5 | MRR@5 | Out-of-scope questions returning related passages |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hybrid | 18 | 16/16 | 1.00000 | 0.96875 | 2/2 |
| Semantic | 18 | 16/16 | 0.96875 | 0.93750 | 2/2 |
| Lexical/BM25 | 18 | 16/16 | 0.96875 | 0.93750 | 2/2 |

`hit@5` asks whether **any** expected document appeared; expected-source recall averages the fraction of each answerable question's expected documents retrieved. MRR uses the first expected document's rank. The two out-of-scope questions are excluded from those denominators. They returned top-ranked passages because a retriever ranks related text even when it lacks enough evidence for an individualized answer. These figures are favorable partly because the curated questions closely match topics in short source summaries; they do not establish recall on novel wording, factual grounding, or superiority of one retrieval mode. No generated-answer correctness or abstention rate was measured.

One separate in-process RAG smoke request used the configured local `qwen3:0.6b` model with hybrid retrieval and a short output limit. It returned HTTP 200 and nonempty answer text, but **zero returned citation references** and `finish_reason: length`. That response must not be called evidence-grounded. Its answer content was not saved. No additional generation trial or performance timing is inferred from it.

To repeat retrieval-only evaluation, with Ollama and the existing embedding model available locally:

```powershell
python -m kuberoute_gateway.knowledge_evaluation knowledge/exercise_science/manifest.json knowledge/exercise_science/evaluation/questions.json --top-k 5
```

The evaluator prints only aggregate counts and rates. It does not store prompts, document text, vectors, or generated answers. Future answer evaluation needs independent claim-level review against the cited original source, including limits and conflicts; marker validity by itself is insufficient.
