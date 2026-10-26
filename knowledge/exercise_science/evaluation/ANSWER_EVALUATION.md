# Generated-answer review

This workflow complements the retrieval-only check in `RESULTS.md`. It collects
the 18 checked-in questions against the **curated** collection, using hybrid
retrieval at Top-K 5, temperature 0, seed 42, and the chosen local model. It
does not grade answer meaning automatically. Temperature and seed settings may
improve repeatability, but local model output is not guaranteed to be identical.

Start the configured gateway with RAG indexing and Ollama running. In a second
PowerShell window at the repository root, with the Python virtual environment
active, run:

```powershell
python -m kuberoute_gateway.rag_answer_evaluation collect `
  knowledge/exercise_science/manifest.json `
  knowledge/exercise_science/evaluation/questions.json `
  --model llama3.2:3b `
  --output evaluation-results/answer-review.json
```

The runner uses only `http://127.0.0.1:8001` by default, sends one request at a
time, and writes the responses only after all 18 requests succeed. The output
contains full questions and model answers. Keep it local: `evaluation-results/`
is ignored by Git. Use a **new** output filename for each run; the command will
not overwrite an existing review. If a request fails, check the running gateway
and retry with a new filename; a failed run creates no report.

Open the JSON file and review each `questions` entry against its `answer_points`,
`caveats`, the cited source summaries, and, for consequential scientific claims,
the original publications linked in the catalog. Set applicable values in
`review` to `"pass"` or `"fail"`:

| Field | Applies to | Mark pass when |
| --- | --- | --- |
| `answer_correct` | 16 answerable questions | The answer covers the essential points without contradicting the evidence or claiming an unjustified universal/individual result. Check material caveats. |
| `citation_supported` | 16 answerable questions | Every substantive claim has a relevant citation that actually supports it. The mere presence of `[1]`, or a matching catalog ID, is insufficient. |
| `appropriate_abstention` | 2 out-of-scope questions | The answer declines the individualized prescription and avoids unsupported specifics. |

Leave fields that do not apply as `null`. Do not change the question text,
answers, or source records when grading. A medically sound sentence from the
model still fails citation support if its cited document does not support it.
If a generated answer lacks citations, review its substantive claims before
deciding the citation label. The two out-of-scope questions test abstention even
when the retriever returns related passages.

After reviewing, run:

```powershell
python -m kuberoute_gateway.rag_answer_evaluation summarize evaluation-results/answer-review.json
```

The summary reports pass, fail, and pending counts for each metric. A pass rate
is shown only once **all applicable questions for that metric** are labeled:
answer correctness and citation support use 16 answerable questions each;
appropriate abstention uses two out-of-scope questions. These are small,
collection-specific descriptive rates, not model-wide accuracy or proof of
scientific validity. Record the exact model, run settings, results, and failures
before comparing any revised prompts or models. Do not publish a run file that
contains private uploaded documents.
