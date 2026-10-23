# Exercise science knowledge base v1

This is an educational, local-first collection of **12 original summaries** of research abstracts and NASM framework pages, verified on 2026-10-23. It is not medical advice, a rehabilitation protocol, or proof that one program is best for everyone. The summaries distinguish findings reported by sources from practical interpretation and uncertainty. NASM's OPT and corrective-exercise systems are labeled professional frameworks, not independent demonstrations of superiority.

`manifest.json` is the curated source catalog. Each entry has a stable **catalog ID**, relative document path, bibliographic metadata, topic labels, and verification date. RAG ingestion hashes each canonical local path to create a **runtime source ID**. With the optional manifest setting below, a matching cited runtime source is accompanied by verified bibliographic metadata in the API and dashboard. Without that setting, citations retain their hashed display. A marker still identifies a retrieved passage, not proof that every claim is supported. Follow the original URLs in the manifest to inspect the underlying publications.

## Run the collection locally on Windows PowerShell

Start with an installed Python environment containing KubeRoute. Validate the catalog and derive the explicit file list without scanning directories:

```powershell
$manifest = 'knowledge/exercise_science/manifest.json'
python -m kuberoute_gateway.knowledge_catalog validate $manifest
$env:KUBEROUTE_RAG_DOCUMENT_PATHS_JSON = python -m kuberoute_gateway.knowledge_catalog paths $manifest
$env:KUBEROUTE_RAG_SOURCE_MANIFEST = (Resolve-Path $manifest).Path
```

The `paths` command emits JSON for environment-variable assignment; it does not change files or start a server. Keep its output and the RAG configuration out of logs or committed configuration. A local Ollama server and the existing `nomic-embed-text` and `qwen3:0.6b` models are required for this example. KubeRoute does not install or download them.

```powershell
$env:KUBEROUTE_RAG_ENABLED = 'true'
$env:KUBEROUTE_RAG_EMBEDDING_MODEL = 'nomic-embed-text'
$env:KUBEROUTE_RAG_EMBEDDING_BASE_URL = 'http://127.0.0.1:11434'
$env:KUBEROUTE_REMOTE_BASE_URL = 'http://127.0.0.1:11434'
$env:KUBEROUTE_REMOTE_MODEL_ID = 'qwen3:0.6b'
python -m uvicorn kuberoute_gateway.main:app --host 127.0.0.1 --port 8001
```

Do not set conflicting `KUBEROUTE_REMOTE_MODELS_JSON` alongside the legacy remote-model variables. Port 8001 leaves the existing Docker/Kubernetes services on port 8000 untouched. Open `http://127.0.0.1:8001/` with a trusted compiled dashboard artifact, select **RAG**, and ask a question from `evaluation/questions.json`. You can also test the API without the dashboard:

```powershell
$body = @{ model='auto'; question='How does weekly set volume relate to hypertrophy?'; retrieval_mode='hybrid'; top_k=5; stream=$false } | ConvertTo-Json
Invoke-RestMethod -Uri 'http://127.0.0.1:8001/v1/rag/chat/completions' -Method Post -ContentType 'application/json' -Body $body
```

The index is built at startup and held in memory; adding, removing, or editing a document requires a FastAPI restart. RAG requests may include retrieved text in the prompt sent to the configured inference backend. Use only trusted local backends for private material. No public network binding, uploads, or document browsing are part of this workflow.

## Add another study

1. Read the original publication or authoritative abstract. Verify title, authors, year, identifier, population, findings, and limits. Omit claims you cannot verify.
2. Write a short, original `.md` file under `sources/` with the same metadata and separate **Findings reported**, **Practical interpretation**, and **Limitations** sections. Do not copy a full paper, paywalled passage, or private material.
3. Add a unique stable ID and a relative `sources/...` path to `manifest.json`. Use an explicit `null` year plus an explanatory note if an undated framework page has no verifiable publication year.
4. Add focused evaluation questions where useful. Run `python -m kuberoute_gateway.knowledge_catalog validate knowledge/exercise_science/manifest.json`, Ruff, and pytest. Restart the local gateway to rebuild its index.

The catalog validator checks paths, duplicate IDs and documents, metadata, UTF-8 text, file limits, and compatibility with existing ingestion. It emits generic errors without document contents or private paths. It does **not** fact-check summaries or fetch publications.

## Evaluation boundaries

`evaluation/questions.json` contains 18 synthetic questions, including two that the collection cannot answer individually. `expected_source_ids` are catalog IDs for offline evaluation, not runtime RAG hashes. Retrieval recall asks whether the expected documents appear in top-K; generation quality asks whether answer points and caveats are supported. These are different tests. A valid `[1]` marker only shows a retrieved passage was cited, not that the claim is true. Unanswerable questions test whether the system admits insufficient evidence; retrieval of a related paper is not itself a failure or proof of sufficient evidence. Small, domain-specific questions cannot establish broad factual reliability.
