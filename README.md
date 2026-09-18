# VoIP RAG

Standalone Retrieval-Augmented Generation service over GetVoIP WordPress content: articles, user reviews, and provider pages. It answers questions with citations from Qdrant.

**Stack:** Python 3.12+, FastAPI, Pydantic, OpenAI (`text-embedding-3-small` for embeddings, GPT for extract / rewrite / rerank / generation), Qdrant 1.13, LangGraph for orchestration only, Langfuse for traces.

**Plan:** `documents/implementation-plan.md` and `documents/implementation-checklist.md`. Retrieval logic stays in `app/retrieval/` and `app/query/`. LangGraph only calls those modules.

---

## What is implemented

### Ingestion

WordPress HTML is cleaned and turned into `NormalizedDocument` JSON (title, content type, provider, URL, dates, heading-aware sections). Three chunkers write retrieval units with heading path and metadata. OpenAI embeds each chunk (`title + provider + section + body`) and upserts vectors plus payload into one Qdrant collection per chunker.

A fixture exporter pulls published posts from local MySQL (`blog_post`, `news_post`, `library_post`, reviews, providers) into JSON. `rag-ingest` normalizes, chunks, embeds, and indexes a fixture file. `rag-push` copies already-embedded collections to another Qdrant (for example local → Cloud) without re-embedding.

The live MySQL reader `wordpress.py` is still a stub. Ingest is fixture-based.

### Retrieval

Four `POST /query` strategies:

| Strategy | Behavior | Index |
| --- | --- | --- |
| `dense` | Cosine search, top 5 | Active `CHUNKER` collection |
| `sparse` | In-process BM25 over chunked JSON | No Qdrant |
| `hybrid` | Dense + sparse, Reciprocal Rank Fusion (`k=60`) | Active collection + local BM25 |
| `rerank` | Hybrid pool of 20, then OpenAI listwise rerank to top 5 | Same as hybrid |

Optional metadata filters: `provider`, `content_type`, `document_id`, plus a section heading hint. Filters can be sent in the request or inferred from the question. A single catalog provider named in the question sets `content_type=provider`. Review kind sets `content_type=review`. Unknown vendors (for example NICE CXone) leave `content_type` unset so articles are searched. There are no hardcoded synonym or cue lists.

When `CHUNKER=parent_child`, child hits are expanded to parent context at query time.

### Query intelligence

Before retrieval, the question is classified and extracted (kind, catalog providers, topics), rewritten into one tighter retrieval query, and optionally decomposed into sub-queries for multi-provider / multi-topic questions.

Kinds: `factual`, `comparison`, `recommendation`, `pricing`, `review`, `multi-hop`.

Extract and rewrite use `gpt-4.1-nano` with a heuristic fallback. Providers must appear both in the live Qdrant catalog and in the question. The model must not invent or substitute vendors. Decomposition stays the multi-hop path; rewrite is a single query, not HyDE and not a synonym expander.

The provider catalog is loaded from Qdrant payload (cached in process).

### Evidence and generation

After retrieve (and optional corrective retry), a heuristic evidence check measures token overlap with the question. If evidence is thin, the pipeline rewrites, drops the section filter, and retrieves again. If still insufficient, the API abstains instead of guessing.

Generation is grounded in retrieved passages only. The response includes citations (`title`, `section`, `url`) and a `retrieval` block (strategy, filters, preprocess, evidence).

### Evaluation and tracing

`evals/golden.json` holds 50 manually written cases (`eval_001`–`eval_050`) for the current corpus, including 25 added comparison questions. `rag-eval --compare` scores dense / sparse / hybrid / rerank (Recall@K, Precision@K, MRR, nDCG, generation metrics). `--compare-chunkers` compares dense retrieval across chunker collections.

Langfuse records `rag.query`, `query.classify`, `query.rewrite`, `retrieval`, `evidence`, and `generation` / abstain. Rerank is part of the retrieval span, not a separate span.

### HTTP API

| Method | Path | Auth |
| --- | --- | --- |
| `GET` | `/health` | Public |
| `GET` | `/docs`, `/redoc`, `/openapi.json` | Public |
| `POST` | `/query` | `X-API-Key` or `Authorization: Bearer` |

Swagger Authorize uses `X-API-Key`. Example body: `{ "query": "who is better for startups Nextiva or Dialpad?", "strategy": "rerank" }`.

### Docker

`Dockerfile` builds `python:3.12-slim` and runs uvicorn on port 8000. `docker-compose.yml` has `api` (reads `.env`) and optional local `qdrant` (`:6333`).

---

## Key implementations

### App entry and contracts

| File | What it owns |
| --- | --- |
| `app/main.py` | FastAPI app, Swagger tags, API-key middleware |
| `app/config.py` | Paths, collection names, all env-backed settings |
| `app/models/schemas.py` | Pydantic models: documents, chunks, query I/O, eval cases, filters, strategies |
| `app/api/query.py` | `POST /query` → `run_rag_graph` |
| `app/api/health.py` | Public `GET /health` |
| `app/api/auth.py` | `X-API-Key` / Bearer check; `/docs` and `/health` stay open |

### Ingestion

| File | What it owns |
| --- | --- |
| `app/ingestion/export_fixtures.py` | MySQL → `articles.json`, `reviews.json`, `providers.json` |
| `app/ingestion/loader.py` | Load fixture JSON into `RawPost` |
| `app/ingestion/cleaner.py` | Strip WordPress nav, ads, scripts, shortcodes, CTAs |
| `app/ingestion/parser.py` | HTML → headings, paragraphs, lists, tables |
| `app/ingestion/normalizer.py` | `RawPost` → `NormalizedDocument` |
| `app/ingestion/normalize.py` | CLI `rag-normalize` |
| `app/ingestion/chunker.py` | `Chunker` ABC; `StructureAwareChunker`, `FixedSizeChunker`, `ParentChildChunker` |
| `app/ingestion/chunk.py` | CLI `rag-chunk` (`--chunker`) |
| `app/ingestion/embed_text.py` | Text sent to the embedder |
| `app/ingestion/embedder.py` | `Embedder` ABC and `OpenAIEmbedder` |
| `app/ingestion/indexer.py` | Qdrant client, collection create, batched upsert with retries |
| `app/ingestion/index.py` | CLI `rag-index` (chunk JSON → embed → Qdrant) |
| `app/ingestion/ingest.py` | CLI `rag-ingest` (normalize + chunk + embed + upsert) |
| `app/ingestion/push.py` | CLI `rag-push` (copy collections, no re-embed) |
| `app/ingestion/wordpress.py` | Stub for a later live MySQL reader |

### Retrieval

| File | What it owns |
| --- | --- |
| `app/retrieval/__init__.py` | `retriever_for(strategy)` factory |
| `app/retrieval/dense.py` | Qdrant dense search |
| `app/retrieval/sparse.py` | BM25 over on-disk chunks |
| `app/retrieval/fusion.py` | Reciprocal Rank Fusion |
| `app/retrieval/hybrid.py` | Dense + sparse + RRF |
| `app/retrieval/reranker.py` | `OpenAIReranker` (listwise JSON), `RerankRetriever` (hybrid 20 → top 5) |
| `app/retrieval/filters.py` | Infer / merge filters; Qdrant filter builder |
| `app/retrieval/parent_child.py` | Expand child hits to parent passages |
| `app/retrieval/chunker_compare.py` | Dense retrievers per chunker collection |

### Query intelligence

| File | What it owns |
| --- | --- |
| `app/query/catalog.py` | Live provider/heading catalog from Qdrant (`get_catalog`) |
| `app/query/extractor.py` | `OpenAIExtractor` + `HeuristicExtractor`; constrain providers to names in the question |
| `app/query/classifier.py` | Kind via the extractor |
| `app/query/rewriter.py` | `OpenAIRewriter` + `HeuristicRewriter`; one retrieval query; no invented vendors |
| `app/query/decomposer.py` | Multi-hop split; `expand_queries` |
| `app/query/evidence.py` | Sufficient / insufficient check; abstain message |
| `app/query/corrective.py` | Rewrite + second retrieve when evidence is thin |
| `app/query/inspect.py` | Offline preprocess / evidence inspect JSON |

### Generation and orchestration

| File | What it owns |
| --- | --- |
| `app/generation/context.py` | Pack retrieved chunks into the LLM context window |
| `app/generation/prompts.py` | Grounded-answer system and user prompts |
| `app/generation/citations.py` | `Source` objects from chunks |
| `app/generation/generator.py` | `OpenAIGenerator` |
| `app/workflows/rag_graph.py` | LangGraph: preprocess → retrieve → generate \| abstain |
| `app/observability/tracing.py` | Langfuse tracer and `span()` helper |

### Evaluation and tests

| File | What it owns |
| --- | --- |
| `evals/golden.json` | 50 golden cases |
| `app/evaluation/dataset.py` | Load eval cases |
| `app/evaluation/retrieval.py` | Recall@K, Precision@K, MRR, nDCG |
| `app/evaluation/generation.py` | Generation / engineering metrics |
| `app/evaluation/experiments.py` | Single-strategy and compare runners |
| `app/evaluation/run.py` | CLI `rag-eval` / `python -m app.evaluation.run` |
| `tests/` | Unit and API tests mirroring `app/` |

### Data and generated artifacts

| Path | Role |
| --- | --- |
| `data/articles.json`, `data/reviews.json`, `data/providers.json` | Current fixture dumps used for ingest |
| `articles.json`, `reviews.json`, `providers.json` (project root) | Default paths in `app/config.py` for export and `load_raw_posts` |
| `documents/normalized/` | Normalized documents |
| `documents/chunked/`, `chunked_fixed/`, `chunked_parent_child/` | Chunk JSON per strategy |
| `documents/indexed/` | Index and eval reports |
| `documents/query/` | Preprocess / evidence inspect dumps |
| `qdrant_storage/` | Local Qdrant volume |

---

## How to run

Python 3.12+ recommended (local venv may be newer). Install and copy env:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Start local Qdrant if you are not using Qdrant Cloud:

```bash
docker compose up -d qdrant
```

Index fixtures (paths can be `data/*.json` or root `*.json`):

```bash
rag-ingest data/articles.json article --chunker all
rag-ingest data/reviews.json review --chunker all
rag-ingest data/providers.json provider --chunker all
```

API:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

```bash
curl -s http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"query":"who is better for startups Nextiva or Dialpad?","strategy":"rerank"}'
```

Docker API (uses `.env`):

```bash
docker compose up --build api
```

Eval and tests:

```bash
python -m app.evaluation.run --compare
pytest
```

Copy a local index to Cloud (destination is `QDRANT_URL`):

```bash
rag-push
```

Re-export fixtures from local MySQL:

```bash
python -m app.ingestion.export_fixtures
```

---

## Environment variables

Copy `.env.example` to `.env`. `app/config.py` loads that file.

### Required to run the API and answer questions

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Embeddings, extract, rewrite, rerank, and generation |
| `API_KEY` | Protects `POST /query`. Send it as `X-API-Key` or `Authorization: Bearer` |

Without `API_KEY`, query returns 503. `/health` and `/docs` stay public.

### Required for Qdrant

| Variable | Purpose | Default |
| --- | --- | --- |
| `QDRANT_URL` | Qdrant HTTP endpoint (local or Cloud) | `http://127.0.0.1:6333` |
| `QDRANT_API_KEY` | Cloud JWT / API key. Empty for local Docker | empty |

Local Docker Qdrant does not need `QDRANT_API_KEY`. Qdrant Cloud does.

### Required only to export fixtures from MySQL

Used by `python -m app.ingestion.export_fixtures` (not by the API):

| Variable | Purpose | Default |
| --- | --- | --- |
| `MYSQL_HOST` | WordPress MySQL host | `127.0.0.1` |
| `MYSQL_PORT` | MySQL port | `3306` |
| `MYSQL_USER` | MySQL user | `root` |
| `MYSQL_PASSWORD` | MySQL password | empty |
| `MYSQL_DATABASE` | Database name | `getvoip` |

### Optional — collections and chunking

| Variable | Purpose | Default |
| --- | --- | --- |
| `CHUNKER` | Active chunker / default collection: `structure_aware`, `fixed_size`, `parent_child` | `structure_aware` |
| `QDRANT_COLLECTION` | Structure-aware collection | `getvoip_chunks_structure_aware` |
| `QDRANT_COLLECTION_FIXED` | Fixed-size collection | `getvoip_chunks_fixed` |
| `QDRANT_COLLECTION_PARENT_CHILD` | Parent-child collection | `getvoip_chunks_parent_child` |
| `QDRANT_SOURCE_URL` | Source URL for `rag-push` | `http://127.0.0.1:6333` |
| `QDRANT_TIMEOUT` | Upsert timeout (seconds) | `60` |
| `QDRANT_UPSERT_BATCH` | Points per upsert batch | `32` |

### Optional — models and retrieval

| Variable | Purpose | Default |
| --- | --- | --- |
| `OPENAI_EMBED_MODEL` | Embedding model | `text-embedding-3-small` |
| `OPENAI_CHAT_MODEL` | Answer generation model | `gpt-4.1-mini` |
| `QUERY_EXTRACTOR` | `openai` or `heuristic` | `openai` |
| `QUERY_EXTRACTOR_MODEL` | Extract / classify model | `gpt-4.1-nano` |
| `QUERY_REWRITER` | `openai` or `heuristic` | `openai` |
| `QUERY_REWRITER_MODEL` | Rewrite model | same as `QUERY_EXTRACTOR_MODEL` |
| `RERANKER` | Reranker backend | `openai` |
| `RERANKER_MODEL` | Rerank model | same as `QUERY_EXTRACTOR_MODEL` |
| `DENSE_TOP_K` | Dense / hybrid cutoff | `5` |
| `RRF_K` | RRF constant | `60` |
| `RERANK_CANDIDATES` | Hybrid pool before rerank | `20` |
| `RERANK_TOP_K` | Results after rerank | `5` |
| `EVIDENCE_CHECKER` | `heuristic` or `openai` | `heuristic` |
| `EVIDENCE_MIN_OVERLAP` | Minimum overlap to accept evidence | `0.3` |
| `EMBEDDER` / `GENERATOR` | Embedding and generation provider | `openai` |

### Optional — Langfuse

If both keys are set, traces are sent. Omit them to run without tracing.

| Variable | Purpose | Default |
| --- | --- | --- |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key | empty |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key | empty |
| `LANGFUSE_BASE_URL` | Langfuse host (`LANGFUSE_HOST` also accepted) | `https://cloud.langfuse.com` |
| `LANGFUSE_ENABLED` | Force on/off (`1`/`0`). Empty = on when keys exist | empty |

### Optional — tests and cost bookkeeping

| Variable | Purpose |
| --- | --- |
| `RUN_LIVE_RAG=1` | Enable live Qdrant + OpenAI tests in `tests/api/test_live_query.py` |
| `OPENAI_EMBED_USD_PER_1M`, `OPENAI_INPUT_USD_PER_1M`, `OPENAI_OUTPUT_USD_PER_1M` | Rough eval cost estimates, not invoices |
