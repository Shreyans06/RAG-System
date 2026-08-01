# SEC Filings RAG System

A production-minded RAG system for question-answering over SEC filings (10-K/10-Q), built to demonstrate real GenAI engineering practice — not just "call an LLM with retrieved context," but the surrounding work that makes a RAG system trustworthy: hybrid retrieval, an independent evaluation harness with real metrics, citation grounding, multi-turn memory, tracing, and guardrails.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full system diagram and [docs/DECISIONS.md](docs/DECISIONS.md) for a running log of every notable design decision — including the real bugs found and fixed along the way, not just the final state.

## What it does

Ingests SEC filings (fetched via the EDGAR API, with authoritative ticker/CIK/filing-date/fiscal-period metadata attached to every chunk) and answers natural-language questions about them, with inline citations back to the specific filing chunk that supports each claim.

**Example, real (not illustrative) — from this system, against the ingested AAPL corpus:**

> **Q:** How did Apple describe its iPad net sales performance in fiscal Q1 2025?
> **A:** Apple said iPad net sales increased during the first quarter of 2025 compared to the same quarter in 2024 due primarily to higher net sales of iPad Air `[99c2c6c0]`.

> **Q:** What is the capital of France and how many moons does Jupiter have?
> **A:** I don't have information about that in the ingested filings.
>
> *(This is a deterministic refusal, not the model's judgment call — retrieval's reranked relevance scores fall below a calibrated threshold, so the system never even calls the generation model. See D-16/guardrails in the decision log.)*

## Architecture, in brief

- **Ingestion:** EDGAR API client → HTML/table parsing (iXBRL-aware) → SEC section-boundary-aware chunking (Item 1, 1A, 7, 7A...) → hierarchical chunking, with content-hash IDs so re-ingestion is idempotent.
- **Retrieval:** Qdrant native hybrid search (dense + BM25-as-sparse-vector, server-side RRF fusion) → HyDE → multi-query expansion → Cohere reranking, with auto-extracted metadata filters (ticker/filing-type/fiscal-year/quarter) inferred from the question itself.
- **Generation:** LangChain/LCEL chain over a provider-agnostic model factory (swap Anthropic ⇄ OpenAI ⇄ Ollama via config, not code), with mandatory inline citations and a deterministic out-of-scope refusal when retrieval confidence is too low.
- **Memory:** Redis-backed, session-scoped conversation history with a condense-question step, so follow-ups like "what about Q3?" resolve correctly without restating the company/period.
- **Evaluation:** A hand-written, hand-verified golden Q&A set scored with RAGAS (faithfulness, answer relevancy, context precision/recall) using an independent OpenAI judge, plus rule-based citation-precision/coverage metrics — `make eval`.
- **Observability:** LangSmith tracing (retrieval → rerank → generation, full pipeline visibility) plus a local structured JSON query log as a fallback/complement.

Full detail, including the specific bugs this surfaced (fiscal-quarter boundary edge cases, a reranker that systematically demotes tables below prose, a citation-ID collision bug caught by the observability work itself) is in [docs/DECISIONS.md](docs/DECISIONS.md).

## Running it

**Docker (recommended):**
```bash
cp .env.example .env   # fill in your API keys
docker-compose up --build
```
This brings up Qdrant, Redis, the FastAPI backend (`:8000`), and the Streamlit UI (`:8501`).

**Local dev** (requires a local Qdrant + Redis, e.g. via `docker run`):
```bash
uv sync
make api   # FastAPI on :8000, --reload
make ui    # Streamlit on :8501
```

**Ingest a company's filings:**
```bash
python scripts/ingest_sec_batch.py --ticker AAPL --years 2023,2024,2025 --filing-types 10-K,10-Q
```

**Run the evaluation harness:**
```bash
make eval
```

**Lint:**
```bash
make lint
```

## Evaluation methodology

The golden set (`eval/datasets/sec_qa_v1.jsonl`) is hand-written and hand-verified against the actual ingested filings, deliberately including factual, numerical, comparative, multi-hop, and unanswerable question types — the last category specifically tests that the system refuses rather than hallucinates. Scores come from an independent OpenAI judge (not the same model family as the generator, to avoid self-grading bias), combined with rule-based citation metrics that check whether cited chunk IDs were actually supplied and whether numeric claims carry a citation at all.

Full rationale for every methodology choice — why hand-written over LLM-generated, why an independent judge, why RAGAS — is in [docs/DECISIONS.md](docs/DECISIONS.md) (D-1, D-3).

## Observability

Every query is traced end-to-end in LangSmith (retrieval → HyDE/multi-query → reranking → generation), and locally logged to `logs/queries.jsonl` as a structured, LangSmith-independent fallback. Tracing is fail-open: with no `LANGSMITH_API_KEY` configured, the system runs identically, just untraced.

## Future work

Scoped out deliberately, not silently omitted:

- **XBRL structured-numeric lookup** — bypassing chunk retrieval entirely for exact numeric facts would strengthen the "correct every time" story specifically for numbers, but is a meaningful chunk of additional work beyond this pass's scope.
- **Cached public demo** — a hosted, pre-computed demo page (serving cached answers from an eval run, zero marginal cost per visitor, no live API exposure) is planned but not yet built; see D-6.
- **Reranker table-demotion (D-16, open)** — the Cohere reranker systematically scores narrative prose above answer-bearing tables for numerical questions, a real, root-caused, currently-unfixed gap. Documented in detail in the decision log rather than silently left as a known-bad behavior.
- **Multimodal image pipeline (discovered, not wired in)** — `image_extractor.py`/`vision_processor.py` exist and function standalone but were never actually wired into the main ingestion pipeline (`pipeline.py`); PDF image extraction/description is not currently part of the live ingestion flow.
- **Automated test suite** — this project leaned on extensive live/real-API verification throughout development (every feature was checked against a real Qdrant instance, real LLM calls, real HTTP requests) rather than a parallel automated test suite; that tradeoff was made deliberately given the project's scope, not by default.
