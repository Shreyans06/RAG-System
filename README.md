# SEC Filings RAG System

A production-minded RAG system for question-answering over SEC filings (10-K/10-Q), built to demonstrate real GenAI engineering practice — not just "call an LLM with retrieved context," but the surrounding work that makes a RAG system trustworthy: hybrid retrieval, an independent evaluation harness with real metrics, citation grounding, multi-turn memory, tracing, and guardrails.

**[Live demo →](https://claude.ai/code/artifact/fc65862c-3ea3-4db9-8816-bd4645e27da2)** — real cached model output, citations included, zero live API calls per visitor.

This repo's `docs/` folder (kept out of version control) holds the full system architecture diagram and a running decision log — every notable design decision, including the real bugs found and fixed along the way, not just the final state.

## What it does

Ingests SEC filings (fetched via the EDGAR API, with authoritative ticker/CIK/filing-date/fiscal-period metadata attached to every chunk) and answers natural-language questions about them, with inline citations back to the specific filing chunk that supports each claim. The ingested corpus currently spans **10 NASDAQ companies** (AAPL, MSFT, NVDA, AMZN, GOOGL, META, AVGO, TSLA, COST, NFLX) across **fiscal years 2020–2026** — roughly **70,000 chunks**, not a handful of sample PDFs.

**Example, real (not illustrative) — from this system, against the live ingested corpus:**

> **Q:** How did Apple describe its iPad net sales performance in fiscal Q1 2025?
> **A:** Apple said iPad net sales increased during the first quarter of 2025 compared to the same quarter in 2024 due primarily to higher net sales of iPad Air `[99c2c6c0]`.

> **Q:** What is the capital of France and how many moons does Jupiter have?
> **A:** I don't have information about that in the ingested filings.
>
> *(This is a deterministic refusal, not the model's judgment call — retrieval's reranked relevance scores fall below a calibrated threshold, so the system never even calls the generation model. See D-7/guardrails in the decision log.)*

> **Q (typed straight into the chat, not a CLI command):** "Add Tesla's most recent 10-K"
> **A:** Kicks off ingestion as a background job and returns immediately with a job ID; a follow-up like "is Tesla ready yet?" checks status without needing to remember it. No separate ingestion UI required — the same chat model that answers questions also recognizes and dispatches ingestion requests via tool-calling. See D-18.

> **Q:** Compare Apple's net sales in 2024 vs Microsoft's net revenue in 2024.
> **A:** Apple's total net sales reached $391,035 million in 2024 `[815fe6bf]` ... Microsoft reported revenue of $245,122 million in 2024 `[3fb3f3ac]` ... Apple's figure was significantly higher, though Microsoft grew faster year-over-year (16% vs. 2%).
>
> *(Cross-company questions run one full retrieval pass per company and merge the results — a single shared search let the stronger-scoring company crowd the other out of the answer entirely; found live, fixed same-day. See D-23.)*

## Architecture, in brief

- **Ingestion:** EDGAR API client → HTML/table parsing (iXBRL-aware) → SEC section-boundary-aware chunking (Item 1, 1A, 7, 7A...) → hierarchical chunking, with content-hash IDs so re-ingestion is idempotent. Tables get an LLM-generated natural-language description prepended at ingestion time (D-16) — the reranker otherwise systematically scores raw markdown-table syntax below prose, regardless of relevance. Table descriptions and contextual summaries run through a provider-agnostic model factory (D-25) and are generated concurrently rather than one at a time (D-26).
- **Cost-optimized bulk ingestion:** dense embeddings can be submitted through OpenAI's Batch API (~50% cheaper, D-27) instead of synchronous calls — built after hitting three distinct real failures scaling to 10 companies: duplicate chunk IDs colliding across different filings' boilerplate cover pages, an org-wide enqueued-token ceiling shared across all in-flight batches, and a single company's filings alone exceeding that ceiling. All three diagnosed and fixed live; the pipeline now stages, token-budgets, and submits batches one at a time automatically.
- **Retrieval:** Qdrant native hybrid search (dense + sparse vectors, server-side RRF fusion) → HyDE → multi-query expansion → Cohere reranking, with auto-extracted metadata filters (ticker/filing-type/fiscal-year/quarter) inferred from the question itself. Comparative questions spanning multiple periods, years, *or companies* run a separate retrieval pass per item and combine the results, so one period or company can't crowd the others out of the context window (D-19, D-23).
- **Generation:** LangChain/LCEL chain over a provider-agnostic model factory (swap Anthropic ⇄ OpenAI ⇄ Ollama via config, not code), with mandatory inline citations and a deterministic out-of-scope refusal when retrieval confidence is too low. Currently configured on `claude-sonnet-5`, chosen after head-to-head eval testing showed it handling large, redundant multi-chunk contexts more reliably than GPT-family models (D-19).
- **Chat-driven ingestion:** the chat model itself recognizes ingestion requests (via LangChain tool-calling) and dispatches them as background jobs, distinct from normal question-answering (D-18).
- **Memory:** Redis-backed, session-scoped conversation history with a condense-question step, so follow-ups like "what about Q3?" resolve correctly without restating the company/period.
- **Evaluation:** A hand-written, hand-verified golden set — 63 examples, 68 graded turns, spanning factual/numerical/comparative/multi-hop/unanswerable/adversarial question types, multi-turn conversations, and deliberate known-limitation regression probes — scored with RAGAS (faithfulness, answer relevancy, context precision/recall) using an independent OpenAI judge, plus rule-based citation-precision/coverage metrics — `make eval`.
- **Observability:** LangSmith tracing (retrieval → rerank → generation, full pipeline visibility) plus a local structured JSON query log as a fallback/complement.

Full detail, including the specific bugs this surfaced (fiscal-quarter boundary edge cases, a reranker that systematically demotes tables below prose, a citation-ID collision bug caught by the observability work itself, a GPT-vs-Claude generation-quality gap on noisy multi-chunk context, and a whitespace-normalization bug found *inside the eval harness itself*) is in the project's decision log.

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

The golden set (`eval/datasets/sec_qa_v1.jsonl`, 63 examples / 68 graded turns) is hand-written and hand-verified against the actual ingested filings — every ground-truth excerpt is checked to be a real, present substring of the actual retrieved-scope content, not just plausible-sounding. It deliberately spans factual, numerical, comparative (within-year, cross-year, and cross-company), multi-hop, unanswerable, and adversarial question types, plus multi-turn conversations that exercise condense-question + memory resolution, and two deliberate known-limitation regression probes (expected to be flaky — they track a documented, still-open gap rather than a surprise). The unanswerable category alone covers six distinct kinds of gap: a company never ingested, a fiscal year that hasn't happened yet, a fiscal year before the ingestion window starts, a topic never discussed in any filing, and more. Scores come from an independent OpenAI judge (not the same model family as the generator, to avoid self-grading bias), combined with rule-based citation metrics (`citation_precision`/`citation_coverage`) that check whether cited chunk IDs were actually among those supplied to the model.

Full rationale for every methodology choice — why hand-written over LLM-generated, why an independent judge, why RAGAS, why an earlier `unsupported_numeric_claims` metric was removed after it turned out to mostly flag false positives rather than real unsupported claims, and a whitespace-normalization bug found *in the eval harness itself* that was silently producing false negatives on any excerpt spanning a line-wrap in the raw filing text — is in the project's decision log (D-1, D-3, D-20, D-22, D-28).

## Observability

Every query is traced end-to-end in LangSmith (retrieval → HyDE/multi-query → reranking → generation), and locally logged to `logs/queries.jsonl` as a structured, LangSmith-independent fallback. Tracing is fail-open: with no `LANGSMITH_API_KEY` configured, the system runs identically, just untraced.

## Future work

Scoped out deliberately, not silently omitted:

- **XBRL structured-numeric lookup** — bypassing chunk retrieval entirely for exact numeric facts would strengthen the "correct every time" story specifically for numbers, but is a meaningful chunk of additional work beyond this pass's scope.
- **Multi-table-per-filing disambiguation (D-19, open)** — when a single filing contains several tables (e.g. an income statement and a separate segment breakdown), the reranker doesn't always surface the *specific* table a question needs, for both single-period and cross-period/cross-year comparative questions. Narrower and longer-tail than the table-vs-prose bias fixed in D-16; documented rather than silently left as a known-bad behavior. Two eval questions (`kl1`/`kl2`) deliberately track this — expect them to be flaky, not broken.
- **Multimodal image pipeline (discovered, not wired in)** — `image_extractor.py`/`vision_processor.py` exist and function standalone but were never actually wired into the main ingestion pipeline (`pipeline.py`); PDF image extraction/description is not currently part of the live ingestion flow.
- **Automated test suite** — this project leaned on extensive live/real-API verification throughout development (every feature was checked against a real Qdrant instance, real LLM calls, real HTTP requests) rather than a parallel automated test suite; that tradeoff was made deliberately given the project's scope, not by default.
