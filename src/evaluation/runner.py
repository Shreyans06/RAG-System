import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from src.config import get_settings
from src.evaluation.dataset import load_eval_dataset
from src.evaluation.metrics import build_judge_embeddings, build_judge_llm, compute_ragas_metrics
from src.evaluation.models import EvalReport, EvalResult
from src.generation.history import get_session_history
from src.generation.lc_rag_chain import LCRAGChain
from src.retrieval.filters import build_filter
from src.retrieval.vector_store import make_client


def _build_rag_chain(settings) -> LCRAGChain:
    qdrant = make_client(settings.QDRANT_URL)
    count = qdrant.count(settings.QDRANT_COLLECTION_NAME).count
    if count == 0:
        raise ValueError("No chunks found in the Qdrant collection. Please ensure that the collection is populated.")
    return LCRAGChain(qdrant, settings.QDRANT_COLLECTION_NAME)


def _mean(values: list[float | None]) -> float | None:
    """Compute the mean of a list of values, ignoring None values."""
    filtered_values = [v for v in values if v is not None]
    return sum(filtered_values) / len(filtered_values) if filtered_values else None


def _run_turn(
    chain: LCRAGChain,
    example_id: str,
    question: str,
    ground_truth_excerpt: str,
    session_id: str | None,
    filters,
) -> EvalResult:
    start = time.perf_counter()
    try:
        output = chain.query(question, session_id=session_id, filters=filters)
    except Exception as e:
        return EvalResult(example_id=example_id, question=question, answer="", mode="lc_rag_chain", error=str(e))

    latency = (time.perf_counter() - start) * 1000  # Convert to milliseconds
    contexts = output.get("contexts", [])
    excerpt_found = any(ground_truth_excerpt.lower() in ctx.lower() for ctx in contexts)

    return EvalResult(
        example_id=example_id,
        question=question,
        answer=output.get("answer", ""),
        retrieved_chunk_ids=[s["chunk_id"] for s in output["sources"]],
        retrieved_contexts=contexts,
        citation_precision=output.get("citation_precision"),
        citation_coverage=output.get("citation_coverage"),
        excerpt_found_in_retrieval=excerpt_found,
        latency_ms=latency,
        mode="lc_rag_chain",
    )


def run_eval(
    dataset_path: str,
    limit: int | None = None,
    ragas_metrics: list[str] | None = None,
) -> EvalReport:
    settings = get_settings()
    examples = load_eval_dataset(dataset_path)
    if limit:
        examples = examples[:limit]
    chain = _build_rag_chain(settings)

    results: list[EvalResult] = []
    ground_truth_by_id: dict[str, str] = {}
    question_type_by_id: dict[str, str] = {}

    for example in examples:
        filters = build_filter(
            tickers=[example.ticker] if example.ticker else None,
            filing_type=example.filing_type or None,
            fiscal_year=example.fiscal_year,
        )

        # Multi-turn examples get a dedicated conversation session so follow-ups can
        # exercise real condense-question + Redis-backed memory resolution; single-turn
        # examples keep session_id=None, identical to the original stateless behavior.
        session_id = f"eval-{example.id}" if example.follow_ups else None
        if session_id:
            get_session_history(session_id).clear()

        results.append(
            _run_turn(chain, example.id, example.question, example.ground_truth_excerpt, session_id, filters)
        )
        ground_truth_by_id[example.id] = example.ground_truth_answer
        question_type_by_id[example.id] = example.question_type.value

        for i, follow_up in enumerate(example.follow_ups, start=1):
            turn_id = f"{example.id}-f{i}"
            # Deliberately no explicit filters here — the point is testing whether
            # conversation memory + condense-question resolve context on their own,
            # the same way a real multi-turn chat session would.
            results.append(
                _run_turn(chain, turn_id, follow_up.question, follow_up.ground_truth_excerpt, session_id, None)
            )
            ground_truth_by_id[turn_id] = follow_up.ground_truth_answer
            question_type_by_id[turn_id] = example.question_type.value

    scored = [r for r in results if r.error is None]
    if scored:
        judge_llm = build_judge_llm(
            provider=settings.evaluation.get("judge_provider", "openai"),
            model=settings.evaluation.get("judge_model", "gpt-5.4-mini"),

        )
        judge_embeddings = build_judge_embeddings(
            provider=settings.evaluation.get("judge_provider", "openai"),
            model=settings.evaluation.get("judge_embedding_model", "text-embedding-3-small"),
        )

        per_row = compute_ragas_metrics(
            questions=[r.question for r in scored],
            answers=[r.answer for r in scored],
            contexts=[r.retrieved_contexts for r in scored],
            ground_truths=[ground_truth_by_id[r.example_id] for r in scored],
            judge_llm=judge_llm,
            judge_embeddings=judge_embeddings,
            metric_names=ragas_metrics,
        )

        for result, scores in zip(scored, per_row):
            result.faithfulness = scores.get("faithfulness")
            result.answer_relevancy = scores.get("answer_relevancy")
            result.context_precision = scores.get("context_precision")
            result.context_recall = scores.get("context_recall")

    aggregates = {
        "faithfulness": _mean([r.faithfulness for r in results if r.faithfulness is not None]),
        "answer_relevancy": _mean([r.answer_relevancy for r in results if r.answer_relevancy is not None]),
        "context_precision": _mean([r.context_precision for r in results if r.context_precision is not None]),
        "context_recall": _mean([r.context_recall for r in results if r.context_recall is not None]),
        "citation_precision": _mean([r.citation_precision for r in results if r.citation_precision is not None]),
        "citation_coverage": _mean([r.citation_coverage for r in results if r.citation_coverage is not None]),
        "retrieval_recall_at_k": _mean(
            [1.0 if r.excerpt_found_in_retrieval else 0.0 for r in results if r.error is None]
        ),
        "error_rate": (sum(1 for r in results if r.error is not None) / len(results)) if results else 0.0,
    }

    report = EvalReport(
        results=results,
        aggregates=aggregates,
        run_metadata={
            "dataset_path": str(dataset_path),
            "mode": "lc_rag_chain",
            "num_examples": len(examples),
            "num_turns": len(results),
            "timestamp": datetime.now(UTC).isoformat(),
        },
    )
    _write_report(report, question_type_by_id)
    return report


def _write_report(report: EvalReport, question_type_by_id: dict[str, str]) -> None:
    reports_dir = Path("eval/reports")
    reports_dir.mkdir(parents=True , exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    (reports_dir / f"{stamp}.json").write_text(json.dumps(asdict(report), indent=2, default=str))

    lines = [f"# Eval Report — {stamp}", "", "## Aggregates", "", "| Metric | Value |", "|---|---|"]
    for name, value in report.aggregates.items():
        lines.append(f"| {name} | {value:.3f} |" if value is not None else f"| {name} | N/A |")

    lines += [
        "",
        "## Per-Example Results",
        "",
        "| ID | Type | Faithfulness | Citation Precision | Excerpt Found | Error |",
        "|---|---|---|---|---|---|",
    ]
    for r in report.results:
        qtype = question_type_by_id.get(r.example_id, "")
        lines.append(
            f"| {r.example_id} | {qtype} "
            f"| {r.faithfulness if r.faithfulness is not None else 'N/A'} "
            f"| {r.citation_precision if r.citation_precision is not None else 'N/A'} "
            f"| {r.excerpt_found_in_retrieval} | {r.error or ''} |"
        )
    (reports_dir / f"{stamp}.md").write_text("\n".join(lines))

