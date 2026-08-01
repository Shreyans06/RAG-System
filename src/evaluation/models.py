from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class QuestionType(StrEnum):
    FACTUAL = "factual"
    NUMERICAL = "numerical"
    COMPARATIVE = "comparative"
    MULTI_HOP = "multi-hop"
    UNANSWERABLE = "unanswerable"

@dataclass
class EvalExample:
    id : str
    question : str
    question_type : QuestionType
    ticker : str
    filing_type : str
    fiscal_year : int
    ground_truth_answer: str
    ground_truth_excerpt: str
    difficulty: str = "medium" # "easy"

@dataclass
class EvalResult:
    example_id: str
    question: str
    answer: str
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    retrieved_contexts: list[str] = field(default_factory=list)

    # RAGAS metrics (None of the metric call failed or returned None)
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None

    # Rule-based citation metrics
    citation_precision: float | None = None
    citation_coverage: float | None = None

    # Retrieval-only signal: did any retrieved context contain the ground truth answer?
    excerpt_found_in_retrieval: bool = False

    latency_ms: float = 0.0
    mode: str = "hybrid_rerank"
    error: str | None = None

@dataclass
class EvalReport:
    results: list[EvalResult] = field(default_factory=list)
    aggregates: dict[str , float] = field(default_factory=dict)
    run_metadata: dict[str , Any] = field(default_factory=dict)