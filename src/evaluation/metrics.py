from __future__ import annotations

from datasets import Dataset
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from src.config import get_settings

_METRICS = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "context_precision": context_precision,
    "context_recall": context_recall,
}



def build_judge_llm(provider: str = "openai", model: str = "gpt-5.4-mini") -> BaseChatModel:
    """Build a judge LLM for evaluation."""
    settings = get_settings()
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, api_key=settings.OPENAI_API_KEY, temperature=0.0)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, api_key=settings.ANTHROPIC_API_KEY)
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, temperature=0.0)
    raise ValueError(f"Unsupported judge LLM provider: {provider}")

def build_judge_embeddings(provider: str = "openai", model: str = "text-embedding-3-small") -> Embeddings:
    """Build a judge embeddings model for evaluation."""
    settings = get_settings()
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model, api_key=settings.OPENAI_API_KEY)
    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=model)
    raise ValueError(f"Unsupported judge embeddings provider: {provider}")


def compute_ragas_metrics(
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: list[str],
        judge_llm: BaseChatModel,
        judge_embeddings: Embeddings,
        metric_names: list[str] | None = None
) -> list[dict[str , float | None]]:
    """Compute RAGAS metrics for questions, answers, contexts, and ground truths
    using the provided judge LLM and embeddings.

    Args:
        questions: List of questions.
        answers: List of answers corresponding to the questions.
        contexts: List of lists of contexts corresponding to the questions.
        ground_truths: List of ground truth answers corresponding to the questions.
        judge_llm: A language model used for evaluation.
        judge_embeddings: An embeddings model used for evaluation."""

    metrics_to_run = {name : _METRICS[name] for name in metric_names} if metric_names else _METRICS

    dataset = Dataset.from_dict(
        {
            "question" : questions,
            "answer" : answers,
            "contexts" : contexts,
            "ground_truth" : ground_truths,
        }
    )

    wrapped_llm = LangchainLLMWrapper(judge_llm)
    wrapped_embeddings = LangchainEmbeddingsWrapper(judge_embeddings)

    n = len(questions)
    per_row: list[dict[str , float | None]] = [dict.fromkeys(_METRICS.keys(), None) for _ in range(n)]

    for name , metric in metrics_to_run.items():
        try:
            result = evaluate(dataset , metrics = [metric] , llm = wrapped_llm , embeddings = wrapped_embeddings)
            scores = result.to_pandas()[name].tolist()
            for i , score in enumerate(scores):
                per_row[i][name] = float(score)
        except Exception as e:
            print(f"[metrics] RAGAS metric '{name}' evaluation failed: {e}")
    
    return per_row