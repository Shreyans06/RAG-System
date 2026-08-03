from src.generation.lc_rag_chain import OUT_OF_SCOPE_MESSAGE

# The system is instructed to emit one of two near-exact phrases when it correctly
# declines to answer (see OUT_OF_SCOPE_MESSAGE and SYSTEM_PROMPT rule 4 in
# lc_rag_chain.py) rather than an arbitrary hedge. Checking for those directly is more
# precise than pattern-matching generic uncertainty language, and avoids false-positiving
# on a real answer that merely mentions a limitation in passing.
_REFUSAL_PHRASES = [
    OUT_OF_SCOPE_MESSAGE.lower(),
    "i don't have enough information to answer this",
    "i do not have enough information to answer this",
]


def is_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(phrase in lowered for phrase in _REFUSAL_PHRASES)


def refusal_correctness(answer: str) -> float:
    """Score for the `unanswerable` question category: 1.0 if the system correctly
    declined to answer, 0.0 if it fabricated an answer instead. Kept separate from the
    RAGAS claim-verification metrics (faithfulness, context_precision, etc.), which
    assume a real answer exists to check claims/rankings against — a correct refusal
    makes no claims at all, so those metrics structurally score it 0 regardless of
    whether declining was the right call (see D-33)."""
    return 1.0 if is_refusal(answer) else 0.0
