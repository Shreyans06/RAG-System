from src.models.factory import get_chat_model

_HYDE_PROMPT = (
    "Write a short, plausible passage (2-4 sentences) in the style of SEC filing "
    "disclosure language that would answer this question. Do not mention that this is "
    "hypothetical — write it as if it were an excerpt from a real filing.\n\nQuestion: {question}"
)


def generate_hypothetical_answer(question: str) -> str:
    """Generates a hypothetical answer passage (HyDE), embedded instead of the raw query for
    the dense retrieval leg — its embedding tends to sit closer to real filing text than the
    bare question does. Does not fail open itself; the caller (QdrantHybridRetriever) is
    responsible for catching errors and falling back to the raw query.
    """
    model = get_chat_model("query_transform")
    response = model.invoke(_HYDE_PROMPT.format(question=question))
    return response.content
