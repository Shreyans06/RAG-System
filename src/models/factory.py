from typing import Literal

from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from src.config import get_settings

Role = Literal["generation" , "contextual" , "query_transform"]

def get_chat_model(role: Role) -> BaseChatModel:
    settings = get_settings()
    if role == "generation":
        provider = settings.generation.get("provider", "anthropic")
        model = settings.generation.get("model", "claude-sonnet-5")
    elif role == "contextual":
        provider = settings.ingestion.get("contextual_summary_provider", "anthropic")
        model = settings.ingestion.get("contextual_summary_model", "claude-haiku-4-5")
    elif role == "query_transform":
        provider = settings.retrieval.get("query_transform_provider", "anthropic")
        model = settings.retrieval.get("query_transform_model", "claude-haiku-4-5")
    else:
        raise ValueError(f"Unknown chat model role: {role}")
    return _build_chat_model(provider, model, settings)

def _build_chat_model(provider: str, model: str, settings) -> BaseChatModel:
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, api_key=settings.ANTHROPIC_API_KEY)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, api_key=settings.OPENAI_API_KEY, temperature=0.0)
    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, temperature=0.0)
    raise ValueError(f"Unknown chat model provider: {provider}")

def get_embeddings() -> Embeddings:
    settings = get_settings()
    provider = settings.retrieval.get("embedding_provider", "openai")
    model = settings.retrieval.get("embedding_model", "text-embedding-3-large")

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model, api_key=settings.OPENAI_API_KEY)
    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=model)
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=model)
    raise ValueError(f"Unknown embeddings provider: {provider}")

def get_sparse_embeddings():
    """Sparse embeddings for Qdrant's hybrid dense+sparse retrieval"""
    from langchain_qdrant import FastEmbedSparse
    settings = get_settings()
    model = settings.retrieval.get("sparse_embedding_model", "Qdrant/bm25")
    return FastEmbedSparse(model_name=model)

def get_reranker() -> BaseDocumentCompressor:
    settings = get_settings()
    provider = settings.retrieval.get("rerank_provider", "cohere")
    model = settings.retrieval.get("rerank_model", "rerank-v3.5")

    if provider == "cohere":
        from langchain_cohere import CohereRerank
        return CohereRerank(model=model, cohere_api_key=settings.COHERE_API_KEY)
    if provider == "local_cross_encoder":
        from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
        cross_encoder = HuggingFaceCrossEncoder(model_name=model)
        return CrossEncoderReranker(model=cross_encoder, top_n=settings.retrieval.get("rerank_top_n", 5))
    raise ValueError(f"Unknown rerank provider: {provider}")