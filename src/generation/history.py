from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from src.config import get_settings


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    settings = get_settings()

    return RedisChatMessageHistory(
        session_id=session_id,
        url=settings.REDIS_URL,
        ttl=settings.memory.get("session_ttl_seconds" , 1800)
    )
