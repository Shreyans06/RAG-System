
from enum import Enum
from dataclasses import dataclass, field
from typing import Any

class ContentType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"

@dataclass
class IngestionItem:
    id: str
    content: str
    content_type: ContentType
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class Chunk:
    id: str
    document_id: str
    parent_chunk_id: str | None
    content: str
    content_type: ContentType
    metadata: dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
    embedding: list[float] = field(default_factory=list)




