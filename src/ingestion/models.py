
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ContentType(StrEnum):
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
@dataclass
class IngestResult:
    chunks: list[Chunk] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
