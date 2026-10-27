from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel


# --- LLM response models (Pydantic for validation) ---

class ExtractedEntityRaw(BaseModel):
    text: str
    token_prefix: str
    entity_type: Literal["PERSON", "ORG", "ADDRESS", "DATE", "MONEY", "ID"]
    alias_of: str | None = None
    reasoning: str
    confidence: Literal["HIGH", "MEDIUM", "LOW"]


class ExtractionResponse(BaseModel):
    entities: list[ExtractedEntityRaw]


# --- Internal pipeline models ---

@dataclass
class Chunk:
    text: str
    char_offset: int  # absolute start position in original document
    chunk_index: int


@dataclass
class ExtractedEntity:
    text: str
    token_prefix: str
    entity_type: str
    alias_of: str | None
    reasoning: str
    confidence: str  # HIGH | MEDIUM | LOW
    char_start: int  # absolute position in original doc
    char_end: int


@dataclass
class ResolvedEntity:
    canonical_text: str
    token_prefix: str
    entity_type: str
    token: str | None  # assigned after store lookup
    confidence: str
    reasoning: str


@dataclass
class ReplacementSpan:
    char_start: int
    char_end: int
    original_text: str
    replacement_token: str
