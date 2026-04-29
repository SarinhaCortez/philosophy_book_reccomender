"""Data contracts for the philosophy recommender."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class BookRecord:
    """One local catalogue record, currently generated from Open Library works."""

    book_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    description: str = ""
    subjects: list[str] = field(default_factory=list)
    language: str = ""
    ratings_count: int = 0
    first_publish_year: str = ""
    page_count: int | None = None
    thumbnail_url: str = ""
    preview_link: str = ""
    source: str = "openlibrary"
    source_query: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SearchRequest:
    """User supplied retrieval request. No inferred preferences."""

    query: str
    subjects: list[str] = field(default_factory=list)
    authors: list[str] = field(default_factory=list)
    max_results: int = 12

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SearchResult:
    """A scored local catalogue hit."""

    book: BookRecord
    score: float
    matched_fields: list[str] = field(default_factory=list)
    explanation: str = ""
    match_reasons: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


BookCandidate = BookRecord


@dataclass(slots=True)
class BookProfile:
    """LLM-extracted structured understanding of a book."""

    book_id: str = ""
    summary: str = ""
    themes: list[str] = field(default_factory=list)
    traditions: list[str] = field(default_factory=list)
    reader_moods: list[str] = field(default_factory=list)
    style_descriptors: list[str] = field(default_factory=list)
    notable_people: list[str] = field(default_factory=list)
    difficulty: str = ""
    era: str = ""
    semantic_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class UserPreferenceProfile:
    """LLM-extracted reading preference profile for a prompt."""

    original_prompt: str = ""
    reading_goals: list[str] = field(default_factory=list)
    themes: list[str] = field(default_factory=list)
    traditions: list[str] = field(default_factory=list)
    liked_authors: list[str] = field(default_factory=list)
    avoided_authors: list[str] = field(default_factory=list)
    desired_qualities: list[str] = field(default_factory=list)
    avoided_qualities: list[str] = field(default_factory=list)
    difficulty: str = ""
    semantic_query: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SemanticCatalogEntry:
    """Precomputed phase-2 semantic search record."""

    book: BookRecord
    profile: BookProfile
    embedding: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


BOOK_RECORD_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "BookRecord",
    "type": "object",
    "additionalProperties": False,
    "required": ["book_id", "title"],
    "properties": {
        "book_id": {"type": "string"},
        "title": {"type": "string"},
        "authors": {"type": "array", "items": {"type": "string"}},
        "description": {"type": "string"},
        "subjects": {"type": "array", "items": {"type": "string"}},
        "language": {"type": "string"},
        "ratings_count": {"type": "integer", "minimum": 0},
        "first_publish_year": {"type": "string"},
        "page_count": {"type": ["integer", "null"], "minimum": 1},
        "thumbnail_url": {"type": "string"},
        "preview_link": {"type": "string"},
        "source": {"type": "string"},
        "source_query": {"type": "string"},
    },
}


SEARCH_REQUEST_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "SearchRequest",
    "type": "object",
    "additionalProperties": False,
    "required": ["query"],
    "properties": {
        "query": {"type": "string"},
        "subjects": {"type": "array", "items": {"type": "string"}},
        "authors": {"type": "array", "items": {"type": "string"}},
        "max_results": {"type": "integer", "minimum": 1, "maximum": 100},
    },
}


BOOK_PROFILE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "BookProfile",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "book_id": {"type": "string"},
        "summary": {"type": "string"},
        "themes": {"type": "array", "items": {"type": "string"}},
        "traditions": {"type": "array", "items": {"type": "string"}},
        "reader_moods": {"type": "array", "items": {"type": "string"}},
        "style_descriptors": {"type": "array", "items": {"type": "string"}},
        "notable_people": {"type": "array", "items": {"type": "string"}},
        "difficulty": {"type": "string"},
        "era": {"type": "string"},
        "semantic_text": {"type": "string"},
    },
}


USER_PREFERENCE_PROFILE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "UserPreferenceProfile",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "original_prompt": {"type": "string"},
        "reading_goals": {"type": "array", "items": {"type": "string"}},
        "themes": {"type": "array", "items": {"type": "string"}},
        "traditions": {"type": "array", "items": {"type": "string"}},
        "liked_authors": {"type": "array", "items": {"type": "string"}},
        "avoided_authors": {"type": "array", "items": {"type": "string"}},
        "desired_qualities": {"type": "array", "items": {"type": "string"}},
        "avoided_qualities": {"type": "array", "items": {"type": "string"}},
        "difficulty": {"type": "string"},
        "semantic_query": {"type": "string"},
    },
}
