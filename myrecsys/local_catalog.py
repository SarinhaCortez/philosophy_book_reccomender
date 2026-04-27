"""Local Open Library catalogue loading helpers.

The phase-1 lexical retriever has been retired. This module now owns only
catalogue access and record normalization used by the semantic pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .schemas import BookRecord


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CATALOG_PATH = ROOT / "data" / "books.jsonl"


class CatalogNotFoundError(RuntimeError):
    """Raised when the local book catalogue has not been built yet."""


def iter_catalog_rows(path: Path = DEFAULT_CATALOG_PATH) -> Iterable[dict[str, Any]]:
    """Yield raw JSON rows from the local catalogue."""

    if not path.exists():
        raise CatalogNotFoundError(
            f"Local catalogue not found at {path}. "
            "Build it from Open Library dumps with scripts/build_openlibrary_catalog.py."
        )

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)


def load_catalog_books(path: Path = DEFAULT_CATALOG_PATH) -> list[BookRecord]:
    """Load the local catalogue into normalized book records."""

    return [_book_from_row(row, source_query="") for row in iter_catalog_rows(path)]


def _book_from_row(row: dict[str, Any], *, source_query: str) -> BookRecord:
    subjects = row.get("subjects", row.get("categories", []))
    return BookRecord(
        book_id=str(row.get("book_id", "")),
        title=str(row.get("title", "")),
        authors=[str(author) for author in row.get("authors", [])],
        description=str(row.get("description", "")),
        subjects=[str(subject) for subject in subjects],
        first_publish_year=str(row.get("first_publish_year", row.get("published_date", ""))),
        language=str(row.get("language", "")),
        ratings_count=_optional_int(row.get("ratings_count")) or 0,
        page_count=_optional_int(row.get("page_count")),
        thumbnail_url=str(row.get("thumbnail_url", "")),
        preview_link=str(row.get("preview_link", "")),
        source=str(row.get("source", "openlibrary")),
        source_query=source_query,
    )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
