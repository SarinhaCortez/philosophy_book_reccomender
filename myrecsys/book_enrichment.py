"""LLM fallback enrichment for sparse catalogue books shown in the UI."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from pydantic import BaseModel, Field

from .phase2 import build_chat_model, invoke_with_backoff
from .schemas import BookRecord


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENRICHMENT_CACHE_PATH = ROOT / "data" / "book_enrichment_cache.json"


class DisplayBookEnrichmentPayload(BaseModel):
    description: str = Field(
        default="",
        description="A concise cover-style description in 2 to 4 sentences.",
    )
    first_publish_year: str = Field(
        default="",
        description="Best-known first publication year as a 4-digit year when confidently known.",
    )


def enrich_books_for_display(
    books: list[BookRecord],
    *,
    cache_path: Path = DEFAULT_ENRICHMENT_CACHE_PATH,
) -> list[BookRecord]:
    """Fill sparse description/year fields for returned recommendations only."""

    if not books:
        return []

    cache = load_enrichment_cache(cache_path)
    updated: list[BookRecord] = []
    cache_dirty = False

    for book in books:
        if not needs_display_enrichment(book):
            updated.append(book)
            continue

        cached = cache.get(book.book_id, {})
        if cached:
            enriched = merge_book_with_enrichment(book, cached)
            updated.append(enriched)
            continue

        try:
            enrichment = fetch_display_enrichment(book)
        except Exception:
            updated.append(book)
            continue

        cache[book.book_id] = enrichment
        cache_dirty = True
        updated.append(merge_book_with_enrichment(book, enrichment))

    if cache_dirty:
        save_enrichment_cache(cache_path, cache)

    return updated


def needs_display_enrichment(book: BookRecord) -> bool:
    return not (book.description or "").strip() or not normalize_year(book.first_publish_year)


def fetch_display_enrichment(book: BookRecord) -> dict[str, str]:
    llm = build_chat_model()
    structured_llm = llm.with_structured_output(DisplayBookEnrichmentPayload)
    prompt = (
        "You are filling sparse display metadata for a philosophy book recommender UI.\n"
        "Return JSON only.\n"
        "Write a short factual cover-style description that would help a reader decide whether to open the book.\n"
        "It may use general world knowledge about the book when the title/author pair is recognizable,\n"
        "but do not fabricate specifics if you are not reasonably confident.\n"
        "For the year, return only a 4-digit first publication year when you are reasonably confident; otherwise return an empty string.\n"
        "Do not mention uncertainty in the description.\n\n"
        f"Title: {book.title}\n"
        f"Authors: {', '.join(book.authors) or 'Unknown'}\n"
        f"Subjects: {', '.join(book.subjects[:8]) or 'None'}\n"
        f"Existing description: {book.description or 'None'}\n"
        f"Existing year: {book.first_publish_year or 'Unknown'}\n"
    )
    payload = invoke_with_backoff(structured_llm, prompt)
    description = " ".join((payload.description or "").split())
    year = normalize_year(payload.first_publish_year)
    return {
        "description": description,
        "first_publish_year": year,
    }


def merge_book_with_enrichment(book: BookRecord, enrichment: dict[str, str]) -> BookRecord:
    description = (book.description or "").strip() or enrichment.get("description", "").strip()
    year = normalize_year(book.first_publish_year) or enrichment.get("first_publish_year", "")
    return replace(
        book,
        description=description,
        first_publish_year=year,
    )


def normalize_year(value: str | None) -> str:
    text = str(value or "").strip()
    return text if len(text) == 4 and text.isdigit() else ""


def load_enrichment_cache(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return {
        str(book_id): {
            "description": str(item.get("description", "")),
            "first_publish_year": normalize_year(item.get("first_publish_year")),
        }
        for book_id, item in payload.items()
        if isinstance(item, dict)
    }


def save_enrichment_cache(path: Path, cache: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
