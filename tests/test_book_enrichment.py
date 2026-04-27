from pathlib import Path

from myrecsys.book_enrichment import enrich_books_for_display
from myrecsys.schemas import BookRecord


def test_enrich_books_for_display_fills_missing_description_and_year(tmp_path, monkeypatch):
    import myrecsys.book_enrichment as book_enrichment

    book = BookRecord(
        book_id="book-1",
        title="Sparse Book",
        authors=["A. Author"],
        subjects=["Philosophy"],
        description="",
        first_publish_year="",
    )

    monkeypatch.setattr(
        book_enrichment,
        "fetch_display_enrichment",
        lambda _: {
            "description": "A concise cover-style description.",
            "first_publish_year": "1984",
        },
    )

    cache_path = tmp_path / "enrichment-cache.json"
    enriched = enrich_books_for_display([book], cache_path=cache_path)

    assert enriched[0].description == "A concise cover-style description."
    assert enriched[0].first_publish_year == "1984"
    assert cache_path.exists()


def test_enrich_books_for_display_uses_cache_before_llm(tmp_path, monkeypatch):
    import myrecsys.book_enrichment as book_enrichment

    book = BookRecord(
        book_id="book-2",
        title="Cached Book",
        authors=["A. Author"],
        subjects=["Philosophy"],
        description="",
        first_publish_year="",
    )
    cache_path = tmp_path / "enrichment-cache.json"
    cache_path.write_text(
        '{"book-2": {"description": "Cached description.", "first_publish_year": "1972"}}',
        encoding="utf-8",
    )

    def fail_if_called(_: BookRecord) -> dict[str, str]:
        raise AssertionError("LLM fallback should not be called when cache exists")

    monkeypatch.setattr(book_enrichment, "fetch_display_enrichment", fail_if_called)

    enriched = enrich_books_for_display([book], cache_path=cache_path)

    assert enriched[0].description == "Cached description."
    assert enriched[0].first_publish_year == "1972"
