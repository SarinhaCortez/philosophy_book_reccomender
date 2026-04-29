from myrecsys.recommendation import recommend_books
from myrecsys.schemas import BookRecord, SearchResult, UserPreferenceProfile


def test_recommend_books_prefers_semantic_pipeline(monkeypatch):
    import myrecsys.recommendation as recommendation

    fake_profile = UserPreferenceProfile(
        original_prompt="stoicism grief",
        themes=["grief"],
        traditions=["stoicism"],
        semantic_query="themes: grief\ntraditions: stoicism",
    )
    fake_result = SearchResult(
        book=BookRecord(book_id="semantic-1", title="On Grief", source="openlibrary"),
        score=0.92,
        matched_fields=["semantic"],
        explanation="Semantic match on grief, stoicism.",
        match_reasons={"matched_themes": ["grief"]},
    )

    monkeypatch.setattr(
        recommendation,
        "search_semantic_catalog",
        lambda request: (
            [fake_result],
            fake_profile,
            [{"from_book_id": "semantic-1", "to_book_id": "semantic-2", "weight": 0.87}],
        ),
    )
    monkeypatch.setattr(
        recommendation,
        "enrich_books_for_display",
        lambda books: books,
    )

    payload = recommend_books("stoicism grief", max_results=3)

    assert payload["ok"] is True
    assert payload["retrieval_mode"] == "semantic"
    assert payload["user_profile"]["themes"] == ["grief"]
    assert payload["recommendations"][0]["book_id"] == "semantic-1"
    assert payload["recommendations"][0]["match_reasons"]["matched_themes"] == ["grief"]
    assert payload["graph_edges"][0]["weight"] == 0.87


def test_recommend_books_returns_error_when_semantic_pipeline_unavailable(monkeypatch):
    import myrecsys.recommendation as recommendation

    monkeypatch.setattr(
        recommendation,
        "search_semantic_catalog",
        lambda request: (_ for _ in ()).throw(recommendation.Phase2UnavailableError("semantic unavailable")),
    )

    payload = recommend_books("stoicism grief", max_results=3)

    assert payload["ok"] is False
    assert payload["retrieval_mode"] == "semantic"
    assert payload["error"] == "semantic unavailable"


def test_recommend_books_enriches_sparse_books_for_display(monkeypatch):
    import myrecsys.recommendation as recommendation

    fake_profile = UserPreferenceProfile(
        original_prompt="ethics",
        themes=["ethics"],
        semantic_query="themes: ethics",
    )
    sparse_book = BookRecord(
        book_id="semantic-2",
        title="Sparse Ethics",
        source="openlibrary",
        description="",
        first_publish_year="",
    )
    fake_result = SearchResult(
        book=sparse_book,
        score=0.88,
        matched_fields=["semantic"],
        explanation="Semantic match on ethics.",
        match_reasons={"matched_themes": ["ethics"]},
    )

    monkeypatch.setattr(
        recommendation,
        "search_semantic_catalog",
        lambda request: ([fake_result], fake_profile, []),
    )
    monkeypatch.setattr(
        recommendation,
        "enrich_books_for_display",
        lambda books: [
            BookRecord(
                **{
                    **books[0].to_dict(),
                    "description": "Filled from LLM fallback.",
                    "first_publish_year": "1991",
                }
            )
        ],
    )

    payload = recommend_books("ethics", max_results=3)

    assert payload["ok"] is True
    assert payload["recommendations"][0]["description"] == "Filled from LLM fallback."
    assert payload["recommendations"][0]["first_publish_year"] == "1991"
