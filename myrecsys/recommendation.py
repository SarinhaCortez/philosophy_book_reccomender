"""End-to-end semantic recommendation orchestration."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .book_enrichment import enrich_books_for_display
from .intent import normalize_query
from .local_catalog import CatalogNotFoundError
from .phase2 import Phase2UnavailableError, search_semantic_catalog
from .schemas import SearchRequest


def recommend_books(
    prompt: str,
    *,
    max_results: int = 12,
) -> dict[str, Any]:
    """Recommend books with the phase-2 semantic pipeline only."""

    query = normalize_query(prompt)
    request = SearchRequest(query=query, max_results=max_results)
    retrieval_mode = "semantic"
    user_profile: dict[str, Any] = {}

    try:
        results, extracted_profile = search_semantic_catalog(request)
        user_profile = extracted_profile.to_dict()
        enriched_books = enrich_books_for_display([result.book for result in results])
        results = [
            type(result)(
                book=enriched_books[index],
                score=result.score,
                matched_fields=result.matched_fields,
                explanation=result.explanation,
            )
            for index, result in enumerate(results)
        ]
    except (CatalogNotFoundError, Phase2UnavailableError) as exc:
        return {
            "ok": False,
            "error": str(exc),
            "message": str(exc),
            "search_request": request.to_dict(),
            "retrieval_query": query,
            "retrieval_mode": retrieval_mode,
            "user_profile": user_profile,
            "recommendations": [],
        }

    return {
        "ok": True,
        "error": "",
        "search_request": request.to_dict(),
        "retrieval_query": query,
        "retrieval_mode": retrieval_mode,
        "user_profile": user_profile,
        "recommendations": [
            {
                **asdict(result.book),
                "score": result.score,
                "matched_fields": result.matched_fields,
                "why_recommended": result.explanation,
            }
            for result in results
        ],
    }
