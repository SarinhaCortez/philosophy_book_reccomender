"""MVP building blocks for the philosophy recommender."""

from .intent import normalize_query
from .local_catalog import iter_catalog_rows, load_catalog_books
from .phase2 import search_semantic_catalog
from .recommendation import recommend_books
from .schemas import (
    BookProfile,
    BookRecord,
    SearchRequest,
    SearchResult,
    SemanticCatalogEntry,
    UserPreferenceProfile,
)

__all__ = [
    "BookProfile",
    "BookRecord",
    "SearchRequest",
    "SearchResult",
    "SemanticCatalogEntry",
    "UserPreferenceProfile",
    "iter_catalog_rows",
    "load_catalog_books",
    "normalize_query",
    "recommend_books",
    "search_semantic_catalog",
]
