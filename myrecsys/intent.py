"""Query normalization helpers.

Phase 1/2 deliberately avoid inferred intent modelling. This module exists only
to keep user-entered text tidy before local retrieval.
"""

from __future__ import annotations


def normalize_query(text: str) -> str:
    """Collapse whitespace and preserve the user's own words."""

    return " ".join(text.split())
