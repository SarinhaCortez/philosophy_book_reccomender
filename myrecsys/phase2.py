"""Phase-2 semantic recommendation pipeline.

This recommender replaces the old keyword scorer with three simple pieces:

1. LLM-extracted structured book metadata
2. LLM-extracted structured user preference profiles
3. Embedding-based matching between them
"""

from __future__ import annotations

import json
import math
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field

from .local_catalog import _book_from_row
from .schemas import (
    BookProfile,
    BookRecord,
    SearchRequest,
    SearchResult,
    SemanticCatalogEntry,
    UserPreferenceProfile,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEMANTIC_INDEX_PATH = ROOT / "data" / "semantic_index.jsonl"
DEFAULT_PHASE2_CHAT_MODEL = os.environ.get("MYRECSYS_PHASE2_CHAT_MODEL", "gemini-2.5-flash-lite")
DEFAULT_PHASE2_EMBEDDING_MODEL = os.environ.get(
    "MYRECSYS_PHASE2_EMBEDDING_MODEL",
    "gemini-embedding-001",
)
DEFAULT_PHASE2_MAX_RETRIES = int(os.environ.get("MYRECSYS_PHASE2_MAX_RETRIES", "4"))
DEFAULT_PHASE2_RETRY_BASE_SECONDS = float(
    os.environ.get("MYRECSYS_PHASE2_RETRY_BASE_SECONDS", "15")
)


class Phase2UnavailableError(RuntimeError):
    """Raised when the optional semantic stack is not available."""


class BookProfilePayload(BaseModel):
    summary: str = Field(default="", description="Two sentence factual summary of the book.")
    themes: list[str] = Field(default_factory=list, description="Core philosophical or intellectual themes.")
    traditions: list[str] = Field(default_factory=list, description="Relevant philosophical traditions or schools.")
    reader_moods: list[str] = Field(default_factory=list, description="Kinds of reader states this book suits.")
    style_descriptors: list[str] = Field(default_factory=list, description="Descriptors of the writing style.")
    notable_people: list[str] = Field(default_factory=list, description="Named thinkers or authors explicitly central to the book.")
    difficulty: str = Field(default="", description="low, medium, or high if inferable from the metadata.")
    era: str = Field(default="", description="Historical era or orientation if grounded in the metadata.")


class UserPreferenceProfilePayload(BaseModel):
    reading_goals: list[str] = Field(default_factory=list, description="What the user hopes to get from the reading.")
    themes: list[str] = Field(default_factory=list, description="Themes the user is seeking.")
    traditions: list[str] = Field(default_factory=list, description="Traditions or schools the user wants.")
    liked_authors: list[str] = Field(default_factory=list, description="Authors or thinkers the user likes.")
    avoided_authors: list[str] = Field(default_factory=list, description="Authors or thinkers the user wants to avoid.")
    desired_qualities: list[str] = Field(default_factory=list, description="Desired tone, style, or other qualities.")
    avoided_qualities: list[str] = Field(default_factory=list, description="Tones, styles, or traits the user wants to avoid.")
    difficulty: str = Field(default="", description="Preferred difficulty level if stated or implied.")


def search_semantic_catalog(
    request: SearchRequest,
    *,
    semantic_index_path: Path = DEFAULT_SEMANTIC_INDEX_PATH,
    query_profile: UserPreferenceProfile | None = None,
    query_vector: list[float] | None = None,
    max_results: int | None = None,
) -> tuple[list[SearchResult], UserPreferenceProfile, list[dict[str, Any]]]:
    """Search the semantic index using embeddings only."""

    if not semantic_index_path.exists():
        raise Phase2UnavailableError(
            f"Semantic index not found at {semantic_index_path}. "
            "Build it with scripts/build_semantic_index.py."
        )

    if query_profile is None:
        query_profile = extract_user_preference_profile(request.query)
    if query_vector is None:
        query_vector = embed_texts(
            [profile_to_semantic_query(query_profile)],
            task_type="retrieval_query",
        )[0]

    entries = list(iter_semantic_index(semantic_index_path))
    scored: list[tuple[SearchResult, list[float]]] = []
    for entry in entries:
        similarity = cosine_similarity(query_vector, entry.embedding)
        if similarity <= 0:
            continue
        match_reasons = collect_semantic_match_reasons(entry.book, entry.profile, query_profile)
        explanation = explain_semantic_match(entry.book, entry.profile, query_profile, match_reasons)
        scored.append(
            (
                SearchResult(
                book=entry.book,
                score=similarity,
                matched_fields=["semantic"],
                explanation=explanation,
                match_reasons=match_reasons,
                ),
                entry.embedding,
            )
        )

    scored.sort(key=lambda item: item[0].score, reverse=True)
    selected_pairs = scored[: max_results or request.max_results]
    return [result for result, _ in selected_pairs], query_profile, build_knn_graph_edges(selected_pairs)


def iter_semantic_index(path: Path) -> Iterable[SemanticCatalogEntry]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            yield SemanticCatalogEntry(
                book=_book_from_row(row["book"], source_query=""),
                profile=BookProfile(**row["profile"]),
                embedding=[float(value) for value in row["embedding"]],
            )


def extract_book_profile(
    book: BookRecord,
    *,
    model_name: str = DEFAULT_PHASE2_CHAT_MODEL,
) -> BookProfile:
    """Use an LLM to build structured metadata for a book."""

    llm = build_chat_model(model_name)
    structured_llm = llm.with_structured_output(BookProfilePayload)
    prompt = build_book_profile_prompt(book)
    payload = invoke_with_backoff(structured_llm, prompt)
    profile = book_profile_from_payload(book, payload)
    profile.semantic_text = build_book_semantic_text(book, profile)
    return profile


def extract_user_preference_profile(
    prompt: str,
    *,
    model_name: str = DEFAULT_PHASE2_CHAT_MODEL,
) -> UserPreferenceProfile:
    """Use an LLM to turn a free-text prompt into a structured profile."""

    return _extract_user_preference_profile_cached(prompt, model_name)


@lru_cache(maxsize=256)
def _extract_user_preference_profile_cached(
    prompt: str,
    model_name: str,
) -> UserPreferenceProfile:
    llm = build_chat_model(model_name)
    structured_llm = llm.with_structured_output(UserPreferenceProfilePayload)
    extraction_prompt = (
        "You are extracting a structured reading-preference profile for a philosophy recommender.\n"
        "Stay close to the user's wording.\n"
        "Use empty lists instead of guessing.\n"
        "Capture reading goals, themes, traditions, liked or avoided authors,\n"
        "desired or avoided qualities, and difficulty if implied.\n\n"
        f"User prompt: {prompt}"
    )
    payload = invoke_with_backoff(structured_llm, extraction_prompt)
    profile = UserPreferenceProfile(
        original_prompt=prompt,
        reading_goals=payload.reading_goals,
        themes=payload.themes,
        traditions=payload.traditions,
        liked_authors=payload.liked_authors,
        avoided_authors=payload.avoided_authors,
        desired_qualities=payload.desired_qualities,
        avoided_qualities=payload.avoided_qualities,
        difficulty=payload.difficulty,
    )
    profile.semantic_query = profile_to_semantic_query(profile)
    return profile


def build_chat_model(model_name: str = DEFAULT_PHASE2_CHAT_MODEL) -> Any:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise Phase2UnavailableError(
            "Phase 2 requires the optional dependency `langchain-google-genai`."
        ) from exc

    if not os.environ.get("GOOGLE_API_KEY"):
        raise Phase2UnavailableError("Phase 2 requires GOOGLE_API_KEY.")

    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0,
    )


def build_embeddings_model(model_name: str = DEFAULT_PHASE2_EMBEDDING_MODEL) -> Any:
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
    except ImportError as exc:
        raise Phase2UnavailableError(
            "Phase 2 requires the optional dependency `langchain-google-genai`."
        ) from exc

    if not os.environ.get("GOOGLE_API_KEY"):
        raise Phase2UnavailableError("Phase 2 requires GOOGLE_API_KEY.")

    return GoogleGenerativeAIEmbeddings(model=model_name)


def embed_texts(
    texts: list[str],
    *,
    model_name: str = DEFAULT_PHASE2_EMBEDDING_MODEL,
    task_type: str | None = None,
) -> list[list[float]]:
    if not texts:
        return []
    embeddings = build_embeddings_model(model_name)
    return embed_documents_with_backoff(embeddings, texts, task_type=task_type)


def build_book_profile_prompt(book: BookRecord) -> str:
    return (
        "You are enriching a philosophy-book recommendation catalogue.\n"
        "Build structured metadata for retrieval.\n"
        "Use only evidence from the provided metadata.\n"
        "Do not invent philosophical traditions or named thinkers unless they are supported by the text.\n"
        "Use short retrieval-friendly phrases.\n"
        "If something is unclear, leave it empty.\n\n"
        f"Title: {book.title}\n"
        f"Authors: {', '.join(book.authors) or 'Unknown'}\n"
        f"Subjects: {', '.join(book.subjects) or 'None'}\n"
        f"Description: {book.description or 'None'}\n"
        f"Publication: {book.first_publish_year or 'Unknown'}\n"
    )


def book_profile_response_json_schema() -> dict[str, Any]:
    return BookProfilePayload.model_json_schema()


def book_profile_from_payload(
    book: BookRecord,
    payload: BookProfilePayload | dict[str, Any],
) -> BookProfile:
    payload_model = (
        payload if isinstance(payload, BookProfilePayload) else BookProfilePayload.model_validate(payload)
    )
    return BookProfile(
        book_id=book.book_id,
        summary=payload_model.summary,
        themes=payload_model.themes,
        traditions=payload_model.traditions,
        reader_moods=payload_model.reader_moods,
        style_descriptors=payload_model.style_descriptors,
        notable_people=payload_model.notable_people,
        difficulty=payload_model.difficulty,
        era=payload_model.era,
    )


def invoke_with_backoff(
    runnable: Any,
    prompt: str,
    *,
    max_retries: int = DEFAULT_PHASE2_MAX_RETRIES,
    base_delay_seconds: float = DEFAULT_PHASE2_RETRY_BASE_SECONDS,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            return runnable.invoke(prompt)
        except Exception as exc:  # pragma: no cover - depends on provider behavior
            last_error = exc
            if not should_retry_provider_error(exc) or attempt == max_retries - 1:
                raise
            time.sleep(base_delay_seconds * (attempt + 1))
    if last_error is not None:
        raise last_error
    raise RuntimeError("LLM call failed without raising a provider error.")


def embed_documents_with_backoff(
    embeddings: Any,
    texts: list[str],
    *,
    task_type: str | None = None,
    max_retries: int = DEFAULT_PHASE2_MAX_RETRIES,
    base_delay_seconds: float = DEFAULT_PHASE2_RETRY_BASE_SECONDS,
) -> list[list[float]]:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            return embeddings.embed_documents(texts, task_type=task_type)
        except Exception as exc:  # pragma: no cover - depends on provider behavior
            last_error = exc
            if not should_retry_provider_error(exc) or attempt == max_retries - 1:
                raise
            time.sleep(base_delay_seconds * (attempt + 1))
    if last_error is not None:
        raise last_error
    raise RuntimeError("Embedding call failed without raising a provider error.")


def should_retry_provider_error(exc: Exception) -> bool:
    message = str(exc).casefold()
    return any(
        token in message
        for token in (
            "429",
            "resource_exhausted",
            "quota",
            "rate limit",
            "temporarily unavailable",
            "deadline exceeded",
            "timed out",
        )
    )


def build_book_semantic_text(book: BookRecord, profile: BookProfile) -> str:
    parts = [
        f"title: {book.title}",
        f"authors: {', '.join(book.authors)}" if book.authors else "",
        f"summary: {profile.summary}" if profile.summary else "",
        f"themes: {', '.join(profile.themes)}" if profile.themes else "",
        f"traditions: {', '.join(profile.traditions)}" if profile.traditions else "",
        f"reader moods: {', '.join(profile.reader_moods)}" if profile.reader_moods else "",
        f"style: {', '.join(profile.style_descriptors)}" if profile.style_descriptors else "",
        f"difficulty: {profile.difficulty}" if profile.difficulty else "",
        f"era: {profile.era}" if profile.era else "",
        f"subjects: {', '.join(book.subjects[:8])}" if book.subjects else "",
        f"description: {book.description[:1200]}" if book.description else "",
    ]
    return "\n".join(part for part in parts if part.strip())


def profile_to_semantic_query(profile: UserPreferenceProfile) -> str:
    parts = [
        f"prompt: {profile.original_prompt}" if profile.original_prompt else "",
        f"goals: {', '.join(profile.reading_goals)}" if profile.reading_goals else "",
        f"themes: {', '.join(profile.themes)}" if profile.themes else "",
        f"traditions: {', '.join(profile.traditions)}" if profile.traditions else "",
        f"liked authors: {', '.join(profile.liked_authors)}" if profile.liked_authors else "",
        f"avoid authors: {', '.join(profile.avoided_authors)}" if profile.avoided_authors else "",
        f"desired qualities: {', '.join(profile.desired_qualities)}" if profile.desired_qualities else "",
        f"avoid qualities: {', '.join(profile.avoided_qualities)}" if profile.avoided_qualities else "",
        f"difficulty: {profile.difficulty}" if profile.difficulty else "",
    ]
    return "\n".join(part for part in parts if part.strip())


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def build_knn_graph_edges(
    selected_pairs: list[tuple[SearchResult, list[float]]],
    *,
    neighbors_per_node: int = 2,
) -> list[dict[str, Any]]:
    if len(selected_pairs) < 2:
        return []

    edge_weights: dict[tuple[str, str], float] = {}
    for index, (result, embedding) in enumerate(selected_pairs):
        neighbors: list[tuple[float, str]] = []
        for other_index, (other_result, other_embedding) in enumerate(selected_pairs):
            if index == other_index:
                continue
            similarity = cosine_similarity(embedding, other_embedding)
            if similarity <= 0:
                continue
            neighbors.append((similarity, other_result.book.book_id))

        neighbors.sort(key=lambda item: item[0], reverse=True)
        for similarity, other_book_id in neighbors[:neighbors_per_node]:
            edge_key = tuple(sorted((result.book.book_id, other_book_id)))
            edge_weights[edge_key] = max(edge_weights.get(edge_key, 0.0), similarity)

    return [
        {"from_book_id": left, "to_book_id": right, "weight": weight}
        for (left, right), weight in sorted(edge_weights.items(), key=lambda item: item[1], reverse=True)
    ]


def collect_semantic_match_reasons(
    book: BookRecord,
    book_profile: BookProfile,
    user_profile: UserPreferenceProfile,
) -> dict[str, Any]:
    return {
        "matched_themes": shared_items(book_profile.themes, user_profile.themes),
        "matched_traditions": shared_items(book_profile.traditions, user_profile.traditions),
        "matched_reader_moods": shared_items(book_profile.reader_moods, user_profile.desired_qualities),
        "matched_styles": shared_items(book_profile.style_descriptors, user_profile.desired_qualities),
        "matched_authors": shared_items(book.authors, user_profile.liked_authors),
        "matched_notable_people": shared_items(book_profile.notable_people, user_profile.liked_authors),
        "supporting_themes": book_profile.themes[:3],
        "supporting_traditions": book_profile.traditions[:2],
        "supporting_styles": book_profile.style_descriptors[:2],
        "difficulty_relation": describe_difficulty_relation(user_profile.difficulty, book_profile.difficulty),
        "summary": book_profile.summary,
    }


def explain_semantic_match(
    book: BookRecord,
    book_profile: BookProfile,
    user_profile: UserPreferenceProfile,
    match_reasons: dict[str, Any],
) -> str:
    direct_matches = dedupe_preserve_order(
        match_reasons.get("matched_authors", [])
        + match_reasons.get("matched_notable_people", [])
        + match_reasons.get("matched_themes", [])
        + match_reasons.get("matched_traditions", [])
    )
    support_bits: list[str] = []
    if direct_matches:
        support_bits.append(f"your prompt lines up with {human_join(direct_matches[:3])}")
    if match_reasons.get("matched_reader_moods"):
        support_bits.append(
            f"it fits the tone of {human_join(match_reasons['matched_reader_moods'][:2])}"
        )
    elif match_reasons.get("matched_styles"):
        support_bits.append(
            f"its style comes across as {human_join(match_reasons['matched_styles'][:2])}"
        )
    difficulty_relation = str(match_reasons.get("difficulty_relation", "")).strip()
    if difficulty_relation:
        support_bits.append(difficulty_relation)

    if support_bits:
        first_sentence = "This is a strong match because " + "; ".join(support_bits) + "."
    else:
        first_sentence = "This recommendation comes from a close semantic fit between your prompt and the book profile."

    summary = str(match_reasons.get("summary", "")).strip()
    if summary:
        second_sentence = summary if summary[-1] in ".!?" else f"{summary}."
    else:
        metadata_bits: list[str] = []
        if match_reasons.get("supporting_traditions"):
            metadata_bits.append(f"it sits in {human_join(match_reasons['supporting_traditions'])}")
        if match_reasons.get("supporting_themes"):
            metadata_bits.append(f"it focuses on {human_join(match_reasons['supporting_themes'])}")
        if match_reasons.get("supporting_styles"):
            metadata_bits.append(f"the writing reads as {human_join(match_reasons['supporting_styles'])}")
        second_sentence = (
            "Within the catalogue, " + "; ".join(metadata_bits) + "."
            if metadata_bits
            else "The available metadata still places it close to the ideas in your prompt."
        )

    return f"{first_sentence} {second_sentence}".strip()


def describe_difficulty_relation(requested: str, available: str) -> str:
    requested_value = normalize_difficulty_label(requested)
    available_value = normalize_difficulty_label(available)
    if not requested_value or not available_value:
        return ""
    if requested_value == available_value:
        return f"its difficulty is aligned at a {available_value} level"
    scale = {"low": 1, "medium": 2, "high": 3}
    difference = scale[available_value] - scale[requested_value]
    if difference > 0:
        return f"it is somewhat more demanding than your requested {requested_value} level"
    return f"it is somewhat more accessible than your requested {requested_value} level"


def normalize_difficulty_label(value: str) -> str:
    normalized = value.strip().casefold()
    return normalized if normalized in {"low", "medium", "high"} else ""


def human_join(values: list[str]) -> str:
    cleaned = [value.strip() for value in values if value and value.strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"


def shared_items(left: list[str], right: list[str]) -> list[str]:
    right_set = {value.casefold() for value in right}
    return [value for value in left if value.casefold() in right_set]


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result
