#!/usr/bin/env python3
"""Build a local philosophy catalogue from Open Library works/authors/ratings dumps.

Open Library dump lines are tab-separated:
type, key, revision, last_modified, json_payload
"""

from __future__ import annotations

import argparse
import gzip
import heapq
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DUMPS_DIR = Path("dumps")
WORKS_PATTERNS = ("ol_dump_works*.txt.gz", "ol_dump_works*.txt")
AUTHORS_PATTERNS = ("ol_dump_authors*.txt.gz", "ol_dump_authors*.txt")
RATINGS_PATTERNS = ("ol_dump_ratings*.txt.gz", "ol_dump_ratings*.txt")

STRONG_PHILOSOPHY_TERMS = {
    "aesthetics",
    "buddhism",
    "confucianism",
    "continental philosophy",
    "daoism",
    "epistemology",
    "existentialism",
    "history of philosophy",
    "metaphysics",
    "moral philosophy",
    "phenomenology",
    "philosophers",
    "political philosophy",
    "pragmatism",
    "stoicism",
    "taoism",
}

WEAK_PHILOSOPHY_TERMS = {
    "analytic philosophy",
    "ethics",
    "philosophy",
}

DENY_SUBJECT_TERMS = {
    "accounting",
    "adventure",
    "animals",
    "applied psychology",
    "behavior modification",
    "brainwashing",
    "business",
    "children s fiction",
    "children",
    "childhood",
    "classics",
    "conduct of life",
    "consumption economics",
    "courage",
    "dystopias",
    "education",
    "fantasy",
    "fantasy fiction",
    "fiction",
    "finance",
    "friendship",
    "genetic engineering",
    "home schooling",
    "human behavior",
    "identity psychology",
    "imagination play",
    "interpersonal communication",
    "interpersonal relations",
    "juvenile fiction",
    "juvenile literature",
    "leadership",
    "love",
    "marketing",
    "memory",
    "money",
    "motivation psychology",
    "personal",
    "political science",
    "popular works",
    "propaganda",
    "psychology",
    "quality of life",
    "relationships",
    "religious life",
    "research",
    "science fiction",
    "science",
    "self help techniques",
    "self improvement",
    "social classes",
    "social control",
    "social problems",
    "spirituality",
    "stress management",
    "success",
    "survival",
    "television journalists",
    "values juvenile literature",
    "wealth management",
}



NORMALIZED_STRONG_TERMS = tuple()
NORMALIZED_WEAK_TERMS = tuple()
NORMALIZED_DENY_SUBJECT_TERMS = tuple()
NORMALIZED_COMMENTARY_SUBJECT_TERMS = tuple()

COMMENTARY_TITLE_PATTERNS = (
    "a little history of",
    "history of",
    "story of",
    "introduction to",
    "readings in",
    "understanding",
    "conversations with",
    "autobiography",
    "biography",
    "the worldly philosophers",
    "at the existentialist cafe",
)

COMMENTARY_SUBJECT_TERMS = {
    "anthologie",
    "autobiographies",
    "autobiography",
    "biographies",
    "biography",
    "catalogs",
    "catalogues",
    "commentaries",
    "criticism",
    "criticism and interpretation",
    "history and criticism",
    "introductions",
    "readers",
    "reference",
    "textbooks",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract the most-rated philosophy works from Open Library dumps."
    )
    parser.add_argument("--dumps-dir", type=Path, default=DEFAULT_DUMPS_DIR)
    parser.add_argument("--output", type=Path, default=Path("data/books.jsonl"))
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--progress-every", type=int, default=200000)
    parser.add_argument("--checkpoint", type=Path, default=Path("data/books.catalog.checkpoint.json"))
    args = parser.parse_args()

    initialize_term_indexes()
    works_path = find_single_dump(args.dumps_dir, WORKS_PATTERNS, "works")
    authors_path = find_single_dump(args.dumps_dir, AUTHORS_PATTERNS, "authors")
    ratings_path = find_single_dump(args.dumps_dir, RATINGS_PATTERNS, "ratings")

    ratings_counts = load_ratings_counts(ratings_path)
    rows, author_keys = collect_work_rows(
        works_path,
        ratings_counts=ratings_counts,
        limit=args.limit,
        progress_every=max(1, args.progress_every),
        checkpoint_path=args.checkpoint,
    )
    author_names = load_author_names(authors_path, wanted_keys=author_keys)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as output:
        hydrated_rows: list[dict[str, Any]] = []
        for row in rows:
            row["authors"] = primary_author_list(
                [
                    author_names.get(key, key.removeprefix("/authors/"))
                    for key in row.pop("author_keys", [])
                ]
            )
            hydrated_rows.append(row)

        cleaned_rows = clean_catalog_rows(hydrated_rows, limit=args.limit)
        for row in cleaned_rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(cleaned_rows)} cleaned records to {args.output}")
    return 0


def find_single_dump(dumps_dir: Path, patterns: tuple[str, ...], label: str) -> Path:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sorted(dumps_dir.glob(pattern)))

    if not matches:
        expected = ", ".join(patterns)
        raise SystemExit(f"Missing Open Library {label} dump in {dumps_dir}. Expected one of: {expected}")
    if len(matches) > 1:
        names = ", ".join(str(path) for path in matches)
        raise SystemExit(f"Found multiple Open Library {label} dumps. Keep only one in {dumps_dir}: {names}")
    return matches[0]


def collect_work_rows(
    path: Path,
    *,
    ratings_counts: dict[str, int],
    limit: int,
    progress_every: int,
    checkpoint_path: Path,
) -> tuple[list[dict[str, Any]], set[str]]:
    checkpoint = load_checkpoint(checkpoint_path, works_path=path, limit=limit)
    selected_by_title = checkpoint["selected_by_title"]
    heap = [(rating_count, title_key) for title_key, (rating_count, _) in selected_by_title.items()]
    heapq.heapify(heap)
    seen_titles: set[str] = set(selected_by_title)
    processed = checkpoint["processed"]
    start_cursor = checkpoint["cursor"]

    if processed:
        print(
            f"Resuming from checkpoint at {processed:,} works with "
            f"{len(selected_by_title):,} retained candidates.",
            flush=True,
        )

    for payload, cursor in iter_dump_payloads(path, start_cursor=start_cursor):
        processed += 1
        if processed % progress_every == 0:
            print(
                f"Processed {processed:,} works; current selected candidates: {len(selected_by_title):,}",
                flush=True,
            )
            save_checkpoint(
                checkpoint_path,
                works_path=path,
                processed=processed,
                cursor=cursor,
                limit=limit,
                selected_by_title=selected_by_title,
            )
        work_key = str(payload.get("key", ""))
        rating_count = ratings_counts.get(work_key, 0)
        row = normalize_work(payload)
        if not row:
            continue
        title_key = row["title"].casefold()
        if title_key in seen_titles and rating_count <= selected_by_title.get(title_key, (0, {}))[0]:
            continue
        seen_titles.add(title_key)
        row["ratings_count"] = rating_count
        selected_by_title[title_key] = (rating_count, row)

        if rating_count > 0:
            heapq.heappush(heap, (rating_count, title_key))
            if len(heap) > limit:
                _, removed_title = heapq.heappop(heap)
                if removed_title in selected_by_title and selected_by_title[removed_title][0] > 0:
                    del selected_by_title[removed_title]

    rows = [row for _, row in selected_by_title.values()]
    rows.sort(key=lambda item: (-int(item.get("ratings_count", 0)), item["title"].casefold()))
    selected = rows[:limit]
    clear_checkpoint(checkpoint_path)
    selected_author_keys = {
        key
        for row in selected
        for key in row.get("author_keys", [])
    }
    return selected, selected_author_keys


def load_ratings_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if not parts or not parts[0].startswith("/works/"):
                continue
            work_key = parts[0]
            counts[work_key] = counts.get(work_key, 0) + 1
    return counts


def load_author_names(path: Path, *, wanted_keys: set[str]) -> dict[str, str]:
    names: dict[str, str] = {}
    for payload, _ in iter_dump_payloads(path):
        key = payload.get("key")
        if key not in wanted_keys:
            continue
        name = payload.get("name") or payload.get("personal_name")
        if key and name:
            names[str(key)] = str(name)
        if len(names) >= len(wanted_keys):
            break
    return names


def iter_dump_payloads(path: Path, *, start_cursor: int = 0) -> Iterable[tuple[dict[str, Any], int]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        if start_cursor:
            handle.seek(start_cursor)
        while True:
            line = handle.readline()
            if not line:
                break
            cursor = handle.tell()
            parts = line.rstrip("\n").split("\t", 4)
            if len(parts) != 5:
                continue
            try:
                yield json.loads(parts[4]), cursor
            except json.JSONDecodeError:
                continue


def normalize_work(
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    subjects = [str(subject) for subject in payload.get("subjects", [])]
    title = str(payload.get("title", "")).strip()
    if not title:
        return None
    matched_subjects, filter_score = matching_philosophy_subjects(subjects, title=title)
    if not matched_subjects or filter_score < 2:
        return None

    description = extract_description(payload.get("description"))
    author_keys = [
        str(author.get("author", {}).get("key", ""))
        for author in payload.get("authors", [])
        if isinstance(author, dict)
    ]
    cover_id = first_value(payload.get("covers", []))
    work_key = str(payload.get("key", ""))

    return {
        "book_id": work_key,
        "title": title,
        "author_keys": [key for key in author_keys if key],
        "authors": [],
        "description": description,
        "subjects": subjects[:12],
        "matched_subjects": matched_subjects[:12],
        "filter_score": filter_score,
        "first_publish_year": str(payload.get("first_publish_date", "")),
        "language": "",
        "ratings_count": 0,
        "page_count": None,
        "thumbnail_url": f"https://covers.openlibrary.org/b/id/{cover_id}-M.jpg" if cover_id else "",
        "preview_link": f"https://openlibrary.org{work_key}" if work_key else "",
        "source": "openlibrary",
    }


def matching_philosophy_subjects(subjects: list[str], *, title: str = "") -> tuple[list[str], int]:
    """Return philosophy-matching subjects plus an inclusion score.

    Strong philosophy terms add +2, weak terms add +1, and deny signals in
    subjects/title subtract 2. This keeps the catalogue closer to academic
    philosophy and filters out broad fiction/self-help leakage.
    """

    matches: list[str] = []
    score = 0
    normalized_title = normalize_subject(title)
    for subject in subjects:
        normalized = normalize_subject(subject)
        if any(term_matches_subject(term, normalized) for term in NORMALIZED_STRONG_TERMS):
            matches.append(subject)
            score += 2
            continue
        if any(term_matches_subject(term, normalized) for term in NORMALIZED_WEAK_TERMS):
            matches.append(subject)
            score += 1
        if any(term_matches_subject(term, normalized) for term in NORMALIZED_DENY_SUBJECT_TERMS):
            score -= 2

    if any(term_matches_subject(term, normalized_title) for term in NORMALIZED_STRONG_TERMS):
        score += 2
    elif any(term_matches_subject(term, normalized_title) for term in NORMALIZED_WEAK_TERMS):
        score += 1

    return dedupe_preserve_order(matches), score


def normalize_subject(subject: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", subject.casefold()))


def term_matches_subject(term: str, normalized_subject: str) -> bool:
    if not term:
        return False
    if term == normalized_subject:
        return True
    return f" {term} " in f" {normalized_subject} "


def initialize_term_indexes() -> None:
    global NORMALIZED_STRONG_TERMS
    global NORMALIZED_WEAK_TERMS
    global NORMALIZED_DENY_SUBJECT_TERMS
    global NORMALIZED_COMMENTARY_SUBJECT_TERMS

    NORMALIZED_STRONG_TERMS = tuple(normalize_subject(term) for term in sorted(STRONG_PHILOSOPHY_TERMS))
    NORMALIZED_WEAK_TERMS = tuple(normalize_subject(term) for term in sorted(WEAK_PHILOSOPHY_TERMS))
    NORMALIZED_DENY_SUBJECT_TERMS = tuple(
        normalize_subject(term) for term in sorted(DENY_SUBJECT_TERMS)
    )
    NORMALIZED_COMMENTARY_SUBJECT_TERMS = tuple(
        normalize_subject(term) for term in sorted(COMMENTARY_SUBJECT_TERMS)
    )

def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def primary_author_list(values: list[str]) -> list[str]:
    authors = [value.strip() for value in values if value and value.strip()]
    if not authors:
        return []
    return [authors[0]]


def normalized_title_key(title: str) -> str:
    normalized = unicodedata.normalize("NFKD", title)
    ascii_title = normalized.encode("ascii", "ignore").decode("ascii")
    return " ".join(re.findall(r"[a-z0-9]+", ascii_title.casefold()))


def is_commentary_or_reference_row(row: dict[str, Any]) -> bool:
    normalized_title = normalized_title_key(str(row.get("title", "")))
    if not normalized_title:
        return True
    if any(pattern in normalized_title for pattern in COMMENTARY_TITLE_PATTERNS):
        return True

    normalized_subjects = {
        normalize_subject(str(subject))
        for subject in row.get("subjects", [])
    }
    return any(term in normalized_subjects for term in NORMALIZED_COMMENTARY_SUBJECT_TERMS)


def clean_catalog_rows(rows: list[dict[str, Any]], *, limit: int | None = None) -> list[dict[str, Any]]:
    selected_by_title: dict[str, dict[str, Any]] = {}
    for row in rows:
        cleaned = dict(row)
        cleaned["authors"] = primary_author_list(list(cleaned.get("authors", [])))
        if is_commentary_or_reference_row(cleaned):
            continue

        title_key = normalized_title_key(str(cleaned.get("title", "")))
        if not title_key:
            continue

        incumbent = selected_by_title.get(title_key)
        if incumbent is None or (
            int(cleaned.get("ratings_count", 0)),
            int(cleaned.get("filter_score", 0)),
        ) > (
            int(incumbent.get("ratings_count", 0)),
            int(incumbent.get("filter_score", 0)),
        ):
            selected_by_title[title_key] = cleaned

    cleaned_rows = list(selected_by_title.values())
    cleaned_rows.sort(key=lambda item: (-int(item.get("ratings_count", 0)), normalized_title_key(item["title"])))
    if limit is not None:
        return cleaned_rows[:limit]
    return cleaned_rows


def extract_description(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return str(value.get("value", "")).strip()
    return ""


def first_value(values: Any) -> Any:
    if isinstance(values, list) and values:
        return values[0]
    return None


def load_checkpoint(
    checkpoint_path: Path,
    *,
    works_path: Path,
    limit: int,
) -> dict[str, Any]:
    if not checkpoint_path.exists():
        return {
            "processed": 0,
            "cursor": 0,
            "selected_by_title": {},
        }

    row = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if row.get("works_path") != str(works_path.resolve()) or int(row.get("limit", limit)) != limit:
        return {
            "processed": 0,
            "cursor": 0,
            "selected_by_title": {},
        }

    selected_by_title = {
        title_key: (int(item["rating_count"]), item["row"])
        for title_key, item in row.get("selected_by_title", {}).items()
    }
    return {
        "processed": int(row.get("processed", 0)),
        "cursor": int(row.get("cursor", 0)),
        "selected_by_title": selected_by_title,
    }


def save_checkpoint(
    checkpoint_path: Path,
    *,
    works_path: Path,
    processed: int,
    cursor: int,
    limit: int,
    selected_by_title: dict[str, tuple[int, dict[str, Any]]],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "works_path": str(works_path.resolve()),
        "processed": processed,
        "cursor": cursor,
        "limit": limit,
        "selected_by_title": {
            title_key: {
                "rating_count": rating_count,
                "row": row,
            }
            for title_key, (rating_count, row) in selected_by_title.items()
        },
    }
    checkpoint_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def clear_checkpoint(checkpoint_path: Path) -> None:
    if checkpoint_path.exists():
        checkpoint_path.unlink()


initialize_term_indexes()


if __name__ == "__main__":
    raise SystemExit(main())
