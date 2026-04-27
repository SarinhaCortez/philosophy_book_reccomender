#!/usr/bin/env python3
"""Append zero-rated works by selected philosophers to the existing catalogue."""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Any

from build_openlibrary_catalog import (
    AUTHORS_PATTERNS,
    DEFAULT_DUMPS_DIR,
    WORKS_PATTERNS,
    extract_description,
    find_single_dump,
    first_value,
    iter_dump_payloads,
    load_author_names,
)


PHILOSOPHER_NAMES = [
    "Pythagoras",
    "Confucius",
    "Heracleitus",
    "Parmenides",
    "Zeno of Elea",
    "Socrates",
    "Democritus",
    "Plato",
    "Aristotle",
    "Mencius",
    "Zhuangzi",
    "Pyrrhon of Elis",
    "Epicurus",
    "Zeno of Citium",
    "Philo Judaeus",
    "Marcus Aurelius",
    "Nagarjuna",
    "Plotinus",
    "Sextus Empiricus",
    "Saint Augustine",
    "Hypatia",
    "Anicius Manlius Severinus Boethius",
    "Śaṅkara",
    "Yaqūb ibn Ishāq aṣ-Ṣabāḥ al-Kindī",
    "Al-Fārābī",
    "Avicenna",
    "Rāmānuja",
    "Ibn Gabirol",
    "Saint Anselm of Canterbury",
    "al-Ghazālī",
    "Peter Abelard",
    "Averroës",
    "Zhu Xi",
    "Moses Maimonides",
    "Ibn al-'Arabī",
    "Shinran",
    "Saint Thomas Aquinas",
    "John Duns Scotus",
    "William of Ockham",
    "Niccolò Machiavelli",
    "Wang Yangming",
    "Francis Bacon, Viscount Saint Alban (or Albans), Baron of Verulam",
    "Thomas Hobbes",
    "René Descartes",
    "John Locke",
    "Benedict de Spinoza",
    "Gottfried Wilhelm Leibniz",
    "Giambattista Vico",
    "George Berkeley",
    "Charles-Louis de Secondat, baron de La Brède et de Montesquieu",
    "David Hume",
    "Jean-Jacques Rousseau",
    "Immanuel Kant",
    "Moses Mendelssohn",
    "Marie-Jean-Antoine-Nicolas de Caritat, marquis de Condorcet",
    "Jeremy Bentham",
    "Georg Wilhelm Friedrich Hegel",
    "Arthur Schopenhauer",
    "Auguste Comte",
    "Virginia Woolf"
    "John Stuart Mill",
    "Søren Kierkegaard",
    "Karl Marx",
    "Herbert Spencer",
    "Wilhelm Dilthey",
    "William James",
    "Friedrich Nietzsche",
    "Friedrich Ludwig Gottlob Frege",
    "Edmund Husserl",
    "Henri Bergson",
    "John Dewey",
    "Alfred North Whitehead",
    "Benedetto Croce",
    "Nishida Kitarō",
    "Bertrand Russell",
    "G.E. Moore",
    "Martin Buber",
    "Ludwig Wittgenstein",
    "Martin Heidegger",
    "Rudolf Carnap",
    "Sir Karl Popper",
    "Theodor Wiesengrund Adorno",
    "Jean-Paul Sartre",
    "Hannah Arendt",
    "Simone de Beauvoir",
    "Willard Van Orman Quine",
    "Sir A.J. Ayer",
    "Wilfrid Sellars",
    "John Rawls",
    "Thomas S. Kuhn",
    "Michel Foucault",
    "Noam Chomsky",
    "Jürgeb Gabernas",
    "Sir Bernard Williams",
    "Jacques Derrida",
    "Richard Rorty",
    "Robert Nozick",
    "Saul Kripke",
    "David Kellogg Lewis",
    "Peter (Albert David) Singer",
    "Seneca", "Cicero",
]


def normalize_name(value: str) -> str:
    ascii_text = (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode("ascii")
        .casefold()
    )
    cleaned = []
    for char in ascii_text:
        if char.isalnum():
            cleaned.append(char)
        else:
            cleaned.append(" ")
    return " ".join("".join(cleaned).split())


AUTHOR_NAME_ALIASES = {
    normalize_name("Jürgeb Gabernas"): {normalize_name("Jürgen Habermas")},
    normalize_name("Peter (Albert David) Singer"): {normalize_name("Peter Singer")},
    normalize_name("Sir Karl Popper"): {normalize_name("Karl Popper")},
    normalize_name("Sir A.J. Ayer"): {normalize_name("A. J. Ayer"), normalize_name("AJ Ayer")},
    normalize_name("G.E. Moore"): {normalize_name("G. E. Moore"), normalize_name("George Edward Moore")},
    normalize_name("Averroës"): {normalize_name("Averroes"), normalize_name("Ibn Rushd")},
    normalize_name("Avicenna"): {normalize_name("Ibn Sina"), normalize_name("Avicenna")},
    normalize_name("Seneca"): {normalize_name("Lucius Annaeus Seneca"), normalize_name("Seneca")},
    normalize_name("Saint Augustine"): {normalize_name("Augustine of Hippo"), normalize_name("Saint Augustine")},
    normalize_name("Saint Thomas Aquinas"): {normalize_name("Thomas Aquinas"), normalize_name("Saint Thomas Aquinas")},
    normalize_name("Saint Anselm of Canterbury"): {normalize_name("Anselm of Canterbury")},
    normalize_name("Śaṅkara"): {normalize_name("Shankara"), normalize_name("Adi Shankara")},
    normalize_name("Al-Fārābī"): {normalize_name("Al-Farabi")},
    normalize_name("al-Ghazālī"): {normalize_name("Al-Ghazali")},
    normalize_name("René Descartes"): {normalize_name("Rene Descartes")},
    normalize_name("Søren Kierkegaard"): {normalize_name("Soren Kierkegaard")},
    normalize_name("Niccolò Machiavelli"): {normalize_name("Niccolo Machiavelli")},
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Append zero-rated works by selected philosophers to the existing catalogue."
    )
    parser.add_argument("--dumps-dir", type=Path, default=DEFAULT_DUMPS_DIR)
    parser.add_argument("--catalog", type=Path, default=Path("data/books.jsonl"))
    args = parser.parse_args()

    works_path = find_single_dump(args.dumps_dir, WORKS_PATTERNS, "works")
    authors_path = find_single_dump(args.dumps_dir, AUTHORS_PATTERNS, "authors")
    print(f"Using works dump: {works_path}", flush=True)
    print(f"Using authors dump: {authors_path}", flush=True)

    existing_rows = load_existing_rows(args.catalog)
    existing_ids = {row["book_id"] for row in existing_rows}
    print(f"Loaded existing catalogue: {args.catalog} ({len(existing_rows)} rows)", flush=True)

    print("Scanning authors dump for target philosophers...", flush=True)
    target_author_keys, matched_names = load_target_author_keys(authors_path)

    print(
        f"Loaded {len(existing_rows)} existing catalogue rows and matched "
        f"{len(target_author_keys)} philosopher author records.",
        flush=True,
    )

    new_rows: list[dict[str, Any]] = []
    seen_new_ids: set[str] = set()
    needed_author_keys: set[str] = set()
    print("Scanning works dump for zero-rated books by matched philosophers...", flush=True)
    for payload, _ in iter_dump_payloads(works_path):
        row = normalize_work_by_author(payload)
        if not row:
            continue
        if row["book_id"] in existing_ids or row["book_id"] in seen_new_ids:
            continue
        if row["ratings_count"] != 0:
            continue
        author_keys = set(row["author_keys"])
        if not author_keys.intersection(target_author_keys):
            continue
        new_rows.append(row)
        seen_new_ids.add(row["book_id"])
        needed_author_keys.update(row["author_keys"])

    author_names = load_author_names(authors_path, wanted_keys=needed_author_keys)
    for row in new_rows:
        row["authors"] = [
            author_names.get(key, key.removeprefix("/authors/"))
            for key in row.pop("author_keys", [])
        ]

    new_rows.sort(key=lambda row: (author_sort_key(row["authors"]), row["title"].casefold()))

    with args.catalog.open("a", encoding="utf-8") as handle:
        for row in new_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    unmatched = sorted(set(PHILOSOPHER_NAMES) - matched_names)
    print(f"Appended {len(new_rows)} zero-rated works to {args.catalog}")
    if unmatched:
        print(f"Unmatched philosopher names ({len(unmatched)}):")
        for name in unmatched:
            print(f"  - {name}")
    return 0


def load_existing_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_target_author_keys(path: Path) -> tuple[set[str], set[str]]:
    target_variants = build_target_variants()
    matched_keys: set[str] = set()
    matched_targets: set[str] = set()

    for payload, _ in iter_dump_payloads(path):
        key = str(payload.get("key", ""))
        name = str(payload.get("name") or payload.get("personal_name") or "").strip()
        if not key or not name:
            continue
        normalized = normalize_name(name)
        for target, variants in target_variants.items():
            if normalized in variants:
                matched_keys.add(key)
                matched_targets.add(target)
                break

    return matched_keys, matched_targets


def build_target_variants() -> dict[str, set[str]]:
    variants: dict[str, set[str]] = {}
    for name in PHILOSOPHER_NAMES:
        normalized = normalize_name(name)
        variants[name] = {normalized}
        alias_values = AUTHOR_NAME_ALIASES.get(normalized, set())
        variants[name].update(alias_values)
    return variants


def normalize_work_by_author(payload: dict[str, Any]) -> dict[str, Any] | None:
    title = str(payload.get("title", "")).strip()
    if not title:
        return None

    author_keys: list[str] = []
    for author in payload.get("authors", []):
        if not isinstance(author, dict):
            continue
        author_ref = author.get("author", {})
        if isinstance(author_ref, dict):
            key = str(author_ref.get("key", ""))
        else:
            key = str(author_ref)
        if key:
            author_keys.append(key)
    if not author_keys:
        return None

    subjects = [str(subject) for subject in payload.get("subjects", [])]
    description = extract_description(payload.get("description"))
    cover_id = first_value(payload.get("covers", []))
    work_key = str(payload.get("key", ""))

    return {
        "book_id": work_key,
        "title": title,
        "author_keys": [key for key in author_keys if key],
        "authors": [],
        "description": description,
        "subjects": subjects[:12],
        "matched_subjects": [],
        "filter_score": 0,
        "first_publish_year": str(payload.get("first_publish_date", "")),
        "language": "",
        "ratings_count": 0,
        "page_count": None,
        "thumbnail_url": f"https://covers.openlibrary.org/b/id/{cover_id}-M.jpg" if cover_id else "",
        "preview_link": f"https://openlibrary.org{work_key}" if work_key else "",
        "source": "openlibrary",
    }


def author_sort_key(authors: list[str]) -> str:
    if not authors:
        return ""
    return normalize_name(authors[0])


if __name__ == "__main__":
    raise SystemExit(main())
