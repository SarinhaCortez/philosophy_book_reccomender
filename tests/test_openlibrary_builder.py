from scripts.build_openlibrary_catalog import (
    clean_catalog_rows,
    find_single_dump,
    load_ratings_counts,
    matching_philosophy_subjects,
    normalize_work,
    primary_author_list,
)


def test_normalize_work_keeps_philosophy_subjects():
    row = normalize_work(
        {
            "key": "/works/OL1W",
            "title": "A Philosophy Book",
            "subjects": ["Philosophy", "Ethics"],
            "authors": [{"author": {"key": "/authors/OL1A"}}],
            "description": {"value": "A useful description."},
        },
    )

    assert row
    assert row["book_id"] == "/works/OL1W"
    assert row["author_keys"] == ["/authors/OL1A"]
    assert row["subjects"] == ["Philosophy", "Ethics"]
    assert row["matched_subjects"] == ["Philosophy", "Ethics"]
    assert row["filter_score"] >= 2
    assert row["ratings_count"] == 0


def test_normalize_work_skips_non_philosophy_subjects():
    row = normalize_work({"key": "/works/OL2W", "title": "Cooking", "subjects": ["Cooking"]})

    assert row is None


def test_subject_matching_does_not_use_substrings():
    subjects = ["Computer software", "Soporte logico", "Internet programming"]

    assert matching_philosophy_subjects(subjects) == ([], 0)


def test_subject_matching_does_not_treat_logic_as_philosophy_signal():
    subjects = ["Logic", "Symbolic logic", "Mathematics"]

    assert matching_philosophy_subjects(subjects) == ([], 0)


def test_weak_philosophy_terms_do_not_admit_self_help():
    row = normalize_work(
        {
            "key": "/works/OL3W",
            "title": "How to Win Friends and Influence People",
            "subjects": ["Success", "Conduct of life", "Philosophy"],
        }
    )

    assert row is None


def test_strong_philosophy_terms_outweigh_broad_subjects():
    row = normalize_work(
        {
            "key": "/works/OL4W",
            "title": "Introduction to Metaphysics",
            "subjects": ["Philosophy", "Metaphysics", "Education"],
        }
    )

    assert row is not None
    assert row["matched_subjects"] == ["Philosophy", "Metaphysics"]
    assert row["filter_score"] >= 2


def test_find_single_dump_uses_dumps_folder_patterns(tmp_path):
    dumps = tmp_path / "dumps"
    dumps.mkdir()
    expected = dumps / "ol_dump_works_latest.txt.gz"
    expected.write_text("", encoding="utf-8")

    assert find_single_dump(dumps, ("ol_dump_works*.txt.gz",), "works") == expected


def test_load_ratings_counts_counts_rows_per_work(tmp_path):
    ratings = tmp_path / "ol_dump_ratings_latest.txt"
    ratings.write_text(
        "/works/OL1W\t/books/OL1M\t5\t2024-01-01\n"
        "/works/OL1W\t/books/OL2M\t3\t2024-01-02\n"
        "/works/OL2W\t\t4\t2024-01-03\n",
        encoding="utf-8",
    )

    assert load_ratings_counts(ratings) == {"/works/OL1W": 2, "/works/OL2W": 1}


def test_primary_author_list_keeps_only_first_author():
    assert primary_author_list(["Plato", "Translator", "Editor"]) == ["Plato"]


def test_clean_catalog_rows_dedupes_titles_and_skips_commentary():
    rows = [
        {
            "title": "Poetics",
            "authors": ["Aristotle", "Translator"],
            "subjects": ["Aesthetics"],
            "ratings_count": 7,
            "filter_score": 6,
        },
        {
            "title": "Poetics",
            "authors": ["Aristotle"],
            "subjects": ["Aesthetics"],
            "ratings_count": 4,
            "filter_score": 6,
        },
        {
            "title": "History of Philosophy",
            "authors": ["A. C. Grayling"],
            "subjects": ["Philosophy", "History"],
            "ratings_count": 9,
            "filter_score": 4,
        },
    ]

    cleaned = clean_catalog_rows(rows)

    assert cleaned == [
        {
            "title": "Poetics",
            "authors": ["Aristotle"],
            "subjects": ["Aesthetics"],
            "ratings_count": 7,
            "filter_score": 6,
        }
    ]
