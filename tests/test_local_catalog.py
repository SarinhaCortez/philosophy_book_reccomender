import json

from myrecsys.local_catalog import CatalogNotFoundError, iter_catalog_rows, load_catalog_books


def test_iter_catalog_rows_reads_jsonl(tmp_path):
    catalog = tmp_path / "books.jsonl"
    rows = [
        {"book_id": "1", "title": "Generic Philosophy", "categories": ["philosophy"], "language": "en"},
        {"book_id": "2", "title": "Stoicism and Grief", "subjects": ["stoicism", "ethics"], "language": "en"},
    ]
    catalog.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    loaded = list(iter_catalog_rows(catalog))

    assert [row["book_id"] for row in loaded] == ["1", "2"]


def test_load_catalog_books_normalizes_rows(tmp_path):
    catalog = tmp_path / "books.jsonl"
    row = {"book_id": "1", "title": "Love and Philosophy", "subjects": ["love"], "language": "en"}
    catalog.write_text(json.dumps(row), encoding="utf-8")

    books = load_catalog_books(catalog)

    assert books[0].book_id == "1"
    assert books[0].subjects == ["love"]


def test_iter_catalog_rows_raises_for_missing_catalog(tmp_path):
    missing = tmp_path / "missing.jsonl"

    try:
        list(iter_catalog_rows(missing))
    except CatalogNotFoundError as exc:
        assert "Local catalogue not found" in str(exc)
    else:
        raise AssertionError("Expected CatalogNotFoundError")
