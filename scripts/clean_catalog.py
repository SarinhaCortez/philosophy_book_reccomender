#!/usr/bin/env python3
"""Clean the local catalogue in place.

This script keeps one primary author, removes commentary/reference rows, and
deduplicates repeated titles by keeping the strongest retained record.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_openlibrary_catalog import clean_catalog_rows  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean and deduplicate data/books.jsonl.")
    parser.add_argument("--catalog", type=Path, default=Path("data/books.jsonl"))
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.catalog.open("r", encoding="utf-8")
        if line.strip()
    ]
    cleaned = clean_catalog_rows(rows)

    with args.catalog.open("w", encoding="utf-8") as handle:
        for row in cleaned:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Cleaned {len(rows)} rows down to {len(cleaned)} in {args.catalog}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
