#!/usr/bin/env python3
"""Build a phase-2 semantic index from the local catalogue."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from myrecsys.env import load_dotenv
from myrecsys.local_catalog import _book_from_row
from myrecsys.phase2 import (
    DEFAULT_SEMANTIC_INDEX_PATH,
    build_book_semantic_text,
    embed_texts,
    extract_book_profile,
)


def main() -> int:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="Extract structured book profiles and embeddings for phase-2 retrieval."
    )
    parser.add_argument("--catalog", type=Path, default=Path("data/books.jsonl"))
    parser.add_argument("--output", type=Path, default=DEFAULT_SEMANTIC_INDEX_PATH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--request-interval", type=float, default=0.0)
    args = parser.parse_args()

    rows = load_rows(args.catalog, limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing_ids = load_existing_book_ids(args.output)

    pending_rows = [row for row in rows if str(row.get("book_id", "")) not in existing_ids]
    if not pending_rows:
        print(
            f"Semantic index already contains all {len(rows)} requested records at {args.output}"
        )
        return 0

    print(
        f"Loaded {len(rows)} catalogue rows, {len(existing_ids)} already indexed, "
        f"{len(pending_rows)} remaining."
    )

    batch_size = max(1, args.batch_size)
    request_interval = max(0.0, args.request_interval)
    write_mode = "a" if args.output.exists() and args.output.stat().st_size > 0 else "w"

    processed = 0
    with args.output.open(write_mode, encoding="utf-8") as handle:
        for start in range(0, len(pending_rows), batch_size):
            batch_rows = pending_rows[start : start + batch_size]
            payloads: list[dict[str, object]] = []
            texts: list[str] = []

            for offset, row in enumerate(batch_rows, start=1):
                book = _book_from_row(row, source_query="")
                print(
                    f"[{processed + offset}/{len(pending_rows)}] profiling {book.title}",
                    flush=True,
                )
                profile = extract_book_profile(book)
                profile.semantic_text = build_book_semantic_text(book, profile)
                payloads.append(
                    {
                        "book": row,
                        "profile": profile.to_dict(),
                    }
                )
                texts.append(profile.semantic_text)
                if request_interval and offset < len(batch_rows):
                    time.sleep(request_interval)

            print(
                f"Embedding batch {start // batch_size + 1} with {len(texts)} books...",
                flush=True,
            )
            embeddings = embed_texts(texts, task_type="retrieval_document")
            for payload, embedding in zip(payloads, embeddings):
                payload["embedding"] = embedding
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.flush()
            processed += len(payloads)

    print(f"Wrote {processed} new semantic records to {args.output}")
    return 0


def load_rows(path: Path, *, limit: int | None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def load_existing_book_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            book = row.get("book", {})
            book_id = str(book.get("book_id", "")).strip()
            if book_id:
                seen.add(book_id)
    return seen


if __name__ == "__main__":
    raise SystemExit(main())
