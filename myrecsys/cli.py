"""Small command-line demo for the first recommender pipeline slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .env import load_dotenv
from .recommendation import recommend_books


def main() -> int:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    parser = argparse.ArgumentParser(description="Search the local philosophy catalogue.")
    parser.add_argument("prompt", help="A natural-language reading request.")
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--no-network", action="store_true", help="Only parse intent.")
    args = parser.parse_args()

    if args.no_network:
        from .intent import normalize_query

        query = normalize_query(args.prompt)
        payload = {
            "ok": True,
            "search_request": {"query": query},
            "retrieval_query": query,
            "recommendations": [],
        }
    else:
        payload = recommend_books(
            args.prompt,
            max_results=args.max_results,
        )

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
