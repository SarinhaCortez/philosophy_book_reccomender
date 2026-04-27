#!/usr/bin/env python3
"""Build the semantic index through Gemini Batch API jobs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from myrecsys.env import load_dotenv
from myrecsys.local_catalog import _book_from_row
from myrecsys.phase2 import (
    DEFAULT_PHASE2_CHAT_MODEL,
    DEFAULT_PHASE2_EMBEDDING_MODEL,
    DEFAULT_SEMANTIC_INDEX_PATH,
    BookProfilePayload,
    book_profile_from_payload,
    book_profile_response_json_schema,
    build_book_profile_prompt,
    build_book_semantic_text,
)


DEFAULT_BATCH_DIR = ROOT / "data" / "batch"
DEFAULT_PROFILE_REQUESTS_PATH = DEFAULT_BATCH_DIR / "profile_requests.jsonl"
DEFAULT_PROFILE_MANIFEST_PATH = DEFAULT_BATCH_DIR / "profile_job.json"
DEFAULT_PROFILE_RESPONSES_PATH = DEFAULT_BATCH_DIR / "profile_responses.jsonl"
DEFAULT_PROFILE_OUTPUT_PATH = ROOT / "data" / "semantic_profiles.jsonl"
DEFAULT_EMBEDDING_REQUESTS_PATH = DEFAULT_BATCH_DIR / "embedding_requests.jsonl"
DEFAULT_EMBEDDING_MANIFEST_PATH = DEFAULT_BATCH_DIR / "embedding_job.json"
DEFAULT_EMBEDDING_RESPONSES_PATH = DEFAULT_BATCH_DIR / "embedding_responses.jsonl"


def main() -> int:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Batch-build the semantic index with Gemini.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_profiles = subparsers.add_parser("prepare-profiles")
    prepare_profiles.add_argument("--catalog", type=Path, default=Path("data/books.jsonl"))
    prepare_profiles.add_argument("--output", type=Path, default=DEFAULT_PROFILE_REQUESTS_PATH)
    prepare_profiles.add_argument("--limit", type=int, default=None)
    prepare_profiles.add_argument("--model", default=DEFAULT_PHASE2_CHAT_MODEL)

    submit_profiles = subparsers.add_parser("submit-profiles")
    submit_profiles.add_argument("--requests", type=Path, default=DEFAULT_PROFILE_REQUESTS_PATH)
    submit_profiles.add_argument("--manifest", type=Path, default=DEFAULT_PROFILE_MANIFEST_PATH)
    submit_profiles.add_argument("--model", default=DEFAULT_PHASE2_CHAT_MODEL)

    fetch_profiles = subparsers.add_parser("fetch-profiles")
    fetch_profiles.add_argument("--catalog", type=Path, default=Path("data/books.jsonl"))
    fetch_profiles.add_argument("--manifest", type=Path, default=DEFAULT_PROFILE_MANIFEST_PATH)
    fetch_profiles.add_argument("--responses", type=Path, default=DEFAULT_PROFILE_RESPONSES_PATH)
    fetch_profiles.add_argument("--profiles-output", type=Path, default=DEFAULT_PROFILE_OUTPUT_PATH)
    fetch_profiles.add_argument("--limit", type=int, default=None)

    prepare_embeddings = subparsers.add_parser("prepare-embeddings")
    prepare_embeddings.add_argument("--profiles", type=Path, default=DEFAULT_PROFILE_OUTPUT_PATH)
    prepare_embeddings.add_argument("--output", type=Path, default=DEFAULT_EMBEDDING_REQUESTS_PATH)
    prepare_embeddings.add_argument("--model", default=DEFAULT_PHASE2_EMBEDDING_MODEL)

    submit_embeddings = subparsers.add_parser("submit-embeddings")
    submit_embeddings.add_argument("--requests", type=Path, default=DEFAULT_EMBEDDING_REQUESTS_PATH)
    submit_embeddings.add_argument("--manifest", type=Path, default=DEFAULT_EMBEDDING_MANIFEST_PATH)
    submit_embeddings.add_argument("--model", default=DEFAULT_PHASE2_EMBEDDING_MODEL)

    fetch_embeddings = subparsers.add_parser("fetch-embeddings")
    fetch_embeddings.add_argument("--profiles", type=Path, default=DEFAULT_PROFILE_OUTPUT_PATH)
    fetch_embeddings.add_argument("--manifest", type=Path, default=DEFAULT_EMBEDDING_MANIFEST_PATH)
    fetch_embeddings.add_argument("--responses", type=Path, default=DEFAULT_EMBEDDING_RESPONSES_PATH)
    fetch_embeddings.add_argument("--output", type=Path, default=DEFAULT_SEMANTIC_INDEX_PATH)

    status = subparsers.add_parser("status")
    status.add_argument("--manifest", type=Path, required=True)

    args = parser.parse_args()

    if args.command == "prepare-profiles":
        return prepare_profile_requests(args.catalog, args.output, limit=args.limit, model_name=args.model)
    if args.command == "submit-profiles":
        return submit_generate_batch(args.requests, args.manifest, model_name=args.model)
    if args.command == "fetch-profiles":
        return fetch_profile_batch(
            args.catalog,
            args.manifest,
            args.responses,
            args.profiles_output,
            limit=args.limit,
        )
    if args.command == "prepare-embeddings":
        return prepare_embedding_requests(args.profiles, args.output, model_name=args.model)
    if args.command == "submit-embeddings":
        return submit_embedding_batch(args.requests, args.manifest, model_name=args.model)
    if args.command == "fetch-embeddings":
        return fetch_embedding_batch(args.profiles, args.manifest, args.responses, args.output)
    if args.command == "status":
        return print_batch_status(args.manifest)
    raise SystemExit(f"Unknown command: {args.command}")


def prepare_profile_requests(
    catalog_path: Path,
    output_path: Path,
    *,
    limit: int | None,
    model_name: str,
) -> int:
    rows = load_rows(catalog_path, limit=limit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema = book_profile_response_json_schema()

    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            book = _book_from_row(row, source_query="")
            request = {
                "key": book.book_id,
                "request": {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": build_book_profile_prompt(book)}],
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0,
                        "responseMimeType": "application/json",
                        "responseJsonSchema": schema,
                    },
                },
            }
            handle.write(json.dumps(request, ensure_ascii=False) + "\n")

    print(
        f"Wrote {len(rows)} profile batch requests for {model_name} to {output_path}"
    )
    return 0


def submit_generate_batch(
    requests_path: Path,
    manifest_path: Path,
    *,
    model_name: str,
) -> int:
    client, types = build_genai_client()
    uploaded = client.files.upload(
        file=str(requests_path),
        config=types.UploadFileConfig(
            display_name=requests_path.stem,
            mime_type="jsonl",
        ),
    )
    job = client.batches.create(
        model=model_name,
        src=uploaded.name,
        config={"display_name": requests_path.stem},
    )
    write_manifest(
        manifest_path,
        {
            "kind": "generate_content",
            "model": model_name,
            "requests_path": str(requests_path),
            "uploaded_file_name": uploaded.name,
            "batch_name": job.name,
        },
    )
    print(f"Created profile batch job {job.name} from {uploaded.name}")
    return 0


def fetch_profile_batch(
    catalog_path: Path,
    manifest_path: Path,
    responses_path: Path,
    profiles_output_path: Path,
    *,
    limit: int | None,
) -> int:
    client, _ = build_genai_client()
    manifest = read_manifest(manifest_path)
    job = client.batches.get(name=manifest["batch_name"])
    if not getattr(job, "done", False):
        print(f"Batch {job.name} is still {job.state.name}.")
        return 2
    if job.state.name != "JOB_STATE_SUCCEEDED":
        raise SystemExit(f"Batch {job.name} ended in {job.state.name}: {job.error}")
    if not job.dest or not job.dest.file_name:
        raise SystemExit(f"Batch {job.name} completed without a downloadable response file.")

    responses_path.parent.mkdir(parents=True, exist_ok=True)
    responses_path.write_bytes(client.files.download(file=job.dest.file_name))
    update_manifest(
        manifest_path,
        response_file_name=job.dest.file_name,
        responses_path=str(responses_path),
        state=job.state.name,
    )

    rows = load_rows(catalog_path, limit=limit)
    rows_by_id = {str(row.get("book_id", "")): row for row in rows}
    request_entries = read_jsonl(Path(manifest["requests_path"]))
    response_entries = read_jsonl(responses_path)

    written = 0
    profiles_output_path.parent.mkdir(parents=True, exist_ok=True)
    with profiles_output_path.open("w", encoding="utf-8") as handle:
        for request_entry, response_entry in zip(request_entries, response_entries):
            book_id = str(request_entry.get("key", ""))
            row = rows_by_id.get(book_id)
            if not row:
                continue
            if response_has_error(response_entry):
                continue
            payload = extract_profile_payload(response_entry)
            book = _book_from_row(row, source_query="")
            profile = book_profile_from_payload(book, payload)
            profile.semantic_text = build_book_semantic_text(book, profile)
            handle.write(
                json.dumps(
                    {
                        "book": row,
                        "profile": profile.to_dict(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1

    print(f"Downloaded {job.dest.file_name} and materialized {written} semantic profiles.")
    return 0


def prepare_embedding_requests(
    profiles_path: Path,
    output_path: Path,
    *,
    model_name: str,
) -> int:
    entries = read_jsonl(profiles_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            book = entry["book"]
            profile = entry["profile"]
            semantic_text = profile.get("semantic_text", "")
            request = {
                "key": book["book_id"],
                "request": {
                    "model": f"models/{model_name}",
                    "content": {
                        "role": "user",
                        "parts": [{"text": semantic_text}],
                    },
                    "taskType": "RETRIEVAL_DOCUMENT",
                    "title": book["title"],
                },
            }
            handle.write(json.dumps(request, ensure_ascii=False) + "\n")

    print(
        f"Wrote {len(entries)} embedding batch requests for {model_name} to {output_path}"
    )
    return 0


def submit_embedding_batch(
    requests_path: Path,
    manifest_path: Path,
    *,
    model_name: str,
) -> int:
    client, types = build_genai_client()
    uploaded = client.files.upload(
        file=str(requests_path),
        config=types.UploadFileConfig(
            display_name=requests_path.stem,
            mime_type="jsonl",
        ),
    )
    job = client.batches.create_embeddings(
        model=model_name,
        src={"file_name": uploaded.name},
        config={"display_name": requests_path.stem},
    )
    write_manifest(
        manifest_path,
        {
            "kind": "embeddings",
            "model": model_name,
            "requests_path": str(requests_path),
            "uploaded_file_name": uploaded.name,
            "batch_name": job.name,
        },
    )
    print(f"Created embedding batch job {job.name} from {uploaded.name}")
    return 0


def fetch_embedding_batch(
    profiles_path: Path,
    manifest_path: Path,
    responses_path: Path,
    output_path: Path,
) -> int:
    client, _ = build_genai_client()
    manifest = read_manifest(manifest_path)
    job = client.batches.get(name=manifest["batch_name"])
    if not getattr(job, "done", False):
        print(f"Batch {job.name} is still {job.state.name}.")
        return 2
    if job.state.name != "JOB_STATE_SUCCEEDED":
        raise SystemExit(f"Batch {job.name} ended in {job.state.name}: {job.error}")
    if not job.dest or not job.dest.file_name:
        raise SystemExit(f"Batch {job.name} completed without a downloadable response file.")

    responses_path.parent.mkdir(parents=True, exist_ok=True)
    responses_path.write_bytes(client.files.download(file=job.dest.file_name))
    update_manifest(
        manifest_path,
        response_file_name=job.dest.file_name,
        responses_path=str(responses_path),
        state=job.state.name,
    )

    profile_entries = read_jsonl(profiles_path)
    response_entries = read_jsonl(responses_path)
    request_entries = read_jsonl(Path(manifest["requests_path"]))
    profile_by_id = {entry["book"]["book_id"]: entry for entry in profile_entries}

    written = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for request_entry, response_entry in zip(request_entries, response_entries):
            book_id = str(request_entry.get("key", ""))
            profile_entry = profile_by_id.get(book_id)
            if not profile_entry or response_has_error(response_entry):
                continue
            embedding = extract_embedding_values(response_entry)
            handle.write(
                json.dumps(
                    {
                        "book": profile_entry["book"],
                        "profile": profile_entry["profile"],
                        "embedding": embedding,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1

    print(f"Downloaded {job.dest.file_name} and materialized {written} semantic index rows.")
    return 0


def print_batch_status(manifest_path: Path) -> int:
    client, _ = build_genai_client()
    manifest = read_manifest(manifest_path)
    job = client.batches.get(name=manifest["batch_name"])
    print(f"Batch: {job.name}")
    print(f"State: {job.state.name if job.state else 'UNKNOWN'}")
    if job.dest and job.dest.file_name:
        print(f"Result file: {job.dest.file_name}")
    if job.error:
        print(f"Error: {job.error}")
    return 0


def build_genai_client() -> tuple[Any, Any]:
    api_key = load_google_api_key()
    from google import genai
    from google.genai import types

    return genai.Client(api_key=api_key), types


def load_google_api_key() -> str:
    api_key = str(os.environ.get("GOOGLE_API_KEY", "")).strip()
    if not api_key or api_key == "your_api_key_here":
        raise SystemExit("Set GOOGLE_API_KEY in .env before using batch mode.")
    return api_key


def load_rows(path: Path, *, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def update_manifest(path: Path, **updates: Any) -> None:
    payload = read_manifest(path)
    payload.update(updates)
    write_manifest(path, payload)


def response_has_error(response_entry: dict[str, Any]) -> bool:
    return bool(response_entry.get("error") or response_entry.get("response", {}).get("error"))


def extract_profile_payload(response_entry: dict[str, Any]) -> dict[str, Any]:
    response = response_entry.get("response", response_entry)
    text = extract_generate_content_text(response)
    payload = json.loads(text)
    return BookProfilePayload.model_validate(payload).model_dump()


def extract_generate_content_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("text"), str):
        return response["text"]
    for candidate in response.get("candidates", []):
        content = candidate.get("content", {})
        texts = [
            str(part.get("text", ""))
            for part in content.get("parts", [])
            if isinstance(part, dict) and part.get("text")
        ]
        if texts:
            return "".join(texts)
    raise ValueError(f"Could not extract text from batch response: {response}")


def extract_embedding_values(response_entry: dict[str, Any]) -> list[float]:
    response = response_entry.get("response", response_entry)
    embedding = response.get("embedding", {})
    values = embedding.get("values")
    if not isinstance(values, list):
        raise ValueError(f"Could not extract embedding values from batch response: {response}")
    return [float(value) for value in values]


if __name__ == "__main__":
    raise SystemExit(main())
