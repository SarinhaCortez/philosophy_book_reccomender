"""Tiny local web server for the MVP UI."""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .env import load_dotenv
from .recommendation import recommend_books


ROOT = Path(__file__).resolve().parent.parent
STATIC_FILES = {
    "/": ROOT / "ui.html",
    "/ui.html": ROOT / "ui.html",
    "/favicon.svg": ROOT / "favicon.svg",
}


class RecommenderHandler(BaseHTTPRequestHandler):
    server_version = "MyRecSys/0.1"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        file_path = STATIC_FILES.get(path)
        if file_path and file_path.exists():
            self._send_bytes(file_path.read_bytes(), _content_type_for(file_path))
            return
        self._send_json({"ok": False, "error": "Not found"}, status=404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/recommend":
            self._send_json({"ok": False, "error": "Not found"}, status=404)
            return

        try:
            body = self._read_json()
            prompt = str(body.get("prompt", "")).strip()
            max_results = int(body.get("max_results", 12))
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json({"ok": False, "error": f"Invalid request: {exc}"}, status=400)
            return

        if not prompt:
            self._send_json({"ok": False, "error": "Prompt is required"}, status=400)
            return

        payload = recommend_books(
            prompt,
            max_results=max_results,
        )
        self._send_json(payload, status=200)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _send_json(self, payload: dict[str, Any], *, status: int = 200) -> None:
        self._send_bytes(
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
            status=status,
        )

    def _send_bytes(self, payload: bytes, content_type: str, *, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> int:
    load_dotenv(ROOT / ".env")
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), RecommenderHandler)
    print(f"Serving philosophy recommender at http://127.0.0.1:{port}")
    server.serve_forever()
    return 0


def _content_type_for(path: Path) -> str:
    if path.suffix == ".svg":
        return "image/svg+xml"
    if path.suffix in {".html", ".htm"}:
        return "text/html; charset=utf-8"
    return "application/octet-stream"


if __name__ == "__main__":
    raise SystemExit(main())
