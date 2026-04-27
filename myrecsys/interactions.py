"""Local interaction logging for collaborative filtering."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INTERACTIONS_PATH = ROOT / "data" / "interactions.jsonl"
ALLOWED_EVENT_TYPES = {"select", "open", "save"}


@dataclass(slots=True)
class InteractionEvent:
    user_id: str
    book_id: str
    event_type: str
    title: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def record_interaction(
    user_id: str,
    book_id: str,
    event_type: str,
    *,
    title: str = "",
    path: Path = DEFAULT_INTERACTIONS_PATH,
) -> None:
    user_id = str(user_id).strip()
    book_id = str(book_id).strip()
    event_type = str(event_type).strip().lower()
    if not user_id or not book_id or event_type not in ALLOWED_EVENT_TYPES:
        return

    event = InteractionEvent(
        user_id=user_id,
        book_id=book_id,
        event_type=event_type,
        title=str(title).strip(),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")


def load_interactions(path: Path = DEFAULT_INTERACTIONS_PATH) -> list[InteractionEvent]:
    if not path.exists():
        return []

    events: list[InteractionEvent] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            events.append(
                InteractionEvent(
                    user_id=str(row.get("user_id", "")),
                    book_id=str(row.get("book_id", "")),
                    event_type=str(row.get("event_type", "")),
                    title=str(row.get("title", "")),
                    timestamp=str(row.get("timestamp", "")),
                )
            )
    return events
