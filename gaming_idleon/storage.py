from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SESSION_FILE = Path(__file__).resolve().parents[1] / "session.json"


def load_session() -> dict[str, Any]:
    if not SESSION_FILE.exists():
        return {}
    try:
        return json.loads(SESSION_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_session(data: dict[str, Any]) -> None:
    SESSION_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def clear_session() -> None:
    if SESSION_FILE.exists():
        SESSION_FILE.unlink()
