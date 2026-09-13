"""Redacted JSONL observability for Claude Agent SDK runs."""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SECRET_PATTERNS = (
    re.compile(r"sk-ant-[A-Za-z0-9_-]+"),
    re.compile(r"(?i)(api[_ -]?key|authorization|token)\s*[:=]\s*\S+"),
)
PII_PATTERNS = (
    re.compile(r"\b\d{10,16}\b"),
    re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
)
BASE64_PATTERN = re.compile(r"[A-Za-z0-9+/]{256,}={0,2}")


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        if "tool_use_id" in value and "content" in value:
            content = str(value.get("content"))
            value = {
                **value,
                "content": {
                    "redacted": True,
                    "sha256": hashlib.sha256(content.encode()).hexdigest(),
                    "length": len(content),
                },
            }
        result = {}
        for key, item in value.items():
            if key.lower() in {"data", "api_key", "authorization", "message_text"}:
                text = str(item)
                result[key] = {
                    "redacted": True,
                    "sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "length": len(text),
                }
            else:
                result[key] = redact(item)
        return result
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        text = value
        for pattern in (*SECRET_PATTERNS, *PII_PATTERNS):
            text = pattern.sub("[REDACTED]", text)
        text = BASE64_PATTERN.sub("[BASE64_REDACTED]", text)
        return text[:4000]
    return value


def serializable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "__dict__"):
        return {
            key: serializable(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return value


class TraceWriter:
    def __init__(self, path: Path, run_id: str, request_id: str) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.request_id = request_id
        self.sequence = 0
        self._lock = threading.Lock()

    def write(self, event_type: str, payload: Any = None) -> None:
        with self._lock:
            self.sequence += 1
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "run_id": self.run_id,
                "request_id": self.request_id,
                "sequence": self.sequence,
                "event_type": event_type,
                "payload": redact(serializable(payload)),
            }
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
