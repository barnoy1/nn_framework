from __future__ import annotations

from pathlib import Path


def read_log_tail(path: Path, lines: int = 8) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return ""
    if not content:
        return ""
    return " | ".join(content[-lines:])
