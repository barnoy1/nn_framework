from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

from infra.common.logging import logger


class RuntimePathResolver:
    _ENV_TOKEN_PATTERN = re.compile(r"\$\{(?:oc\.env|env):([^}]+)\}")

    def __init__(
        self,
        *,
        repo_root: Path,
        extra_search_roots: Iterable[Path] = (),
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.extra_search_roots = tuple(Path(root).resolve() for root in extra_search_roots)

    @classmethod
    def expand_runtime_tokens(cls, path: str) -> str:
        raw = str(path or "").strip()
        if not raw:
            return raw

        def _replace(match: re.Match[str]) -> str:
            variable_name = match.group(1).strip()
            return os.environ.get(variable_name, match.group(0))

        expanded = cls._ENV_TOKEN_PATTERN.sub(_replace, raw)
        return os.path.expandvars(expanded)

    def checkpoint_search_roots(self) -> list[Path]:
        return [
            (self.repo_root / "weights").resolve(),
            (self.repo_root.parent / "weights").resolve(),
            *self.extra_search_roots,
        ]

    @staticmethod
    def _weights_suffix(value: Path) -> Path | None:
        parts = value.parts
        for index, part in enumerate(parts):
            if part == "weights" and index + 1 < len(parts):
                return Path(*parts[index + 1 :])
        return None

    def _checkpoint_candidates(self, candidate: Path) -> list[Path]:
        candidates: list[Path] = []
        if not candidate.is_absolute():
            candidates.append((self.repo_root / candidate).resolve())
            candidates.append((self.repo_root.parent / candidate).resolve())

        search_roots = self.checkpoint_search_roots()
        relative_to_weights = self._weights_suffix(candidate)
        if relative_to_weights is not None:
            candidates.extend((root / relative_to_weights).resolve() for root in search_roots)

        if not candidate.is_absolute() and candidate.parts and candidate.parts[0] == "weights":
            tail = Path(*candidate.parts[1:]) if len(candidate.parts) > 1 else Path(candidate.name)
            candidates.extend((root / tail).resolve() for root in search_roots)

        candidates.extend((root / candidate.name).resolve() for root in search_roots)
        unique: list[Path] = []
        seen: set[Path] = set()
        for item in candidates:
            if item in seen:
                continue
            seen.add(item)
            unique.append(item)
        return unique

    def resolve_checkpoint(self, path: str) -> Path:
        resolved_input = self.expand_runtime_tokens(path)
        candidate = Path(resolved_input).expanduser()
        if candidate.exists():
            return candidate.resolve()

        candidates = self._checkpoint_candidates(candidate)
        for fallback in candidates:
            if fallback.exists():
                logger.warning("checkpoint not found at {}, using {}", candidate, fallback)
                return fallback

        for root in self.checkpoint_search_roots():
            if not root.exists():
                continue
            matches = sorted(root.rglob(candidate.name))
            if matches:
                fallback = matches[0].resolve()
                logger.warning("checkpoint not found at {}, using {}", candidate, fallback)
                return fallback

        checked = "\n  - ".join(str(p) for p in [candidate, *candidates])
        raise FileNotFoundError(f"Checkpoint file not found. Checked:\n  - {checked}")
