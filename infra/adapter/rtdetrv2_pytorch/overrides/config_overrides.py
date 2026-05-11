from __future__ import annotations

from ..runtime import ensure_repo_import_paths


class RTDETRv2ConfigOverride:
    def apply(self, *, builder, state) -> None:
        ensure_repo_import_paths(builder.repo_root)
