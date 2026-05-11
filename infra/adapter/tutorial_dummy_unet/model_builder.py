from __future__ import annotations

from pathlib import Path

from infra.engine.model.wrappers.common import StagedAdapterModelBuilder


class TutorialDummyUNetModelBuilder(StagedAdapterModelBuilder):
    def __init__(self, app_config, repo_root: Path) -> None:
        super().__init__(
            app_config=app_config,
            repo_root=repo_root,
            adapter_root=Path(__file__).resolve().parent,
        )

    @classmethod
    def manifest(cls):
        from .manifest import create_manifest

        return create_manifest(builder_factory=cls)
