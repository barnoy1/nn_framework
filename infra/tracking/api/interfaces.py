from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Protocol

import numpy as np


class VisualizationLogger(Protocol):
    def log_image(self, tag: str, image: np.ndarray, step: int) -> None: ...

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None: ...

    def log_text(self, tag: str, text: str, step: int) -> None: ...

    def log_artifact(
        self, file_path: Path, artifact_path: str = "artifacts"
    ) -> None: ...

    def close(self) -> None: ...


class NullVisualizationLogger:
    def log_image(self, tag: str, image: np.ndarray, step: int) -> None:
        return

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        return

    def log_text(self, tag: str, text: str, step: int) -> None:
        return

    def log_artifact(self, file_path: Path, artifact_path: str = "artifacts") -> None:
        return

    def close(self) -> None:
        return


@dataclass
class CompositeVisualizationLogger:
    loggers: List[VisualizationLogger]

    def log_image(self, tag: str, image: np.ndarray, step: int) -> None:
        for logger in self.loggers:
            logger.log_image(tag=tag, image=image, step=step)

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        for logger in self.loggers:
            logger.log_metrics(metrics=metrics, step=step)

    def log_text(self, tag: str, text: str, step: int) -> None:
        for logger in self.loggers:
            logger.log_text(tag=tag, text=text, step=step)

    def log_artifact(self, file_path: Path, artifact_path: str = "artifacts") -> None:
        for logger in self.loggers:
            logger.log_artifact(file_path=file_path, artifact_path=artifact_path)

    def close(self) -> None:
        for logger in self.loggers:
            logger.close()
