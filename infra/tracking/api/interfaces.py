from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Protocol

import numpy as np


class ExperimentTracker(Protocol):
    def log_image(self, tag: str, image: np.ndarray, step: int) -> None: ...

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None: ...

    def log_text(self, tag: str, text: str, step: int) -> None: ...

    def log_artifact(
        self, file_path: Path, artifact_path: str = "artifacts"
    ) -> None: ...

    def log_execution_config(self, execution_config: Dict[str, Any]) -> None: ...

    def close(self) -> None: ...


class NullExperimentTracker:
    def log_image(self, tag: str, image: np.ndarray, step: int) -> None:
        return

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        return

    def log_text(self, tag: str, text: str, step: int) -> None:
        return

    def log_artifact(self, file_path: Path, artifact_path: str = "artifacts") -> None:
        return

    def log_execution_config(self, execution_config: Dict[str, Any]) -> None:
        return

    def close(self) -> None:
        return


@dataclass
class CompositeExperimentTracker:
    loggers: List[ExperimentTracker]

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

    def log_execution_config(self, execution_config: Dict[str, Any]) -> None:
        for logger in self.loggers:
            logger.log_execution_config(execution_config=execution_config)

    def close(self) -> None:
        for logger in self.loggers:
            logger.close()
