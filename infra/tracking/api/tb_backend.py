from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image


class TensorBoardVisualizationLogger:
    def __init__(self, log_dir: Path):
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir=str(log_dir))

    def log_image(self, tag: str, image: np.ndarray, step: int) -> None:
        chw = np.transpose(image, (2, 0, 1)) if image.ndim == 3 else image
        self._writer.add_image(tag=tag, img_tensor=chw, global_step=step)

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        for key, value in metrics.items():
            self._writer.add_scalar(
                tag=key, scalar_value=float(value), global_step=step
            )

    def log_text(self, tag: str, text: str, step: int) -> None:
        self._writer.add_text(tag=tag, text_string=str(text), global_step=step)

    def log_artifact(self, file_path: Path, artifact_path: str = "artifacts") -> None:
        resolved = Path(file_path).resolve()
        tag_base = f"artifact/{artifact_path}/{resolved.stem}"
        if not resolved.exists():
            self._writer.add_text(
                tag=f"artifact/{artifact_path}",
                text_string=f"missing artifact file={resolved}",
                global_step=0,
            )
            return

        suffix = resolved.suffix.lower()
        try:
            if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
                image = np.array(Image.open(resolved).convert("RGB"))
                chw = np.transpose(image, (2, 0, 1))
                self._writer.add_image(tag=tag_base, img_tensor=chw, global_step=0)
                return

            if suffix in {".json", ".txt", ".yaml", ".yml", ".csv"}:
                text = resolved.read_text(encoding="utf-8", errors="ignore")
                if len(text) > 20000:
                    text = f"{text[:20000]}\n... (truncated)"
                self._writer.add_text(tag=tag_base, text_string=text, global_step=0)
                return
        except Exception as error:
            self._writer.add_text(
                tag=f"{tag_base}/error",
                text_string=f"failed to read artifact {resolved}: {error}",
                global_step=0,
            )
            return

        summary = f"artifact_path={artifact_path} file={resolved}"
        self._writer.add_text(
            tag=f"artifact/{artifact_path}", text_string=summary, global_step=0
        )

    def close(self) -> None:
        self._writer.close()
