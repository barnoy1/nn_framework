from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any, Dict

import numpy as np
from PIL import Image
import yaml


def _flatten_mapping(payload: Dict[str, Any], prefix: str = "") -> Dict[str, str]:
    flattened: Dict[str, str] = {}
    for key, value in payload.items():
        current = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten_mapping(value, prefix=current))
        elif isinstance(value, (list, tuple)):
            flattened[current] = json.dumps(value, ensure_ascii=False)
        else:
            flattened[current] = str(value)
    return flattened


def _sanitize_param_value(value: str, max_len: int = 490) -> str:
    text = str(value)
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}..."


class TensorBoardVisualizationLogger:
    def __init__(self, log_dir: Path):
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir=str(log_dir))

    def log_image(self, tag: str, image: np.ndarray, step: int) -> None:
        chw = np.transpose(image, (2, 0, 1)) if image.ndim == 3 else image
        self._writer.add_image(tag=tag, img_tensor=chw, global_step=step)

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        for key, value in metrics.items():
            self._writer.add_scalar(tag=key, scalar_value=float(value), global_step=step)

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
        self._writer.add_text(tag=f"artifact/{artifact_path}", text_string=summary, global_step=0)

    def close(self) -> None:
        self._writer.close()


class MlflowVisualizationLogger:
    @staticmethod
    def _resolve_run_folder_name(tracking_dir: Path) -> str:
        tracking_parts = tracking_dir.name.split("__", 1)
        if len(tracking_parts) == 2 and tracking_parts[1].strip():
            return tracking_parts[1].strip()

        parent_parts = tracking_dir.parent.name.split("__", 1)
        if len(parent_parts) == 2 and parent_parts[1].strip():
            return parent_parts[1].strip()

        if tracking_dir.name == "mlflow" and tracking_dir.parent.name == "visualization":
            return tracking_dir.parent.parent.name
        return tracking_dir.parent.name

    @staticmethod
    def _compose_run_name(base_run_name: str, run_context_dir: Path) -> str:
        resolved_base = str(base_run_name or "run").strip() or "run"
        run_folder_name = MlflowVisualizationLogger._resolve_run_folder_name(run_context_dir)
        suffix = f"__{run_folder_name}"
        if resolved_base.endswith(suffix):
            return resolved_base
        return f"{resolved_base}{suffix}"

    def __init__(
        self,
        experiment_name: str,
        run_name: str,
        tracking_dir: Path,
        run_context_dir: Path | None = None,
        tracking_backend: str = "sqlite",
        sqlite_db_name: str = "mlflow.db",
        execution_config: Dict[str, Any] | None = None,
    ):
        import mlflow

        self._mlflow = mlflow
        self._tracking_dir = tracking_dir.resolve()
        self._tracking_dir.mkdir(parents=True, exist_ok=True)
        resolved_run_context = (run_context_dir or self._tracking_dir).resolve()
        resolved_run_name = self._compose_run_name(run_name, resolved_run_context)
        artifact_root = (self._tracking_dir / "mlruns").resolve()
        artifact_root.mkdir(parents=True, exist_ok=True)
        backend = str(tracking_backend).strip().lower()
        if backend == "sqlite":
            sqlite_path = (self._tracking_dir / str(sqlite_db_name)).resolve()
            tracking_uri = f"sqlite:///{sqlite_path}"
            self._mlflow.set_tracking_uri(tracking_uri)
            client = self._mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is None:
                client.create_experiment(experiment_name, artifact_location=artifact_root.as_uri())
            self._mlflow.set_experiment(experiment_name)
        else:
            tracking_uri = self._tracking_dir.as_uri()
            self._mlflow.set_tracking_uri(tracking_uri)
            client = self._mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is None:
                client.create_experiment(experiment_name, artifact_location=artifact_root.as_uri())
            self._mlflow.set_experiment(experiment_name)
        active_run = self._mlflow.active_run()
        if active_run is None:
            self._run = self._mlflow.start_run(run_name=resolved_run_name)
            self._owns_run = True
        else:
            self._run = active_run
            self._owns_run = False
            self._mlflow.set_tag("mlflow.runName", resolved_run_name)
        self._last_step = -1
        if execution_config:
            self.log_execution_config(execution_config)

    @property
    def run_id(self) -> str | None:
        if self._run is None:
            return None
        return getattr(getattr(self._run, "info", None), "run_id", None)

    @property
    def experiment_id(self) -> str | None:
        if self._run is None:
            return None
        return getattr(getattr(self._run, "info", None), "experiment_id", None)

    def _monotonic_step(self, step: int) -> int:
        resolved = int(step)
        if resolved <= self._last_step:
            resolved = self._last_step + 1
        self._last_step = resolved
        return resolved

    def log_image(self, tag: str, image: np.ndarray, step: int) -> None:
        resolved_step = self._monotonic_step(step)
        image_name = tag.replace("/", "_")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            temp_path = Path(tmp.name)
        try:
            Image.fromarray(image).save(temp_path)
            self._mlflow.log_artifact(str(temp_path), artifact_path="images")
        finally:
            temp_path.unlink(missing_ok=True)

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        resolved_step = self._monotonic_step(step)
        for key, value in metrics.items():
            self._mlflow.log_metric(key, float(value), step=resolved_step)

    def log_text(self, tag: str, text: str, step: int) -> None:
        safe_tag = str(tag).replace("/", "_")
        artifact_file = f"text/{safe_tag}_{int(step):06d}.txt"
        self._mlflow.log_text(str(text), artifact_file=artifact_file)

    def log_artifact(self, file_path: Path, artifact_path: str = "artifacts") -> None:
        resolved = Path(file_path).resolve()
        if resolved.exists():
            self._mlflow.log_artifact(str(resolved), artifact_path=artifact_path)

    def log_execution_config(self, execution_config: Dict[str, Any]) -> None:
        if not execution_config:
            return
        flattened = _flatten_mapping(execution_config)
        if flattened:
            batch_size = 100
            items = list(flattened.items())
            for index in range(0, len(items), batch_size):
                chunk = {
                    key: _sanitize_param_value(value)
                    for key, value in items[index : index + batch_size]
                }
                self._mlflow.log_params(chunk)

        config_yaml = yaml.safe_dump(execution_config, sort_keys=True, allow_unicode=True)
        self._mlflow.log_text(config_yaml, artifact_file="config/config.yaml")

    def close(self) -> None:
        if self._run is not None and self._owns_run:
            self._mlflow.end_run()
