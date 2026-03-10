from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import mlflow
import yaml


def artifact_root(tracking_dir: Path) -> Path:
    root = (tracking_dir / "mlruns").resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def flatten_payload(payload: Dict, prefix: str = "") -> Dict[str, str]:
    flattened: Dict[str, str] = {}
    for key, value in payload.items():
        current = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_payload(value, prefix=current))
        elif isinstance(value, (list, tuple)):
            flattened[current] = json.dumps(value, ensure_ascii=False)
        else:
            flattened[current] = str(value)
    return flattened


def trim_param(value: str, max_len: int = 490) -> str:
    text = str(value)
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}..."


def normalize_registered_model_name(value: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-/"
    cleaned = "".join(ch if ch in allowed else "_" for ch in value)
    cleaned = cleaned.strip("._-/")
    return cleaned or "model"


def resolve_run_folder_name(run_output_dir: Path, tracking_dir: Path) -> str:
    resolved_output_dir = run_output_dir.resolve()
    output_name = resolved_output_dir.name.strip()
    if output_name:
        return output_name

    parent_parts = resolved_output_dir.parent.name.split("__", 1)
    if len(parent_parts) == 2 and parent_parts[1].strip():
        return parent_parts[1].strip()

    if tracking_dir.name == "mlflow" and tracking_dir.parent.name == "visualization":
        return tracking_dir.parent.parent.name
    return tracking_dir.parent.name


def extract_run_metadata_from_experiment_yaml(
    experiment_yaml_path: Path,
) -> Dict[str, str]:
    resolved_path = Path(experiment_yaml_path).resolve()
    if not resolved_path.exists() or not resolved_path.is_file():
        return {}

    payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        return {}

    runtime_cfg = payload.get("runtime") if isinstance(payload.get("runtime"), dict) else {}
    model_cfg = payload.get("model") if isinstance(payload.get("model"), dict) else {}

    description = str(runtime_cfg.get("description") or "").strip()
    source_root = str(model_cfg.get("source_root") or "").strip().rstrip("/")
    model_name = Path(source_root).name if source_root else ""

    metadata: Dict[str, str] = {}
    if description:
        metadata["description"] = description
    if source_root:
        metadata["model.source_root"] = source_root
    if model_name:
        metadata["model.name"] = model_name
    return metadata


def register_current_model(
    *,
    trainer,
    client: "mlflow.tracking.MlflowClient",
    experiment_name: str,
    execution_config: Dict,
) -> None:
    if not trainer.accelerator.is_main_process:
        return

    model_cfg = (
        execution_config.get("model") if isinstance(execution_config, dict) else None
    )
    if not isinstance(model_cfg, dict):
        return

    source_root = str(model_cfg.get("source_root") or "").strip()
    model_config_path = str(model_cfg.get("model_config_path") or "").strip()
    if not source_root and not model_config_path:
        return

    model_info = {
        "source_root": source_root,
        "model_config_path": model_config_path,
    }
    mlflow.log_text(
        yaml.safe_dump(model_info, sort_keys=True, allow_unicode=True),
        artifact_file="model/definition.yaml",
    )
    mlflow.set_tags(
        {
            "model.source_root": source_root,
            "model.model_config_path": model_config_path,
        }
    )

    active_run = mlflow.active_run()
    if active_run is None:
        return

    model_cfg_name = Path(model_config_path).stem if model_config_path else "model"
    registered_model_name = normalize_registered_model_name(
        f"{experiment_name}__{model_cfg_name}"
    )
    model_uri = f"runs:/{active_run.info.run_id}/model"

    class _IdentityModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input, params=None):
            return model_input

    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=_IdentityModel(),
        metadata=model_info,
        input_example={
            "source_root": source_root,
            "model_config_path": model_config_path,
        },
    )

    try:
        client.create_registered_model(registered_model_name)
    except Exception:
        pass

    try:
        client.create_model_version(
            name=registered_model_name,
            source=model_uri,
            run_id=active_run.info.run_id,
        )
        trainer.logger.info(
            "Registered MLflow model={} from run={}",
            registered_model_name,
            active_run.info.run_id,
        )
    except Exception as error:
        trainer.logger.warning("MLflow model registration skipped: {}", error)
