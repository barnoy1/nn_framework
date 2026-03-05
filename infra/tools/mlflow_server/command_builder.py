from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path


def resolve_backend(tracking_dir: Path, backend: str, sqlite_db_name: str) -> str:
    if backend != "auto":
        return backend
    sqlite_path = tracking_dir / sqlite_db_name
    return "sqlite" if sqlite_path.exists() else "file"


def build_mlflow_command(
    *,
    tracking_dir: Path,
    backend: str,
    sqlite_db_name: str,
    host: str,
    port: int,
) -> list[str]:
    artifact_root = (tracking_dir / "mlruns").resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    launcher_prefix: list[str] = []

    candidate = Path(sys.executable).resolve().parent / "mlflow"
    if candidate.exists() and candidate.is_file():
        launcher_prefix = [str(candidate)]

    if not launcher_prefix:
        resolved = shutil.which("mlflow") or ""
        if resolved:
            launcher_prefix = [resolved]

    if not launcher_prefix:
        mlflow_main_spec = importlib.util.find_spec("mlflow.__main__")
        if mlflow_main_spec is not None and mlflow_main_spec.origin:
            launcher_prefix = [sys.executable, str(Path(mlflow_main_spec.origin).resolve())]

    if not launcher_prefix:
        raise RuntimeError(
            "Could not find a runnable mlflow launcher (console script or mlflow.__main__.py)."
        )

    if backend == "sqlite":
        sqlite_path = (tracking_dir / sqlite_db_name).resolve()
        backend_store_uri = f"sqlite:///{sqlite_path}"
    else:
        backend_store_uri = tracking_dir.as_uri()

    return [
        *launcher_prefix,
        "ui",
        "--backend-store-uri",
        backend_store_uri,
        "--default-artifact-root",
        artifact_root.as_uri(),
        "--host",
        host,
        "--port",
        str(port),
    ]
