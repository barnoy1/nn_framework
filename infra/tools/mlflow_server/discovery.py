from __future__ import annotations

from pathlib import Path


def resolve_tracking_dir(input_path: str) -> Path:
    candidate = Path(input_path).expanduser().resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Input path does not exist: {candidate}")

    if candidate.is_file():
        candidate = candidate.parent

    def _is_mlflow_root(path: Path) -> bool:
        if not path.exists() or not path.is_dir():
            return False
        return (path / "mlruns").is_dir()

    if _is_mlflow_root(candidate):
        return candidate

    if candidate.name == "mlruns":
        parent = candidate.parent
        if (parent / "mlflow.db").exists() or parent.name == "mlflow":
            return parent.resolve()
        return candidate

    nested_mlruns = candidate / "mlruns"
    if nested_mlruns.exists() and nested_mlruns.is_dir():
        return candidate

    for parent in [candidate, *candidate.parents]:
        if _is_mlflow_root(parent):
            return parent.resolve()
        if parent.name == "mlruns":
            if (parent.parent / "mlflow.db").exists() or parent.parent.name == "mlflow":
                return parent.parent.resolve()
            return parent.resolve()

    return candidate


def discover_tracking_dirs(input_path: str) -> list[Path]:
    resolved = resolve_tracking_dir(input_path)

    def _is_single_tracking_dir(path: Path) -> bool:
        return (
            path.is_dir()
            and (path / "mlruns").is_dir()
            and (path / "mlflow.db").exists()
        )

    def _has_visualization_mlflow(path: Path) -> bool:
        return (path / "visualization" / "mlflow").is_dir()

    if _is_single_tracking_dir(resolved):
        return [resolved]

    tracking_dirs: list[Path] = []
    if resolved.is_dir():
        for child in sorted(resolved.iterdir()):
            if not child.is_dir():
                continue
            if _has_visualization_mlflow(child):
                tracking_dirs.append((child / "visualization" / "mlflow").resolve())

    if tracking_dirs:
        return tracking_dirs

    if _has_visualization_mlflow(resolved):
        return [(resolved / "visualization" / "mlflow").resolve()]

    raise FileNotFoundError(
        f"No run folders with 'visualization/mlflow' found under: {resolved}"
    )
