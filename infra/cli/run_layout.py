from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml

from .constants import REPO_ROOT


def _resolve_base_output_dir(output_dir: str) -> Path:
    base_out = Path(output_dir).expanduser()
    if base_out.is_absolute():
        return base_out
    return (REPO_ROOT / base_out).resolve()


def _create_run_layout(run_root: Path, output_base: Path, run_id: str) -> None:
    for subdir in ("checkpoint", "best", "logs", "configs", "inference", "dataset"):
        (run_root / subdir).mkdir(parents=True, exist_ok=True)
    # Shared MLflow store + per-run TensorBoard logs live under the base, not the run.
    (output_base / "mlflow").mkdir(parents=True, exist_ok=True)
    (output_base / "tensorboard" / run_id).mkdir(parents=True, exist_ok=True)


def prepare_run_layout(args) -> Path:
    output_base = _resolve_base_output_dir(args.output_dir)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_root = output_base / "runs" / run_id

    try:
        _create_run_layout(run_root, output_base, run_id)
    except PermissionError as error:
        raise PermissionError(
            "Output directory is not writable: "
            f"'{run_root}'. Please grant write permission to this exact path "
            "or pass a writable --output-dir."
        ) from error

    payload = {
        "action": args.action,
        "run_root": str(run_root),
        "args": {
            k: v
            for k, v in vars(args).items()
            if not k.startswith("_") and k != "handler"
        },
    }
    with (run_root / "configs" / "execution.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False)

    return run_root


if __name__ == "__main__":  # pragma: no cover - self-check
    import tempfile
    from types import SimpleNamespace

    with tempfile.TemporaryDirectory() as tmp:
        base = SimpleNamespace(output_dir=tmp, action="train")
        r1 = prepare_run_layout(base)
        r2 = prepare_run_layout(base)
        assert r1 != r2, "consecutive runs must get distinct run roots"
        assert (r1 / "checkpoint").is_dir() and (r2 / "checkpoint").is_dir()
        assert r1.parent == r2.parent == Path(tmp) / "runs"
        shared_mlflow = Path(tmp) / "mlflow"
        assert shared_mlflow.is_dir(), "mlflow store must be shared at base"
        assert (Path(tmp) / "tensorboard" / r1.name).is_dir()
        assert (Path(tmp) / "tensorboard" / r2.name).is_dir()
    print("run_layout self-check OK")
