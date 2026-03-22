from __future__ import annotations

from pathlib import Path

import yaml

from .constants import REPO_ROOT


def _resolve_base_output_dir(output_dir: str) -> Path:
    base_out = Path(output_dir).expanduser()
    if base_out.is_absolute():
        return base_out
    return (REPO_ROOT / base_out).resolve()


def _create_run_layout(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for subdir in ("logs", "configs", "inference", "dataset"):
        (root / subdir).mkdir(parents=True, exist_ok=True)


def prepare_run_layout(args) -> Path:
    run_root = _resolve_base_output_dir(args.output_dir)

    try:
        _create_run_layout(run_root)
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
