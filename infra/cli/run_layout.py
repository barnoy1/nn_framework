from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml

from .constants import REPO_ROOT


def prepare_run_layout(args) -> Path:
    base_out = Path(args.output_dir).expanduser()
    if not base_out.is_absolute():
        base_out = (REPO_ROOT / base_out).resolve()

    run_name = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    run_root = base_out / run_name
    for subdir in ("logs", "configs", "inference", "dataset"):
        (run_root / subdir).mkdir(parents=True, exist_ok=True)

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
