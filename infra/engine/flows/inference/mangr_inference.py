from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infra.utils.log.logger import logger
from infra.engine.flows.common.config_loader import get_execution_config
from infra.engine.flows.inference.onnx_backend import run_onnx
from infra.engine.flows.inference.pytorch_backend import run_pytorch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="nn_framework inference manager")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-profile", default="r18", choices=["r18", "r50"])
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--onnx-model", default="")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--score-thr", type=float, default=None)
    parser.add_argument("--overrides", nargs="*", default=[])
    return parser.parse_args()


def invoke(args: argparse.Namespace) -> None:
    app_config = get_execution_config(config_path=args.config, overrides=args.overrides)
    if args.score_thr is None:
        args.score_thr = float(app_config.runtime.common.score_threshold)

    if args.onnx_model:
        run_onnx(args, logger)
        return
    run_pytorch(args, logger)


def main() -> None:
    args = parse_args()
    invoke(args)


if __name__ == "__main__":
    main()
