from __future__ import annotations

import os
import sys

from infra.utils.log import logger, setup_logger

from .constants import REPO_ROOT
from .parser import parse_args
from .run_layout import prepare_run_layout


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    args = parse_args()
    run_root = prepare_run_layout(args)
    args.run_root = str(run_root)
    os.environ["NN_FRAMEWORK_RUN_DIR"] = str(run_root)
    setup_logger(force=True)
    logger.info("Run directory: {}", run_root)
    args.handler(args)
