from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

import yaml
from loguru import logger as _base_logger

_CONFIGURED = False
_CONSOLE_FORMAT = (
    "<bold><green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | "
    "<cyan>{file}:{function}:{line}</cyan> - <level>{message}</level></bold>"
)
_FILE_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{function}:{line} - {message}"
_DEFAULT_LEVEL_COLORS = {
    "TRACE": "<white>",
    "DEBUG": "<white>",
    "INFO": "<white>",
    "WARNING": "<yellow>",
    "ERROR": "<red>",
    "CRITICAL": "<RED><white>",
}


def _configure_level_colors(overrides: Optional[dict[str, str]] = None) -> None:
    colors = dict(_DEFAULT_LEVEL_COLORS)
    if overrides:
        for level_name, color in overrides.items():
            if not isinstance(level_name, str) or not isinstance(color, str):
                continue
            colors[level_name.upper()] = color
    for level_name, color in colors.items():
        _base_logger.level(level_name, color=color)


def _framework_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_config_path() -> Path:
    return Path(__file__).resolve().parent / "config.yaml"


def _resolve_log_path(raw_path: str, root: Path) -> str:
    path = Path(raw_path)
    if path.is_absolute():
        return raw_path
    return str((root / raw_path).resolve())


def setup_logger(config_path: Optional[str | Path] = None, force: bool = False) -> None:
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    root = _framework_root()
    cfg_path = Path(config_path) if config_path is not None else _default_config_path()
    if not cfg_path.is_absolute():
        cfg_path = (root / cfg_path).resolve()

    with cfg_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
        config = payload.get("logging", {})

    console_cfg = config.get("console", {})
    level_colors = config.get("level_colors")
    if not isinstance(level_colors, dict):
        level_colors = console_cfg.get("level_colors", {})
    if not isinstance(level_colors, dict):
        level_colors = {}

    _base_logger.remove()
    _configure_level_colors(level_colors)

    if console_cfg.get("enabled", True):
        _base_logger.add(
            sys.stderr,
            level=console_cfg.get("level", "DEBUG"),
            format=_CONSOLE_FORMAT,
            colorize=True,
        )

    file_cfg = config.get("file", {})
    if file_cfg.get("enabled", True):
        configured_path = str(file_cfg.get("path", "logs/{time:YYYY-MM-DD__HH-mm-ss}.log"))
        run_root = os.environ.get("NN_FRAMEWORK_RUN_DIR", "").strip()
        if run_root:
            file_path = _resolve_log_path(configured_path, Path(run_root).resolve())
        else:
            file_path = _resolve_log_path(configured_path, root)
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        _base_logger.add(
            file_path,
            level=file_cfg.get("level", "INFO"),
            rotation=file_cfg.get("rotation", "10 MB"),
            retention=file_cfg.get("retention", "1 week"),
            format=_FILE_FORMAT,
            colorize=False,
        )

    _CONFIGURED = True


def get_logger(**bind: Any):
    setup_logger()
    return _base_logger.bind(**bind) if bind else _base_logger


setup_logger()
logger = get_logger()
