from __future__ import annotations

from infra.interfaces import LoggerPort
from infra.utils.log import logger as default_logger


class LoguruLoggerAdapter(LoggerPort):
    def __init__(self, bound_logger=None) -> None:
        self._logger = bound_logger or default_logger

    def info(self, message: str, *args) -> None:
        self._logger.info(message, *args)

    def warning(self, message: str, *args) -> None:
        self._logger.warning(message, *args)

    def error(self, message: str, *args) -> None:
        self._logger.error(message, *args)
