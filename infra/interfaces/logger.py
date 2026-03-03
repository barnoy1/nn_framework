from __future__ import annotations

from abc import ABC, abstractmethod


class LoggerPort(ABC):
    @abstractmethod
    def info(self, message: str, *args) -> None:
        raise NotImplementedError

    @abstractmethod
    def warning(self, message: str, *args) -> None:
        raise NotImplementedError

    @abstractmethod
    def error(self, message: str, *args) -> None:
        raise NotImplementedError
