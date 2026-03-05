from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schema_app import AppConfig


_ACTIVE_APP_CONFIG: ContextVar["AppConfig | None"] = ContextVar("active_app_config", default=None)


def set_active_app_config(config: "AppConfig") -> None:
    _ACTIVE_APP_CONFIG.set(config)


def try_get_active_app_config() -> "AppConfig | None":
    return _ACTIVE_APP_CONFIG.get()


def get_active_app_config() -> "AppConfig":
    config = _ACTIVE_APP_CONFIG.get()
    if config is None:
        raise RuntimeError(
            "Active AppConfig is not set. Build flow runtime first (build_flow_runtime) "
            "or set it explicitly with set_active_app_config(config)."
        )
    return config
