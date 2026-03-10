from __future__ import annotations

import inspect
import types
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Mapping


def constructor_accepts_parameter(
    *, module: str, class_name: str, parameter_name: str
) -> bool:
    model_module = import_module(module)
    constructor = getattr(model_module, class_name).__init__
    signature = inspect.signature(constructor)
    if parameter_name in signature.parameters:
        return True
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def patch_yaml_class_section(
    *,
    payload: dict[str, Any],
    module: str,
    class_name: str,
    injected: Mapping[str, Any],
) -> bool:
    class_payload = payload.get(class_name)
    if not isinstance(class_payload, dict):
        class_payload = {}

    changed = False
    for key, value in injected.items():
        supports_key = constructor_accepts_parameter(
            module=module, class_name=class_name, parameter_name=key
        )
        if value is None:
            if key in class_payload:
                class_payload.pop(key, None)
                changed = True
            continue
        if supports_key:
            if class_payload.get(key) != value:
                class_payload[key] = value
                changed = True
            continue
        if key in class_payload:
            class_payload.pop(key, None)
            changed = True

    if class_payload:
        payload[class_name] = class_payload
    return changed


def patch_yaml_include_tokens(
    *, payload: dict[str, Any], replacements: Mapping[str, str]
) -> bool:
    includes = payload.get("__include__")
    if not isinstance(includes, list):
        return False

    changed = False
    resolved_includes: list[str] = []
    for include_path in includes:
        include_text = str(include_path)
        replaced = include_text
        for token, root_path in replacements.items():
            if include_text.startswith(token):
                relative = include_text[len(token) :]
                replaced = str((Path(root_path) / relative).resolve())
                changed = True
                break
        resolved_includes.append(replaced)

    if changed:
        payload["__include__"] = resolved_includes
    return changed


def inject_runtime_functions(
    *,
    target: Any,
    module: str,
    class_name: str,
    injected: Mapping[str, Callable[..., Any] | None],
) -> bool:
    model_module = import_module(module)
    class_type = getattr(model_module, class_name)
    if not isinstance(target, class_type):
        return False

    changed = False
    for function_name, function_impl in injected.items():
        if function_impl is None:
            if hasattr(target, function_name):
                delattr(target, function_name)
                changed = True
            continue
        bound = types.MethodType(function_impl, target)
        current = getattr(target, function_name, None)
        current_func = getattr(current, "__func__", None)
        if current_func is not function_impl:
            setattr(target, function_name, bound)
            changed = True
    return changed
