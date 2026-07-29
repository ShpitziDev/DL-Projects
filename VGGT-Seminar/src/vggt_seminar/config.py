from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .paths import find_project_root


ALLOWED_DEVICES = {"auto", "cuda", "cpu"}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    root = find_project_root()
    default = yaml.safe_load((root / "configs/default.yaml").read_text(encoding="utf-8"))
    config = default if path is None else _merge(default, yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    if config["device"] not in ALLOWED_DEVICES:
        raise ValueError(f"device must be one of {sorted(ALLOWED_DEVICES)}")
    return config
