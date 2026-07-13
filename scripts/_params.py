"""Shared client-side parameter resolution for the generation scripts.

Resolves the `params` a script sends with each request from two client-side
sources: a per-system config file (config/client_params.yaml) and optional
one-off --param KEY=VALUE overrides. The gateway then merges the registry's
default_params underneath, so the full precedence is:

    registry default_params (server) < client config file < --param CLI
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "client_params.yaml"


def _coerce(value: str) -> Any:
    """Best-effort scalar typing for --param values (int, float, bool, else str)."""
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    return value


def load_config_params(config_path: Path, system: str) -> Dict[str, Any]:
    if not config_path.exists():
        return {}
    data = yaml.safe_load(config_path.read_text()) or {}
    systems = data.get("systems", data)  # accept {systems: {...}} or a flat mapping
    return dict(systems.get(system) or {})


def parse_overrides(pairs: Optional[List[str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"--param must be KEY=VALUE, got {pair!r}")
        key, val = pair.split("=", 1)
        out[key.strip()] = _coerce(val.strip())
    return out


def resolve_params(config_path: Path, system: str, overrides: Optional[List[str]]) -> Dict[str, Any]:
    params = load_config_params(config_path, system)
    params.update(parse_overrides(overrides))
    return params
