"""Unified YAML config loader (Section 3.3 of system_architecture.md)."""
from __future__ import annotations

import os
import yaml
from types import SimpleNamespace


def _to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def load_config(path: str = None) -> SimpleNamespace:
    """Load YAML config and expose dotted access (cfg.video.source_dir)."""
    if path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "..", "configs", "pipeline.yaml")
    path = os.path.abspath(path)
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    raw["__path__"] = path
    raw["__root__"] = os.path.dirname(path)
    return _to_namespace(raw)
