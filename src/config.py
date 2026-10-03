"""Configuration loading. All paths are resolved relative to the repository root."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path | None = None) -> dict:
    with open(path or ROOT / "config.yaml") as fh:
        cfg = yaml.safe_load(fh)
    cfg["paths"] = {k: ROOT / v for k, v in cfg["paths"].items()}
    return cfg
