"""Tiny helpers shared by scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else ROOT / "configs" / "default.yaml"
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def converter_config_from_yaml(cfg: dict[str, Any]):
    from opera_repro.converter import ConverterConfig

    dataset = cfg.get("dataset", {})
    converter = cfg.get("converter", {})
    return ConverterConfig(
        include_first_action=converter.get("include_first_action", True),
        include_rationale=converter.get("include_rationale", False),
        max_history_steps=converter.get("max_history_steps", 4),
        max_current_html_chars=converter.get("max_current_html_chars", 4000),
        max_history_html_chars=converter.get("max_history_html_chars", 800),
        observation_mode=converter.get("observation_mode", "candidates"),
        max_label_chars=converter.get("max_label_chars", 90),
        seed=cfg.get("seed", 42),
        split_mode=dataset.get("split_mode", "official"),
        train_ratio=dataset.get("train_ratio", 0.8),
        val_ratio=dataset.get("val_ratio", 0.1),
        test_ratio=dataset.get("test_ratio", 0.1),
    )
