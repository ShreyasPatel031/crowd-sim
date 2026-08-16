"""Load OPeRA-filtered actions from HuggingFace or a local JSONL fixture."""

from __future__ import annotations

from typing import Any

KEEP_FIELDS = (
    "session_id",
    "action_id",
    "timestamp",
    "action_type",
    "click_type",
    "semantic_id",
    "element_meta",
    "simplified_html",
    "input_text",
    "rationale",
)


def load_opera_filtered(cache_dir: str | None = None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset("NEU-HAI/OPeRA", "filtered_action", cache_dir=cache_dir)
    rows: list[dict[str, Any]] = []
    for split_name, split in dataset.items():
        for raw in split:
            row = {field: raw.get(field) for field in KEEP_FIELDS}
            row["hf_split"] = split_name
            rows.append(row)
    return rows


def load_fixture(path: str) -> list[dict[str, Any]]:
    import json

    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    for row in payload:
        row = dict(row)
        row.setdefault("hf_split", row.get("split", "train"))
        rows.append(row)
    return rows
