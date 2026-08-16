"""Session-level split + (history, current UI) → next-action examples."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Sequence

from opera_repro.actions import Action, action_from_opera_row, is_purchase_action
from opera_repro.html_utils import compress_html, render_candidates
from opera_repro.prompts import build_messages, format_user_prompt


@dataclass
class ConverterConfig:
    include_first_action: bool = True
    include_rationale: bool = False
    max_history_steps: int = 4
    max_current_html_chars: int = 4000
    max_history_html_chars: int = 800
    # "candidates" builds the current observation from every named element on
    # the page. "window" splices a window around the gold element, which leaks
    # the label into the input and is kept only for reproducing old runs.
    observation_mode: str = "candidates"
    max_label_chars: int = 90
    seed: int = 42
    split_mode: str = "official"
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10


def rows_to_sessions(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        session_id = str(row.get("session_id") or "").strip()
        if not session_id:
            continue
        sessions[session_id].append(row)
    for session_id, steps in sessions.items():
        steps.sort(key=lambda row: (str(row.get("timestamp") or ""), str(row.get("action_id") or "")))
        sessions[session_id] = steps
    return dict(sessions)


def assign_session_splits(
    sessions: dict[str, list[dict[str, Any]]],
    config: ConverterConfig,
) -> dict[str, str]:
    """Map session_id → split. Splits happen BEFORE examples are created."""
    if config.split_mode == "official":
        return _official_splits(sessions, config.seed)
    if config.split_mode != "random":
        raise ValueError(f"Unknown split_mode: {config.split_mode}")
    return _ratio_splits(sorted(sessions), config)


def _hf_split_of_session(steps: list[dict[str, Any]]) -> str | None:
    labels = {str(step.get("hf_split") or "") for step in steps if step.get("hf_split")}
    labels.discard("")
    if not labels:
        return None
    if len(labels) > 1:
        raise ValueError(f"Session has mixed HuggingFace splits: {labels}")
    return labels.pop()


def _official_splits(sessions: dict[str, list[dict[str, Any]]], seed: int) -> dict[str, str]:
    test_ids = []
    train_pool = []
    for session_id, steps in sessions.items():
        label = _hf_split_of_session(steps)
        if label == "test":
            test_ids.append(session_id)
        else:
            train_pool.append(session_id)
    train_pool = _stable_shuffle(train_pool, seed)
    n_val = max(1, round(len(train_pool) * 0.10)) if train_pool else 0
    val_ids = set(train_pool[:n_val])
    assigned: dict[str, str] = {}
    for session_id in test_ids:
        assigned[session_id] = "test"
    for session_id in train_pool:
        assigned[session_id] = "val" if session_id in val_ids else "train"
    return assigned


def _ratio_splits(session_ids: Sequence[str], config: ConverterConfig) -> dict[str, str]:
    ids = _stable_shuffle(list(session_ids), config.seed)
    n = len(ids)
    if n == 0:
        return {}
    if n == 1:
        return {ids[0]: "train"}
    if n == 2:
        return {ids[0]: "train", ids[1]: "test"}
    n_train = int(n * config.train_ratio)
    n_val = int(n * config.val_ratio)
    n_train = min(max(1, n_train), n - 2)
    n_val = min(max(1, n_val), n - n_train - 1)
    assigned: dict[str, str] = {}
    for i, session_id in enumerate(ids):
        if i < n_train:
            assigned[session_id] = "train"
        elif i < n_train + n_val:
            assigned[session_id] = "val"
        else:
            assigned[session_id] = "test"
    return assigned


def _stable_shuffle(items: list[str], seed: int) -> list[str]:
    def key(item: str) -> str:
        return hashlib.sha256(f"{seed}:{item}".encode()).hexdigest()

    return sorted(items, key=key)


def build_observation(html: str | None, gold: Action, config: ConverterConfig) -> str:
    """Render the current page for the model.

    In "candidates" mode nothing about `gold` is consulted, so the observation
    is identical to what the live agent can build without knowing the answer.
    """
    if config.observation_mode == "candidates":
        return render_candidates(
            html,
            config.max_current_html_chars,
            max_label_chars=config.max_label_chars,
        )
    if config.observation_mode != "window":
        raise ValueError(f"Unknown observation_mode: {config.observation_mode}")
    return compress_html(html, config.max_current_html_chars, must_keep_name=gold.name)


def session_to_examples(
    session_id: str,
    steps: Sequence[dict[str, Any]],
    split: str,
    config: ConverterConfig,
) -> list[dict[str, Any]]:
    parsed: list[tuple[dict[str, Any], Action]] = []
    for row in steps:
        action = action_from_opera_row(row)
        if action is None:
            continue
        parsed.append((row, action))
    if not parsed:
        return []

    examples = []
    start = 0 if config.include_first_action else 1
    for t in range(start, len(parsed)):
        row, gold = parsed[t]
        history_pairs: list[tuple[str, Action]] = []
        hist_start = max(0, t - config.max_history_steps)
        for prev_row, prev_action in parsed[hist_start:t]:
            prev_html = compress_html(
                prev_row.get("simplified_html"),
                config.max_history_html_chars,
                must_keep_name=prev_action.name,
            )
            history_pairs.append((prev_html, prev_action))
        current_html = build_observation(row.get("simplified_html"), gold, config)
        rationale = str(row.get("rationale") or "").strip() if config.include_rationale else ""
        messages = build_messages(
            history_pairs,
            current_html,
            gold,
            rationale=rationale or None,
            reasoning_prompt=config.include_rationale,
        )
        examples.append(
            {
                "session_id": session_id,
                "action_id": row.get("action_id"),
                "step_index": t,
                "n_steps": len(parsed),
                "split": split,
                "gold_action": gold.to_dict(),
                "gold_call": gold.to_call(),
                "click_type": row.get("click_type") or "",
                "rationale": rationale,
                "is_session_outcome": t == len(parsed) - 1,
                "is_purchase": is_purchase_action(row, gold),
                "messages": messages,
                "prompt_text": format_user_prompt(history_pairs, current_html),
            }
        )
    return examples


def convert_rows(
    rows: Sequence[dict[str, Any]],
    config: ConverterConfig | None = None,
) -> dict[str, list[dict[str, Any]]]:
    config = config or ConverterConfig()
    sessions = rows_to_sessions(rows)
    splits = assign_session_splits(sessions, config)
    _assert_no_leakage(splits)

    out: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for session_id, steps in sessions.items():
        split = splits[session_id]
        out[split].extend(session_to_examples(session_id, steps, split, config))
    return out


def _assert_no_leakage(splits: dict[str, str]) -> None:
    buckets: dict[str, set[str]] = defaultdict(set)
    for session_id, split in splits.items():
        buckets[split].add(session_id)
    overlap_tv = buckets["train"] & buckets["val"]
    overlap_tt = buckets["train"] & buckets["test"]
    overlap_vt = buckets["val"] & buckets["test"]
    if overlap_tv or overlap_tt or overlap_vt:
        raise AssertionError(
            f"Session leakage: train∩val={len(overlap_tv)} "
            f"train∩test={len(overlap_tt)} val∩test={len(overlap_vt)}"
        )


def iter_jsonl(path: str) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def split_stats(examples_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for split, rows in examples_by_split.items():
        sessions = {row["session_id"] for row in rows}
        types: dict[str, int] = defaultdict(int)
        for row in rows:
            types[row["gold_action"]["type"]] += 1
        stats[split] = {
            "sessions": len(sessions),
            "examples": len(rows),
            "action_types": dict(types),
        }
    return stats
