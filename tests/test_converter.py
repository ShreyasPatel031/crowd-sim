import json
from pathlib import Path

from opera_repro.converter import ConverterConfig, convert_rows, session_to_examples
from opera_repro.data import load_fixture

FIXTURE = Path(__file__).parent / "fixtures" / "mini_sessions.json"


def test_history_examples_match_the_paper_pattern():
    rows = [row for row in load_fixture(FIXTURE) if row["session_id"] == "sess_shoes"]
    examples = session_to_examples("sess_shoes", rows, "train", ConverterConfig())
    assert [ex["gold_call"] for ex in examples] == [
        'type_and_submit("nav_bar.search_input", "running shoes")',
        'click("product_4")',
        'click("reviews")',
        "terminate()",
    ]

    first = examples[0]
    assert "(no previous actions)" in first["prompt_text"]
    assert first["messages"][-1]["content"] == '{"type": "type_and_submit", "name": "nav_bar.search_input", "text": "running shoes"}'

    second = examples[1]
    assert '"text": "running shoes"' in second["prompt_text"]
    assert "Current observation:" in second["prompt_text"]
    assert "product_4" in second["prompt_text"]
    assert second["messages"][-1]["content"] == '{"type": "click", "name": "product_4"}'

    third = examples[2]
    assert '"name": "product_4"' in third["prompt_text"]
    assert third["gold_action"]["name"] == "reviews"


def test_session_split_has_no_leakage():
    rows = load_fixture(FIXTURE)
    examples = convert_rows(rows, ConverterConfig(split_mode="official", seed=42))
    ids = {split: {row["session_id"] for row in split_rows} for split, split_rows in examples.items()}
    assert ids["test"] == {"sess_heldout"}
    assert not (ids["train"] & ids["test"])
    assert not (ids["train"] & ids["val"])
    assert not (ids["val"] & ids["test"])
    assert "sess_heldout" not in ids["train"]


def test_random_split_is_session_level():
    rows = load_fixture(FIXTURE)
    examples = convert_rows(rows, ConverterConfig(split_mode="random", seed=42))
    by_session = {}
    for split, split_rows in examples.items():
        for row in split_rows:
            by_session.setdefault(row["session_id"], set()).add(split)
    assert all(len(splits) == 1 for splits in by_session.values())
    assert examples["train"] and examples["test"]


def test_skip_first_action_drops_step_zero():
    rows = [row for row in load_fixture(FIXTURE) if row["session_id"] == "sess_shoes"]
    examples = session_to_examples(
        "sess_shoes",
        rows,
        "train",
        ConverterConfig(include_first_action=False),
    )
    assert examples[0]["gold_action"]["name"] == "product_4"
    assert json.loads(examples[0]["messages"][-1]["content"])["type"] == "click"
