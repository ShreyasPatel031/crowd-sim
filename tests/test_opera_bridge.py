from opera_repro.converter import ConverterConfig, session_to_examples
from opera_repro.data import load_fixture
from simulator.opera_bridge import dummy_policy, observation_html

from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "mini_sessions.json"


def _shoes_examples():
    rows = [row for row in load_fixture(FIXTURE) if row["session_id"] == "sess_shoes"]
    return session_to_examples("sess_shoes", rows, "train", ConverterConfig())


def test_dummy_returns_gold_when_name_is_on_page():
    examples = _shoes_examples()
    first = examples[0]
    html = observation_html(first)
    decision = dummy_policy(html, first["gold_action"])
    assert decision["gold_on_page"] is True
    assert decision["opera"] == {
        "type": "type_and_submit",
        "name": "nav_bar.search_input",
        "text": "running shoes",
    }
    assert "nav_bar.search_input" in decision["named_targets"]


def test_dummy_flags_missing_name():
    html = '<html><body><div name="reviews">Reviews</div></body></html>'
    decision = dummy_policy(html, {"type": "click", "name": "add_to_cart"})
    assert decision["gold_on_page"] is False
    assert decision["action"] == "click"


def test_terminate_does_not_need_a_named_node():
    decision = dummy_policy("<html></html>", {"type": "terminate"})
    assert decision["gold_on_page"] is True
    assert decision["action"] == "done"
