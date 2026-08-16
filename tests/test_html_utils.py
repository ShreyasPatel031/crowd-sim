from opera_repro.converter import ConverterConfig, build_observation
from opera_repro.actions import Action
from opera_repro.html_utils import compress_html, extract_candidates, render_candidates


def test_compress_keeps_named_target_when_truncating():
    html = "START" + ("x" * 5000) + '<div name="reviews">Customer reviews</div>' + ("y" * 5000) + "END"
    out = compress_html(html, max_chars=800, must_keep_name="reviews")
    assert 'name="reviews"' in out
    assert len(out) <= 800


def _page(n_filler: int = 200) -> str:
    filler = "".join(
        f'<li name="product_{i}"><a name="product_{i}.title">Item {i}</a></li>' for i in range(n_filler)
    )
    return (
        "<html><head><title>Amazon</title></head><body>"
        '<input name="nav_bar.search_input" aria-label="Search Amazon">'
        f"{filler}"
        '<span name="go_to_cart">Go to Cart</span>'
        "</body></html>"
    )


def test_candidates_capture_name_tag_and_label():
    candidates = {c.name: c for c in extract_candidates(_page(2))}
    assert candidates["nav_bar.search_input"].tag == "input"
    assert candidates["nav_bar.search_input"].label == "Search Amazon"
    assert candidates["product_1.title"].label == "Item 1"
    assert candidates["go_to_cart"].label == "Go to Cart"


def test_candidate_names_survive_a_tight_budget():
    """Labels shrink to fit; the action space never gets truncated away."""
    html = _page(300)
    out = render_candidates(html, max_chars=500)
    for name in ("nav_bar.search_input", "product_299.title", "go_to_cart"):
        assert f'name="{name}"' in out


def test_observation_does_not_depend_on_the_gold_action():
    """Regression guard: the current-page observation must not leak the label."""
    html = _page(50)
    config = ConverterConfig(observation_mode="candidates", max_current_html_chars=150_000)
    first = build_observation(html, Action(type="click", name="go_to_cart"), config)
    second = build_observation(html, Action(type="click", name="product_7.title"), config)
    assert first == second
    assert 'name="go_to_cart"' in first
    assert 'name="product_7.title"' in first


def test_reason_only_when_human_wrote_one():
    from opera_repro.actions import actions_equal, parse_action
    from opera_repro.prompts import gold_target_json

    gold = Action(type="click", name="go_to_cart")
    with_text = gold_target_json(gold, "cart looked cheaper")
    without = gold_target_json(gold, "")
    assert with_text == '{"reason": "cart looked cheaper", "type": "click", "name": "go_to_cart"}'
    assert without == '{"type": "click", "name": "go_to_cart"}'
    assert '"reason"' not in without
    assert actions_equal(parse_action(with_text), gold)
    assert actions_equal(parse_action(without), gold)


def test_window_mode_still_leaks_and_is_opt_in():
    html = _page(300)
    config = ConverterConfig(observation_mode="window", max_current_html_chars=800)
    out = build_observation(html, Action(type="click", name="go_to_cart"), config)
    assert 'name="go_to_cart"' in out
