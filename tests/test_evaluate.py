from opera_repro.evaluate import evaluate_predictions, format_report


def _row(session_id: str, gold: dict, outcome: bool = False) -> dict:
    return {
        "session_id": session_id,
        "gold_action": gold,
        "is_session_outcome": outcome,
    }


def test_oracle_predictions_are_perfect():
    rows = [
        _row("a", {"type": "click", "name": "reviews"}),
        _row("a", {"type": "terminate"}, outcome=True),
        _row("b", {"type": "type_and_submit", "name": "nav_bar.search_input", "text": "yoga mat"}),
    ]
    preds = [
        '{"type": "click", "name": "reviews"}',
        "terminate()",
        '{"type": "type_and_submit", "name": "nav_bar.search_input", "text": "yoga mat"}',
    ]
    result = evaluate_predictions(rows, preds)
    assert result.session_macro_accuracy == 1.0
    assert result.micro_accuracy == 1.0
    assert result.n_illegal == 0


def test_wrong_target_is_incorrect_even_if_type_matches():
    rows = [_row("a", {"type": "click", "name": "reviews"})]
    result = evaluate_predictions(rows, ['{"type": "click", "name": "add_to_cart"}'])
    assert result.micro_accuracy == 0.0
    assert result.action_type_accuracy == 1.0


def test_session_macro_does_not_let_long_sessions_dominate():
    rows = [_row("short", {"type": "click", "name": "reviews"})]
    rows += [_row("long", {"type": "click", "name": "x"}) for _ in range(9)]
    preds = ['{"type": "click", "name": "reviews"}'] + ['{"type": "click", "name": "wrong"}'] * 9
    result = evaluate_predictions(rows, preds)
    assert result.n_examples == 10
    assert result.micro_accuracy == 0.1
    assert result.session_macro_accuracy == 0.5
    report = format_report(result)
    assert "session-macro exact" in report
