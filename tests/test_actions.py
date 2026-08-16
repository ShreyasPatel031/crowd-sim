from opera_repro.actions import Action, action_from_opera_row, actions_equal, parse_action


def test_parse_json_and_call_syntax():
    gold = Action(type="click", name="reviews")
    assert actions_equal(parse_action('{"type": "click", "name": "reviews"}'), gold)
    assert actions_equal(parse_action('click("reviews")'), gold)
    assert actions_equal(parse_action("```json\n{\"type\":\"click\",\"name\":\"reviews\"}\n```"), gold)


def test_type_and_submit_requires_text():
    gold = Action(type="type_and_submit", name="nav_bar.search_input", text="running shoes")
    assert actions_equal(
        parse_action(
            '{"type": "type_and_submit", "name": "nav_bar.search_input", "text": "running shoes"}'
        ),
        gold,
    )
    assert not actions_equal(
        parse_action(
            '{"type": "type_and_submit", "name": "nav_bar.search_input", "text": "trail shoes"}'
        ),
        gold,
    )
    assert not actions_equal(
        parse_action('{"type": "click", "name": "nav_bar.search_input"}'),
        gold,
    )


def test_terminate_ignores_name():
    gold = Action(type="terminate")
    assert actions_equal(parse_action('{"type": "terminate"}'), gold)
    assert actions_equal(parse_action("terminate()"), gold)
    assert not actions_equal(parse_action('{"type": "click", "name": "reviews"}'), gold)


def test_illegal_output_is_not_a_match():
    gold = Action(type="click", name="reviews")
    assert parse_action("I think the user will click reviews") is None
    assert not actions_equal(None, gold)


def test_opera_input_maps_to_type_and_submit():
    action = action_from_opera_row(
        {
            "action_type": "input",
            "semantic_id": "nav_bar.search_input",
            "input_text": "running shoes",
        }
    )
    assert action == Action(type="type_and_submit", name="nav_bar.search_input", text="running shoes")
    assert action.to_call() == 'type_and_submit("nav_bar.search_input", "running shoes")'
