from simulator.competitors import extract_asin, search_query_for_persona
from simulator.personas import get_personas, shopper_task
from simulator.verdict import aggregate_panel, combine_listing_evals, parse_verdict


def test_parse_verdict_json():
    text = """
    {
      "verdict": "buy",
      "product_selected": "eosera drops",
      "product_url": "https://amazon.com/dp/x",
      "confidence": 72,
      "rationale": "Matches ear ache and the unit price looks fair.",
      "price_perception": "fair vs nearby listings",
      "trust_concerns": ["few reviews"],
      "conversion_blockers": []
    }
    """
    parsed = parse_verdict(text)
    assert parsed["verdict"] == "buy"
    assert parsed["confidence"] == 72
    assert parsed["buy_likelihood"] == 72
    assert "ear ache" in parsed["rationale"]


def test_opera_personas_and_task_prompt():
    people = get_personas([])
    assert people
    assert people[0].get("source") == "opera"
    task = shopper_task(people[0], "https://amazon.com/dp/x", ["https://amazon.com/dp/y"], "Would you buy?")
    assert "OPeRA" in task
    assert "buy_likelihood" in task


def test_aggregate_panel():
    summary = aggregate_panel(
        [
            {"verdict": {"verdict": "buy", "confidence": 80, "conversion_blockers": ["price"]}},
            {"verdict": {"verdict": "no", "confidence": 60, "conversion_blockers": ["reviews"]}},
        ]
    )
    assert summary["counts"]["buy"] == 1
    assert summary["n"] == 2
    assert summary["avg_confidence"] == 70
    assert summary["avg_buy_likelihood"] == 60
    assert summary["panel_verdict"] == "uncertain"
    assert "iterate" in summary["panel_label"].lower()
    assert summary["top_blockers"]


def test_combine_skips_same_brand_and_uses_likelihood():
    product = {
        "listing_title": "Sports Research Fish Oil Mini-Softgels",
        "buy_likelihood": 58,
        "would_buy_this": "maybe",
        "rationale": "Fine but not a must-buy.",
        "url": "https://www.amazon.com/dp/B0CHN9X9S2",
    }
    twin = {
        "listing_title": "Sports Research Triple Strength Omega 3 Fish Oil",
        "buy_likelihood": 90,
        "would_buy_this": "buy",
        "url": "https://www.amazon.com/dp/B07DX89ZHN",
    }
    other = {
        "listing_title": "Nature Made Fish Oil 1200 mg Softgels",
        "buy_likelihood": 80,
        "would_buy_this": "buy",
        "url": "https://www.amazon.com/dp/B00L4QJLQ8",
        "rationale": "Cheaper per serving.",
    }
    same = combine_listing_evals("https://www.amazon.com/dp/B0CHN9X9S2", product, [twin])
    assert same["buy_likelihood"] == 58
    assert same["verdict"] == "maybe"
    vs_other = combine_listing_evals("https://www.amazon.com/dp/B0CHN9X9S2", product, [twin, other])
    assert vs_other["verdict"] == "no"
    assert vs_other["buy_likelihood"] < 58
    assert "Nature Made" in vs_other["product_selected"]


def test_opera_competitor_match():
    from simulator.competitors import competitors_from_opera, same_listing

    found = competitors_from_opera(
        "fish oil",
        {"B0CHN9X9S2"},
        4,
        seed_title="Sports Research Fish Oil Mini-Softgels",
        seed_brand="Sports Research",
    )
    asins = {item["asin"] for item in found}
    assert "B0CHN9X9S2" not in asins
    assert "B07DX89ZHN" not in asins
    assert "B014LDT0ZM" in asins
    for item in found:
        assert item["url"].startswith("https://www.amazon.com/dp/")
        assert "oil" in (item.get("title") or "").lower()
    assert competitors_from_opera("ear ache drops", set(), 4) == []
    assert extract_asin("https://www.amazon.com/dp/B00I15SB16/ref=sr") == "B00I15SB16"
    q = search_query_for_persona(
        "eosera Ear Pain MD Earache Relief Drops",
        {"priorities": ["best value"], "avoids": [], "budget": "About $50/month"},
    )
    assert "best value" in q
    assert same_listing(
        "Sports Research Fish Oil Mini-Softgels - Easy to Swallow Omega-3",
        "Sports Research",
        "Sports Research Triple Strength Omega 3 Fish Oil - Burpless Fish Oil Supplement",
        "Sports Research",
    )
    assert not same_listing(
        "Sports Research Fish Oil Mini-Softgels",
        "Sports Research",
        "Nature Made Fish Oil 1200 mg Softgels",
        "Nature Made",
    )


def test_persona_search_finds_grad_student():
    from simulator.persona_search import search_personas

    results = search_personas("PhD student tight budget reads reviews", limit=5)
    assert results
    assert any("student" in (row.get("label") or "").lower() for row in results)


def test_similar_personas_for_seed():
    from simulator.persona_search import similar_personas

    results = similar_personas("85aeec61-2d4e-489d-93bf-76c928d2d795", limit=3)
    assert len(results) == 3
    assert all(row["score"] > 0.5 for row in results)


def test_example_product_personas():
    from simulator.persona_search import list_example_products

    examples = list_example_products()
    assert len(examples) == 5
    asins = {row["asin"] for row in examples}
    assert asins == {"B0CHN9X9S2", "B0B49MZLDB", "B07D37PQGL", "B07YYJLKFW", "B0D5ZXKY3M"}
    fish = next(row for row in examples if row["asin"] == "B0CHN9X9S2")
    assert fish["primary_persona_type"] == "Budget grad"
