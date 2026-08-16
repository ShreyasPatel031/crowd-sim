from simulator.opera_mapper import lookup_names, opera_locator, simplified_html, slugify


def test_slugify_matches_opera_style_titles():
    assert slugify("All-New Amazon Kindle Paperwhite 16 GB") == "all_new_amazon_kindle_paperwhite_16_gb"


def test_lookup_aliases_add_to_cart():
    names = lookup_names("add_to_cart")
    assert names[0] == "add_to_cart"
    assert "buybox.purchase_form.add_to_cart" in names


def test_locator_includes_data_opera_name_and_html_name():
    sel = opera_locator("nav_bar.search_input")
    assert '[data-opera-name="nav_bar.search_input"]' in sel
    assert '[name="nav_bar.search_input"]' in sel


def test_coverage_families_detects_spine_and_extras():
    from simulator.opera_mapper import coverage_families

    names = [
        "nav_bar.search_input",
        "nav_bar.cart_button",
        "product_1",
        "pagination.2",
        "refinements.prime",
        "buybox.purchase_form.add_to_cart",
        "product_options.color.button_list.black",
        "reviews",
        "search_results.sort",
    ]
    cov = coverage_families(names)
    assert cov["search"] and cov["results"] and cov["pagination"]
    assert cov["filters"] and cov["add_to_cart"] and cov["variants"]

    html = simplified_html(
        [
            {"name": "nav_bar.search_input", "text": "Search Amazon"},
            {"name": "nav_bar.cart_button", "text": "Cart"},
        ],
        title="Amazon.com",
    )
    assert 'name="nav_bar.search_input"' in html
    assert 'name="nav_bar.cart_button"' in html
    assert "<title>Amazon.com</title>" in html
