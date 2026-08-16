"""Map live Amazon DOM → OPeRA semantic_id (name=).

Amazon does not ship OPeRA names. ShoppingFlow stamped them at collection
time. This adapter stamps `data-opera-name` on live nodes (it does not
overwrite Amazon form `name=` like field-keywords) and builds simplified
HTML the train/eval prompt expects.

High-frequency OPeRA ids (from filtered_action train):
  nav_bar.search_input / search_button / cart_button / homepage
  buybox.purchase_form.add_to_cart / buy_now
  search_results.{slug}  and  product_N  (live index)
  reviews, check_out, cart_side_bar.go_to_cart
"""

from __future__ import annotations

import html as html_lib
import re
from typing import Any

# Canonical OPeRA name → other names the dummy/model might emit.
ALIASES: dict[str, tuple[str, ...]] = {
    "nav_bar.search_input": ("field-keywords",),
    "nav_bar.cart_button": ("go_to_cart", "nav_bar.cart"),
    "go_to_cart": ("nav_bar.cart_button", "nav_bar.cart", "cart_side_bar.go_to_cart"),
    "nav_bar.cart": ("nav_bar.cart_button", "go_to_cart"),
    "cart_side_bar.go_to_cart": ("go_to_cart", "nav_bar.cart_button"),
    "buybox.purchase_form.add_to_cart": (
        "add_to_cart",
        "buybox.one_time_purchase.purchase_form.add_to_cart",
        "buybox.buy_new.purchase_form.add_to_cart",
    ),
    "add_to_cart": ("buybox.purchase_form.add_to_cart",),
    "buybox.purchase_form.buy_now": (
        "buy_now",
        "add_to_cart.buy_now",
        "buybox.one_time_purchase.purchase_form.buy_now",
    ),
    "buy_now": ("buybox.purchase_form.buy_now", "add_to_cart.buy_now"),
    "add_to_cart.buy_now": ("buybox.purchase_form.buy_now", "buy_now"),
    "check_out": ("proceed_to_checkout",),
    "proceed_to_checkout": ("check_out",),
}

STAMP_JS = r"""
() => {
  const assigned = [];
  const used = new Set();

  const visible = (el) => {
    if (!el) return false;
    const r = el.getBoundingClientRect();
    const s = getComputedStyle(el);
    if (r.width < 4 || r.height < 4) return false;
    if (s.visibility === 'hidden' || s.display === 'none' || s.opacity === '0') return false;
    if (el.getAttribute('type') === 'hidden') return false;
    return true;
  };

  const first = (sels) => {
    for (const sel of sels) {
      const el = document.querySelector(sel);
      if (el && visible(el)) return el;
    }
    for (const sel of sels) {
      const el = document.querySelector(sel);
      if (el) return el;
    }
    return null;
  };

  const slug = (text) => (text || '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 80);

  const stamp = (el, name, aliases = []) => {
    if (!el || !name || used.has(name)) return;
    el.setAttribute('data-opera-name', name);
    const extra = aliases.filter((a) => a && a !== name);
    if (extra.length) el.setAttribute('data-opera-aliases', extra.join(' '));
    used.add(name);
    extra.forEach((a) => used.add(a));
    assigned.push({
      name,
      aliases: extra,
      tag: el.tagName.toLowerCase(),
      id: el.id || '',
      text: (el.innerText || el.value || el.getAttribute('aria-label')
        || el.getAttribute('placeholder') || '').trim().slice(0, 80),
    });
  };

  stamp(
    first(['#twotabsearchtextbox', 'input[name="field-keywords"]:not([type="hidden"])']),
    'nav_bar.search_input',
  );
  stamp(first(['#nav-search-submit-button', 'input[type="submit"][value="Go"]']), 'nav_bar.search_button');
  stamp(
    first(['#nav-cart', '#nav-cart-count-container', 'a[href="/gp/cart/view.html"]', 'a[href*="/cart"]']),
    'nav_bar.cart_button',
    ['go_to_cart', 'nav_bar.cart'],
  );
  stamp(first(['#nav-logo-sprites', '#nav-logo', 'a.nav-logo-link']), 'nav_bar.homepage');
  stamp(first(['#nav-link-accountList', '#nav-link-accountList-nav-line-1']), 'nav_bar.account_and_list_button');
  stamp(first(['#nav-orders']), 'nav_bar.order_button');
  stamp(first(['#nav-hamburger-menu']), 'nav_bar.menu');
  stamp(first(['#searchDropdownBox']), 'nav_bar.search_drop_down_list');

  stamp(
    first(['#add-to-cart-button', 'input[name="submit.add-to-cart"]', 'input[name="submit.addToCart"]']),
    'buybox.purchase_form.add_to_cart',
    ['add_to_cart'],
  );
  stamp(
    first(['#buy-now-button', 'input[name="submit.buy-now"]']),
    'buybox.purchase_form.buy_now',
    ['buy_now', 'add_to_cart.buy_now'],
  );
  stamp(
    first(['#acrCustomerReviewLink', 'a[href*="#customerReviews"]', '#averageCustomerReviews']),
    'reviews',
  );
  stamp(first(['#feature-bullets', '#productFactsDesktopExpander', '#productOverview_feature_div']), 'about_this_item');
  stamp(first(['#quantity', 'select[name="quantity"]']), 'buybox.purchase_form.quantity_selector.drop_down_list.open_drop_down_list');

  stamp(
    first([
      'input[name="proceedToRetailCheckout"]',
      'input[name="proceedToCheckout"]',
      '[name="proceedToRetailCheckout"]',
      '#sc-buy-box-ptc-button',
      'input[name="proceedToALMCheckout-announce"]',
    ]),
    'check_out',
    ['proceed_to_checkout'],
  );
  stamp(
    first(['#sw-gtc', '#attach-sidesheet-view-cart-button', 'a#attach-sidesheet-view-cart-button']),
    'cart_side_bar.go_to_cart',
    ['go_to_cart'],
  );

  stamp(first(['#s-result-sort-select', 'form.s-result-sort-select', '[data-action="a-dropdown-button"]']), 'search_results.sort');

  const cards = [...document.querySelectorAll('[data-component-type="s-search-result"]')]
    .filter((el) => (el.getAttribute('data-asin') || '').length > 4);
  cards.slice(0, 16).forEach((card, i) => {
    const link = card.querySelector('h2 a[href*="/dp/"], h2 a, a.a-link-normal[href*="/dp/"]');
    const title = (card.querySelector('h2')?.innerText || link?.innerText || '').trim();
    const price = (card.querySelector('.a-price .a-offscreen')?.innerText || '').trim();
    const s = slug(title);
    const aliases = [];
    if (s) {
      aliases.push('search_results.' + s);
      aliases.push('search_results.' + s + '.view_product');
      aliases.push('search_results.' + s + '.product_name');
    }
    stamp(link || card, 'product_' + (i + 1), aliases);
    if (assigned.length) {
      assigned[assigned.length - 1].text = (title + (price ? ' ' + price : '')).slice(0, 80);
    }
  });

  const nextPage = first(['a.s-pagination-next', 'a[aria-label*="Go to next page"]']);
  stamp(nextPage, 'pagination.next');
  [...document.querySelectorAll('a.s-pagination-button, a.s-pagination-item')].forEach((el) => {
    const label = (el.getAttribute('aria-label') || el.innerText || '').trim();
    const m = label.match(/page\s+(\d+)/i) || label.match(/^(\d+)$/);
    if (m) stamp(el, 'pagination.' + m[1]);
  });

  stamp(first(['#low-price', 'input[name="low-price"]']), 'refinements.price_refinements.price_min_value');
  stamp(first(['#high-price', 'input[name="high-price"]']), 'refinements.price_refinements.price_max_value');
  stamp(
    first(['input.a-button-input[type="submit"][aria-labelledby*="price"]', '#a-autoid-1-announce']),
    'refinements.price_refinements.submit_price_range',
  );

  const filterRoot = document.querySelector('#s-refinements, #s-filters, #filters');
  if (filterRoot) {
    const filters = [...filterRoot.querySelectorAll('a, input[type="checkbox"], label')]
      .filter((el) => visible(el) && (el.innerText || el.getAttribute('aria-label') || '').trim().length > 1)
      .slice(0, 24);
    filters.forEach((el) => {
      const label = (el.innerText || el.getAttribute('aria-label') || '').trim();
      const s = slug(label).slice(0, 40);
      if (s) stamp(el.closest('a, label, li, span') || el, 'refinements.' + s);
    });
  }

  const stampSwatches = (nodes, kind) => {
    nodes.slice(0, 12).forEach((el) => {
      const label = (
        el.getAttribute('title') || el.getAttribute('alt') || el.getAttribute('aria-label')
        || el.innerText || el.querySelector('img')?.getAttribute('alt') || ''
      ).trim();
      const s = slug(label).slice(0, 40);
      if (!s) return;
      const clickable = el.closest('li, button, a, span') || el;
      stamp(clickable, 'product_options.' + kind + '.button_list.' + s, [
        'product_options.' + kind + '.' + s,
      ]);
    });
  };
  stampSwatches(
    [...document.querySelectorAll(
      '#variation_color_name li, #inline-twister-row-color_name li, [id*="color_name"] li, img.imgSwatch'
    )].filter(visible),
    'color',
  );
  stamp(
    first(['#dropdown_selected_size_name', '#native_dropdown_selected_size_name', '#variation_size_name .a-button-text']),
    'product_options.size.drop_down_list.open_drop_down_list',
  );
  stampSwatches(
    [...document.querySelectorAll('#variation_size_name li, #inline-twister-row-size_name li, [id*="size_name"] li')].filter(visible),
    'size',
  );
  stamp(
    first(['#dropdown_selected_color_name', '#variation_color_name .a-button-text']),
    'product_options.color.drop_down_list.open_drop_down_list',
  );

  stamp(first(['a[data-hook="see-all-reviews-link-foot"]', 'a[href*="/product-reviews/"]']), 'reviews.see_all');
  [...document.querySelectorAll('[data-hook="review-image"] img, .review-image-tile img')].slice(0, 8).forEach((el, i) => {
    stamp(el.closest('a, button, li') || el, 'reviews.reviews_with_images.images.' + i);
  });
  stamp(
    first(['.cr-lightbox-next', 'button[aria-label*="Next image"]', '.a-popover button[aria-label*="Next"]']),
    'reviews.popover.review_images.next',
  );
  stamp(
    first(['.cr-lightbox-close', '.a-popover-header .a-button-close', 'button[aria-label="Close"]']),
    'reviews.popover.close',
  );

  stamp(first(['#unclippedCoupon', 'i.a-icon-checkbox', 'label[id*="coupon"]', '[data-a-selector="coupon"]']), 'coupon.checkbox');
  stamp(
    first(['#rcx-subscribe-submit-button-announce', '#snsAccordionRowMiddle input', '[name="submit.subscribe"]']),
    'buybox.subscribe_save.purchase_form.set_up_now',
  );
  stamp(
    first(['#buybox-see-all-buying-choices', 'a[href*="offer-listing"]', '#a-autoid-offer-display-string-0']),
    'buybox.see_all_buying_options',
  );

  stamp(first(['#sc-buy-box-ptc-button', 'input[name="proceedToRetailCheckout"]', '[name="proceedToRetailCheckout"]']), 'check_out', ['proceed_to_checkout']);
  stamp(first(['input[name="chkSelectAll"]', '[name="submit.select-all"]', '#select-all-in-cart']), 'cart_header.select_all_items');
  [...document.querySelectorAll('#sc-active-cart [data-asin], #sc-active-cart .sc-list-item')].slice(0, 8).forEach((row) => {
    const title = (row.querySelector('span.a-truncate-full, span.sc-product-title, h4')?.innerText || '').trim();
    const s = slug(title).slice(0, 50) || ('item_' + (row.getAttribute('data-asin') || ''));
    const box = row.querySelector('input[type="checkbox"]');
    if (box) stamp(box, 'active_item_list.' + s + '.checkbox');
    const qty = row.querySelector('select[name*="quantity"], input[name*="quantity"]');
    if (qty) stamp(qty, 'active_item_list.' + s + '.quantity');
    const del = row.querySelector('input[value="Delete"], input[name*="delete"], [data-action="delete"]');
    if (del) stamp(del, 'active_item_list.' + s + '.delete');
  });

  [...document.querySelectorAll('.s-suggestion, div.s-suggestion-container div')].filter(visible).slice(0, 8).forEach((el) => {
    const s = slug(el.innerText).slice(0, 40);
    if (s) stamp(el, 'suggested_term.' + s);
  });

  stamp(first(['.a-carousel-goto-nextpage', 'a[aria-label="Carousel next page"]']), 'products_related_to_this_item.next_page');

  return assigned;
}
"""


def slugify(text: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return slug[:max_len]


def lookup_names(name: str) -> list[str]:
    """Ordered candidate semantic ids to try on the live DOM."""
    if not name:
        return []
    out = [name]
    for canon, alts in ALIASES.items():
        if name == canon or name in alts:
            out.append(canon)
            out.extend(alts)
    seen: set[str] = set()
    uniq: list[str] = []
    for item in out:
        if item and item not in seen:
            seen.add(item)
            uniq.append(item)
    return uniq


def opera_locator(name: str) -> str:
    """CSS selector that matches a stamped live node or frozen OPeRA HTML."""
    parts = []
    for candidate in lookup_names(name):
        esc = candidate.replace("\\", "\\\\").replace('"', '\\"')
        parts.append(f'[data-opera-name="{esc}"]')
        parts.append(f'[data-opera-aliases~="{esc}"]')
        parts.append(f'[name="{esc}"]')
    return ", ".join(parts)


def simplified_html(stamped: list[dict[str, Any]], title: str = "") -> str:
    """OPeRA-shaped observation: named controls only."""
    chunks = ["<html><head><title>", html_lib.escape(title or ""), "</title></head><body>"]
    for item in stamped:
        name = item.get("name") or ""
        if not name:
            continue
        text = html_lib.escape((item.get("text") or "")[:80])
        tag = "input" if "search_input" in name else "div"
        chunks.append(f'<{tag} name="{html_lib.escape(name, quote=True)}">{text}</{tag}>')
    chunks.append("</body></html>")
    return "".join(chunks)


def coverage_families(names: list[str]) -> dict[str, bool]:
    """Which shopper-widget families are present in a stamped name list."""
    nset = set(names)
    return {
        "search": "nav_bar.search_input" in nset,
        "cart_nav": any("cart" in n for n in nset),
        "results": any(n.startswith("product_") for n in nset),
        "sort": "search_results.sort" in nset,
        "pagination": any(n.startswith("pagination.") for n in nset),
        "filters": any(n.startswith("refinements") for n in nset),
        "add_to_cart": any("add_to_cart" in n for n in nset),
        "buy_now": any("buy_now" in n for n in nset),
        "variants": any(n.startswith("product_options") for n in nset),
        "reviews": any(n.startswith("reviews") for n in nset),
        "checkout": "check_out" in nset or "proceed_to_checkout" in nset,
        "quantity": any("quantity" in n for n in nset),
    }
