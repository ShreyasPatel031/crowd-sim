#!/usr/bin/env python3
"""Probe live Amazon: do OPeRA semantic name= attributes exist in the real DOM?"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from playwright.async_api import async_playwright

from simulator.agent import _dismiss_banners, _open_browser

OPERA_NAMES = [
    "nav_bar.search_input",
    "nav_bar.search_button",
    "nav_bar.cart",
    "go_to_cart",
    "add_to_cart",
    "add_to_cart.buy_now",
    "buy_now",
    "reviews",
    "product_4",
    "check_out",
    "proceed_to_checkout",
]

AMAZON_HINTS = [
    "#twotabsearchtextbox",
    "#nav-search-submit-button",
    "#nav-cart",
    "#add-to-cart-button",
    "#buy-now-button",
    "#acrCustomerReviewLink",
    "[name='field-keywords']",
    "[name='site-search']",
]

PROBE_JS = """
(names) => {
  const html = document.documentElement.innerHTML;
  const hits = {};
  for (const name of names) {
    const els = [...document.querySelectorAll('[name="' + name + '"]')];
    hits[name] = {
      selectorCount: els.length,
      inHtml: html.includes('name="' + name + '"') || html.includes("name='" + name + "'"),
      inHtmlLoose: html.includes(name),
    };
  }
  const liveNames = [...document.querySelectorAll('[name]')]
    .map((el) => el.getAttribute('name'))
    .filter(Boolean);
  const counts = {};
  for (const n of liveNames) counts[n] = (counts[n] || 0) + 1;
  const top = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 25);
  const hints = {};
  for (const sel of [
    '#twotabsearchtextbox', '#nav-search-submit-button', '#nav-cart',
    '#add-to-cart-button', '#buy-now-button', '#acrCustomerReviewLink',
    '[name="field-keywords"]',
  ]) {
    hints[sel] = document.querySelector(sel) ? true : false;
  }
  return {
    title: document.title,
    url: location.href,
    opera: hits,
    amazonHints: hints,
    liveNameSample: top,
    nNamedElements: liveNames.length,
  };
}
"""


def opera_names_from_dataset(limit: int = 200) -> list[str]:
    try:
        from datasets import load_dataset

        ds = load_dataset(
            "NEU-HAI/OPeRA",
            "filtered_action",
            split="test",
            cache_dir=str(ROOT / "data" / "cache"),
        )
        names: Counter[str] = Counter()
        for i, row in enumerate(ds):
            if i >= limit:
                break
            sid = (row.get("semantic_id") or "").strip()
            if sid:
                names[sid] += 1
            html = row.get("simplified_html") or ""
            for match in re.findall(r'name="([^"]+)"', html):
                names[match] += 1
        return [n for n, _ in names.most_common(40)]
    except Exception as exc:
        print(f"dataset names unavailable ({exc}); using builtin list", flush=True)
        return OPERA_NAMES


async def probe_page(page, label: str, names: list[str]) -> dict:
    await page.wait_for_timeout(1500)
    await _dismiss_banners(page)
    data = await page.evaluate(PROBE_JS, names)
    data["label"] = label
    present = [n for n, info in data["opera"].items() if info["selectorCount"] or info["inHtml"]]
    loose = [n for n, info in data["opera"].items() if info["inHtmlLoose"] and n not in present]
    print(f"\n=== {label} ===", flush=True)
    print(f"  url: {data['url'][:100]}", flush=True)
    print(f"  title: {data['title'][:80]}", flush=True)
    print(f"  OPeRA name= exact hits: {present or 'NONE'}", flush=True)
    print(f"  OPeRA string somewhere in HTML: {loose[:8] or 'NONE'}", flush=True)
    print(f"  Amazon native locators: { {k: v for k, v in data['amazonHints'].items() if v} }", flush=True)
    print(f"  live [name] count: {data['nNamedElements']}; top: {data['liveNameSample'][:8]}", flush=True)
    return data


async def main() -> None:
    names = list(dict.fromkeys(OPERA_NAMES + opera_names_from_dataset()))
    print(f"checking {len(names)} OPeRA names against live Amazon", flush=True)
    out_dir = ROOT / "data" / "sim_runs" / "amazon_opera_name_probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as pw:
        browser, page, backend = await _open_browser(pw, headed=False)
        pages = []
        try:
            await page.goto("https://www.amazon.com", wait_until="domcontentloaded", timeout=45000)
            pages.append(await probe_page(page, "home", names))
            search = page.locator("#twotabsearchtextbox")
            if await search.count():
                await search.fill("fleece bathrobe")
                await search.press("Enter")
                await page.wait_for_timeout(2500)
                pages.append(await probe_page(page, "search", names))
                result = page.locator('[data-component-type="s-search-result"] h2 a').first
                if await result.count():
                    await result.click()
                    await page.wait_for_timeout(2500)
                    pages.append(await probe_page(page, "product", names))
            else:
                print("WARNING: #twotabsearchtextbox missing — Amazon may have blocked/captcha", flush=True)
        finally:
            await browser.close()

    exact_any = sorted(
        {
            n
            for p in pages
            for n, info in p["opera"].items()
            if info["selectorCount"] or info["inHtml"]
        }
    )
    report = {
        "backend": backend,
        "n_names_checked": len(names),
        "names_checked": names,
        "exact_opera_name_hits_any_page": exact_any,
        "pages": pages,
    }
    path = out_dir / "report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nexact OPeRA name= on any live page: {exact_any or 'NONE'}", flush=True)
    print(f"Wrote {path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
