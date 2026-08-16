#!/usr/bin/env python3
"""Stamp a live Amazon search + PDP and report which shopper families are mapped."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from playwright.async_api import async_playwright

from simulator.agent import _dismiss_banners, _open_browser
from simulator.opera_bridge import stamp_opera_names
from simulator.opera_mapper import coverage_families

NEEDED = ("search", "cart_nav", "results", "pagination", "filters", "add_to_cart")


async def main() -> None:
    pages = []
    async with async_playwright() as pw:
        browser, page, _ = await _open_browser(pw, headed=False)
        try:
            await page.goto("https://www.amazon.com", wait_until="domcontentloaded", timeout=45000)
            await page.wait_for_timeout(1200)
            await _dismiss_banners(page)
            try:
                await page.wait_for_selector("#twotabsearchtextbox", timeout=8000)
            except Exception:
                pass
            box = page.locator("#twotabsearchtextbox")
            await box.fill("fleece bathrobe")
            await box.press("Enter")
            await page.wait_for_timeout(2500)
            search_names = [x["name"] for x in await stamp_opera_names(page)]
            pages.append({"label": "search", "url": page.url, "names": search_names, "coverage": coverage_families(search_names)})
            print("SEARCH", json.dumps(pages[-1]["coverage"], indent=2), "n=", len(search_names))
            print(" names sample", search_names[:25])

            link = page.locator('[data-opera-name="product_1"]').first
            if await link.count():
                await link.click()
                await page.wait_for_timeout(2500)
            prod_names = [x["name"] for x in await stamp_opera_names(page)]
            pages.append({"label": "product", "url": page.url, "names": prod_names, "coverage": coverage_families(prod_names)})
            print("PRODUCT", json.dumps(pages[-1]["coverage"], indent=2), "n=", len(prod_names))
            print(" names sample", prod_names[:30])
        finally:
            await browser.close()

    merged = coverage_families([n for p in pages for n in p["names"]])
    missing = [k for k in NEEDED if not merged.get(k)]
    out = ROOT / "data" / "sim_runs" / "mapper_coverage.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"pages": pages, "merged": merged, "missing_needed": missing}, indent=2), encoding="utf-8")
    print("MERGED", json.dumps(merged, indent=2))
    print("missing needed families", missing or "NONE")
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
