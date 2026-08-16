#!/usr/bin/env python3
"""Stamp OPeRA names on live Amazon and execute a few gold-style actions by name."""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from playwright.async_api import async_playwright

from simulator.agent import _dismiss_banners, _open_browser
from simulator.opera_bridge import apply_opera_action, stamp_opera_names
from simulator.opera_mapper import simplified_html


async def snapshot(page, label: str, out_dir: Path) -> dict:
    stamped = await stamp_opera_names(page)
    title = await page.title()
    (out_dir / f"{label}.png").write_bytes(await page.screenshot(type="png"))
    rec = {
        "label": label,
        "url": page.url,
        "title": title,
        "stamped": [item["name"] for item in stamped],
        "opera_html": simplified_html(stamped, title),
    }
    print(f"\n=== {label} ===", flush=True)
    print(f"  {rec['url'][:90]}", flush=True)
    print(f"  stamped: {rec['stamped']}", flush=True)
    return rec


async def act(page, gold: dict) -> dict:
    decision = {
        "action": "type" if gold["type"] == "type_and_submit" else gold["type"],
        "name": gold.get("name") or "",
        "text": gold.get("text") or "",
        "opera": gold,
    }
    applied = await apply_opera_action(page, decision)
    print(f"  apply {gold} → {applied}", flush=True)
    await page.wait_for_timeout(2200)
    return applied


async def main() -> None:
    run_id = uuid.uuid4().hex[:10]
    out_dir = ROOT / "data" / "sim_runs" / f"mapper_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = []
    actions = []

    async with async_playwright() as pw:
        browser, page, backend = await _open_browser(pw, headed=False)
        try:
            await page.goto("https://www.amazon.com", wait_until="domcontentloaded", timeout=45000)
            await page.wait_for_timeout(1500)
            await _dismiss_banners(page)
            try:
                await page.wait_for_selector("#twotabsearchtextbox", timeout=8000)
            except Exception:
                pass
            pages.append(await snapshot(page, "home", out_dir))

            typed = await act(
                page,
                {
                    "type": "type_and_submit",
                    "name": "nav_bar.search_input",
                    "text": "fleece bathrobe",
                },
            )
            actions.append({"gold": "type_and_submit(nav_bar.search_input)", **typed})
            await page.wait_for_timeout(1500)
            pages.append(await snapshot(page, "search", out_dir))

            clicked = await act(page, {"type": "click", "name": "product_1"})
            actions.append({"gold": "click(product_1)", **clicked})
            await page.wait_for_timeout(2000)
            pages.append(await snapshot(page, "product", out_dir))

            cart = await act(page, {"type": "click", "name": "add_to_cart"})
            actions.append({"gold": "click(add_to_cart) alias of buybox.purchase_form.add_to_cart", **cart})
        finally:
            await browser.close()

    report = {
        "backend": backend,
        "pages": [{k: p[k] for k in ("label", "url", "title", "stamped")} for p in pages],
        "actions": actions,
        "all_applied": all(a.get("ok") for a in actions),
    }
    path = out_dir / "report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"id": run_id, "all_applied": report["all_applied"], "actions": actions}, indent=2))
    print(f"Report: {path}")


if __name__ == "__main__":
    asyncio.run(main())
