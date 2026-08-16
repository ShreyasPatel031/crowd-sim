"""OPeRA-format adapter for the live Playwright loop.

Dummy brain: given converted simplified HTML, return the gold next action.
Hands: execute click/type_and_submit/terminate by HTML name= (not data-sim-id).

This does not call Amazon or an LLM. It checks whether the live agent can
consume the same observation/action schema as train/eval.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from playwright.async_api import TimeoutError as PlaywrightTimeout
from playwright.async_api import async_playwright

from opera_repro.actions import Action, actions_equal, parse_action
from opera_repro.html_utils import named_targets
from simulator.opera_mapper import STAMP_JS, opera_locator, simplified_html


def observation_html(example: dict[str, Any]) -> str:
    text = example.get("prompt_text") or ""
    marker = "Current observation:\n"
    idx = text.find(marker)
    if idx < 0:
        return ""
    rest = text[idx + len(marker) :]
    end = rest.find("\n\nNext action JSON:")
    return (rest[:end] if end >= 0 else rest).strip()


def dummy_policy(html: str, gold: dict[str, Any]) -> dict[str, Any]:
    """Oracle policy: emit gold if the named target is on the page."""
    action = Action(**gold) if isinstance(gold, dict) else gold
    names = named_targets(html)
    on_page = action.type == "terminate" or bool(action.name and action.name in names)
    decision = {
        "action": "done" if action.type == "terminate" else ("type" if action.type == "type_and_submit" else "click"),
        "name": action.name or "",
        "text": action.text or "",
        "reason": "dummy oracle: gold action from converted OPeRA example",
        "would_prefer": "",
        "opera": action.to_dict(),
        "gold_on_page": on_page,
        "named_targets": names,
        "brain": "opera_dummy",
    }
    if not on_page:
        decision["reason"] += f" (name {action.name!r} missing from HTML)"
    return decision


async def apply_opera_action(page, decision: dict[str, Any]) -> dict[str, Any]:
    opera = decision.get("opera") or {}
    kind = opera.get("type") or decision.get("action")
    if kind in ("terminate", "done"):
        return {"ok": True, "detail": "terminate — no DOM write"}

    name = opera.get("name") or decision.get("name") or ""
    if not name:
        return {"ok": False, "detail": "missing name"}

    loc = page.locator(opera_locator(name)).first
    try:
        count = await loc.count()
    except Exception as exc:
        return {"ok": False, "detail": f"locator failed: {exc}"}
    if count == 0:
        return {"ok": False, "detail": f"no element [name={name}]"}

    try:
        if kind in ("type_and_submit", "type"):
            await loc.click(timeout=4000)
            await loc.fill(str(decision.get("text") or opera.get("text") or ""))
            await loc.press("Enter")
            filled = await loc.input_value()
            return {"ok": True, "detail": f"typed {filled!r} into {name}"}
        await loc.click(timeout=4000)
        return {"ok": True, "detail": f"clicked {name}"}
    except PlaywrightTimeout:
        return {"ok": False, "detail": f"timeout on {name}"}
    except Exception as exc:
        return {"ok": False, "detail": str(exc)}


async def run_opera_dummy(
    examples: list[dict[str, Any]],
    out_dir: Path,
    headed: bool = False,
) -> dict[str, Any]:
    """Replay converted examples: set_content(HTML) → dummy gold → click/type by name."""
    out_dir.mkdir(parents=True, exist_ok=True)
    steps: list[dict[str, Any]] = []
    n_ok = 0

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=not headed, slow_mo=300 if headed else 0)
        page = await browser.new_page(viewport={"width": 1280, "height": 800})
        try:
            for i, example in enumerate(examples, start=1):
                html = observation_html(example)
                gold = example["gold_action"]
                await page.set_content(html or "<html><body></body></html>", wait_until="domcontentloaded")
                decision = dummy_policy(html, gold)
                applied = await apply_opera_action(page, decision)
                decision["applied"] = applied
                if applied.get("ok"):
                    n_ok += 1
                parsed = parse_action(json.dumps(decision["opera"]))
                hit = actions_equal(parsed, Action(**gold))
                shot_name = f"step_{i:02d}.png"
                (out_dir / shot_name).write_bytes(await page.screenshot(type="png"))
                steps.append(
                    {
                        "step": i,
                        "session_id": example.get("session_id"),
                        "url": "opera-dummy://converted-html",
                        "title": await page.title(),
                        "screenshot": shot_name,
                        "decision": decision,
                        "gold_action": gold,
                        "exact_match": hit,
                        "applied_ok": bool(applied.get("ok")),
                        "brain": "opera_dummy",
                        "browser": "playwright-local",
                    }
                )
                print(
                    f"  step {i}: gold={gold} applied={applied.get('ok')} "
                    f"{applied.get('detail')} on_page={decision['gold_on_page']}",
                    flush=True,
                )
        finally:
            await browser.close()

    report = {
        "start_url": "opera-dummy://converted-html",
        "intent": "dummy oracle: execute gold OPeRA actions on converted HTML",
        "llm": False,
        "model": "opera_dummy",
        "brain": "opera_dummy",
        "browser": "playwright-local",
        "summary": (
            f"Dummy OPeRA adapter executed {n_ok}/{len(steps)} gold actions "
            "on converted simplified HTML via Playwright name= locators."
        ),
        "would_prefer": "Replace dummy_policy with the trained next-action model.",
        "friction": [] if n_ok == len(steps) else ["One or more gold names were missing or failed to click."],
        "products_noticed": [],
        "n_applied_ok": n_ok,
        "n_steps": len(steps),
        "all_applied": n_ok == len(steps) and len(steps) > 0,
        "steps": steps,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


async def stamp_opera_names(page) -> list[dict[str, Any]]:
    """Stamp data-opera-name on the live page. Returns assigned controls."""
    try:
        raw = await page.evaluate(STAMP_JS)
    except Exception:
        return []
    return list(raw or [])
