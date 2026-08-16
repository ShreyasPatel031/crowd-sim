"""Playwright shopper loop: open a public URL, click/type a few times, screenshot each step."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from playwright.async_api import TimeoutError as PlaywrightTimeout
from playwright.async_api import async_playwright

from opera_repro.actions import Action, parse_action
from opera_repro.prompts import SYSTEM_PROMPT, format_user_prompt
from simulator.llm import MODEL, complete_json, llm_available
from simulator.opera_bridge import apply_opera_action, stamp_opera_names
from simulator.opera_mapper import simplified_html

SKIP_TEXT = (
    "log in",
    "login",
    "sign in",
    "password",
    "cookie settings",
    "privacy",
    "terms",
)
MAX_ELEMENTS = 30
NAV_SKIP = (
    "all", "health ai", "amazon haul", "medical care", "amazon basics",
    "best sellers", "prime", "new releases", "today's deals", "books",
    "groceries", "whole foods", "gift cards", "sell", "returns & orders",
    "account & lists", "en", "hello, sign in",
)

# Numbered bounding boxes (Set-of-Marks). The screenshot is what the model sees.
MARK_JS = """
() => {
  document.getElementById('sim-marks')?.remove();
  const skip = /log in|login|sign in|password|cookie settings|privacy|terms/i;
  const sel = [
    'a', 'button', 'input', 'textarea', 'select',
    '[role="button"]', '[role="link"]',
    '[data-asin]', '#add-to-cart-button',
  ].join(',');
  const vh = window.innerHeight, vw = window.innerWidth;
  const seen = new Set();
  const nodes = [...document.querySelectorAll(sel)].filter((el) => {
    if (seen.has(el)) return false;
    const r = el.getBoundingClientRect();
    const style = window.getComputedStyle(el);
    if (r.width < 8 || r.height < 8) return false;
    if (r.bottom < 0 || r.top > vh || r.right < 0 || r.left > vw) return false;
    if (style.visibility === 'hidden' || style.display === 'none' || style.opacity === '0') return false;
    const text = (el.innerText || el.value || el.getAttribute('aria-label')
      || el.getAttribute('placeholder') || '').trim();
    if (skip.test(text)) return false;
    seen.add(el);
    return true;
  });
  nodes.sort((a, b) => {
    const ra = a.getBoundingClientRect(), rb = b.getBoundingClientRect();
    return ra.top - rb.top || ra.left - rb.left;
  });
  const picked = nodes.slice(0, 30);
  const layer = document.createElement('div');
  layer.id = 'sim-marks';
  layer.style.cssText = 'position:fixed;inset:0;z-index:2147483647;pointer-events:none;';
  const marks = picked.map((el, i) => {
    el.setAttribute('data-sim-id', String(i));
    const r = el.getBoundingClientRect();
    const box = document.createElement('div');
    box.style.cssText = [
      'position:absolute',
      `left:${Math.max(0, r.x)}px`,
      `top:${Math.max(0, r.y)}px`,
      `width:${Math.min(r.width, vw - r.x)}px`,
      `height:${Math.min(r.height, vh - r.y)}px`,
      'border:2px solid #FFD400',
      'box-sizing:border-box',
    ].join(';');
    const badge = document.createElement('div');
    badge.textContent = String(i);
    badge.style.cssText = 'position:absolute;left:0;top:0;background:#FFD400;color:#111;font:bold 11px/14px ui-sans-serif,system-ui;padding:0 4px;';
    box.appendChild(badge);
    layer.appendChild(box);
    return {
      id: i,
      tag: el.tagName.toLowerCase(),
      type: el.getAttribute('type') || '',
      text: (el.innerText || el.value || el.getAttribute('aria-label')
        || el.getAttribute('placeholder') || '').trim().slice(0, 80),
      href: el.href || '',
      bbox: {
        x: Math.round(r.x), y: Math.round(r.y),
        w: Math.round(r.width), h: Math.round(r.height),
      },
    };
  });
  document.documentElement.appendChild(layer);
  return marks;
}
"""


def validate_public_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw.startswith(("http://", "https://")):
        raw = "https://" + raw
    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("URL must be a public http(s) link")
    host = parsed.hostname or ""
    if host in ("localhost", "127.0.0.1") or host.endswith(".local"):
        raise ValueError("Local URLs are not allowed")
    return raw


async def run_simulation(
    url: str,
    intent: str,
    out_dir: Path,
    max_steps: int = 30,
    headed: bool = False,
    opera_dummy_examples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if opera_dummy_examples is not None:
        from simulator.opera_bridge import run_opera_dummy

        return await run_opera_dummy(opera_dummy_examples[:max_steps], out_dir, headed=headed)
    url = validate_public_url(url)
    intent = (intent or "Browse as a first-time visitor and find something you might buy.").strip()
    out_dir.mkdir(parents=True, exist_ok=True)
    _touch_progress(
        out_dir,
        status="running",
        message="Starting browser…",
        url=url,
        intent=intent,
        max_steps=max_steps,
        steps=[],
    )

    steps: list[dict[str, Any]] = []
    async with async_playwright() as pw:
        browser, page, backend = await _open_browser(pw, headed)
        try:
            _touch_progress(out_dir, message="Opening page…")
            await page.goto(url, wait_until="domcontentloaded", timeout=45000)
            await page.wait_for_timeout(1800)
            await _dismiss_banners(page)
            try:
                await page.wait_for_selector(
                    '#twotabsearchtextbox, #nav-cart, #add-to-cart-button',
                    timeout=8000,
                )
            except PlaywrightTimeout:
                pass
            _touch_progress(out_dir, message="Page loaded — choosing next action…")

            for step_i in range(max_steps):
                await page.wait_for_timeout(400)
                opera_stamped = await stamp_opera_names(page)
                elements = await _mark_elements(page)
                shot_name = f"step_{step_i + 1:02d}.png"
                shot = await page.screenshot(type="png")
                (out_dir / shot_name).write_bytes(shot)
                title = await page.title()
                current_url = page.url
                remaining = max_steps - step_i - 1
                opera_html = simplified_html(opera_stamped, title)
                decision = _decide(
                    intent,
                    title,
                    current_url,
                    elements,
                    steps,
                    remaining,
                    shot,
                    opera_stamped=opera_stamped,
                    opera_html=opera_html,
                )
                record = {
                    "step": step_i + 1,
                    "url": current_url,
                    "title": title,
                    "screenshot": shot_name,
                    "decision": decision,
                    "elements_shown": len(elements),
                    "opera_names": [item.get("name") for item in opera_stamped],
                    "opera_html": opera_html,
                    "browser": backend,
                }
                steps.append(record)
                target = decision.get("target_text") or decision.get("text") or ""
                _touch_progress(
                    out_dir,
                    message=f"Step {step_i + 1}: {decision.get('action')} {target}".strip(),
                    steps=steps,
                    current_url=current_url,
                    current_title=title,
                )
                print(
                    f"  step {step_i + 1}: {decision.get('action')} "
                    f"{target} "
                    f"— {current_url[:80]}",
                    flush=True,
                )
                if headed:
                    await page.wait_for_timeout(1200)
                await _clear_marks(page)
                if decision.get("action") == "done":
                    break
                await _apply(page, decision)
            if headed:
                await page.wait_for_timeout(5000)
        finally:
            await browser.close()

    _touch_progress(out_dir, message="Summarizing walkthrough…")
    report = _write_report(intent, url, steps, out_dir)
    report["browser"] = steps[0]["browser"] if steps else "unknown"
    _touch_progress(out_dir, status="done", message="Complete", report_ready=True)
    return report


def browserbase_configured() -> bool:
    return bool(os.environ.get("BROWSERBASE_API_KEY") and os.environ.get("BROWSERBASE_PROJECT_ID"))


async def _open_browser(pw, headed: bool):
    if browserbase_configured():
        from browserbase import Browserbase

        bb = Browserbase(api_key=os.environ["BROWSERBASE_API_KEY"])
        session = bb.sessions.create(project_id=os.environ["BROWSERBASE_PROJECT_ID"])
        connect_url = getattr(session, "connect_url", None) or session.connectUrl
        browser = await pw.chromium.connect_over_cdp(connect_url)
        context = browser.contexts[0] if browser.contexts else await browser.new_context()
        page = context.pages[0] if context.pages else await context.new_page()
        return browser, page, "browserbase"
    browser = await pw.chromium.launch(
        headless=not headed,
        slow_mo=400 if headed else 0,
        args=["--disable-blink-features=AutomationControlled"],
    )
    page = await browser.new_page(
        viewport={"width": 1280, "height": 800},
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
    )
    return browser, page, "playwright-local"


async def _mark_elements(page) -> list[dict[str, Any]]:
    try:
        raw = await page.evaluate(MARK_JS)
    except Exception:
        return []
    cleaned = []
    for item in raw[:MAX_ELEMENTS]:
        text = (item.get("text") or "").strip()
        if text.lower() in NAV_SKIP:
            continue
        if item.get("type") == "password":
            continue
        cleaned.append(item)
    return cleaned


async def _clear_marks(page) -> None:
    try:
        await page.evaluate("() => document.getElementById('sim-marks')?.remove()")
    except Exception:
        pass


async def _dismiss_banners(page) -> None:
    for label in ("Continue shopping", "Accept", "Accept all", "I agree", "Got it", "OK"):
        try:
            loc = page.get_by_role("button", name=re.compile(label, re.I))
            if await loc.count():
                await loc.first.click(timeout=1200)
                await page.wait_for_timeout(800)
                return
        except Exception:
            continue


LIVE_OPERA_SYSTEM = (
    SYSTEM_PROMPT
    + """

This is a LIVE shopping session. A screenshot is attached; read prices and
product photos from the image. Named controls are in the HTML (name="...").
Copy `name` exactly from that HTML. Do not invent names. Do not log in.
Honor the shopper intent in the user message. Do not stop on search results.
Use {"type":"terminate"} only after a matching product decision or no useful control remains.
"""
)


def _decide(
    intent: str,
    title: str,
    url: str,
    elements: list[dict[str, Any]],
    history: list[dict[str, Any]],
    remaining: int = 0,
    image_png: bytes | None = None,
    opera_stamped: list[dict[str, Any]] | None = None,
    opera_html: str = "",
) -> dict[str, Any]:
    stamped = opera_stamped or []
    if llm_available():
        try:
            if stamped:
                decision = _llm_decide_opera(
                    intent, title, url, stamped, opera_html, history, remaining, image_png, elements
                )
            else:
                decision = _llm_decide(intent, title, url, elements, history, remaining, image_png)
        except Exception as exc:
            decision = _heuristic_decide(intent, elements, history, stamped)
            decision["reason"] = f"LLM failed ({exc}); used heuristic. " + decision.get("reason", "")
    else:
        decision = _heuristic_decide(intent, elements, history, stamped)
    return _block_premature_stop(decision, intent, title, url, remaining, elements, history, stamped)


def _legal_opera_names(stamped: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for item in stamped:
        if item.get("name"):
            names.add(item["name"])
        for alias in item.get("aliases") or []:
            names.add(alias)
    return names


def _llm_decide_opera(
    intent,
    title,
    url,
    stamped,
    opera_html,
    history,
    remaining: int,
    image_png: bytes | None,
    elements: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    pairs: list[tuple[str, Action]] = []
    for row in history[-4:]:
        html = row.get("opera_html") or ""
        dec = row.get("decision") or {}
        opera = dec.get("opera")
        if opera:
            payload = {"type": opera.get("type") or "click"}
            if opera.get("name"):
                payload["name"] = opera["name"]
            if opera.get("text"):
                payload["text"] = opera["text"]
            action = Action(**payload)
        elif dec.get("name"):
            kind = "type_and_submit" if dec.get("action") == "type" else (
                "terminate" if dec.get("action") in ("done", "terminate") else "click"
            )
            action = Action(type=kind, name=dec.get("name") or None, text=dec.get("text") or None)
        else:
            continue
        pairs.append((html, action))
    user = (
        f"Shopper intent: {intent}\n"
        f"URL: {url}\n"
        f"steps_left_after_this: {remaining}\n\n"
        + format_user_prompt(pairs, opera_html or simplified_html(stamped, title))
    )
    parsed = complete_json(LIVE_OPERA_SYSTEM, user, image_png=image_png)
    legal = _legal_opera_names(stamped)
    action = parse_action(json.dumps(parsed)) if parsed else None
    if action is None or (action.type != "terminate" and action.name not in legal):
        # Old screenshot schema fallback
        if str(parsed.get("action") or "").lower() in ("click", "type") and parsed.get("id") is not None:
            return _llm_decide(intent, title, url, elements or [], history, remaining, image_png)
        return _heuristic_decide(intent, elements or [], history, stamped)
    out = {
        "action": "done" if action.type == "terminate" else ("type" if action.type == "type_and_submit" else "click"),
        "name": action.name or "",
        "text": action.text or "",
        "reason": str(parsed.get("reason") or ""),
        "would_prefer": str(parsed.get("would_prefer") or ""),
        "opera": action.to_dict(),
        "brain": "opera_live",
        "target_text": next((i.get("text") or "" for i in stamped if i.get("name") == action.name), action.name or ""),
    }
    return out


def _llm_decide(intent, title, url, elements, history, remaining: int, image_png: bytes | None) -> dict[str, Any]:
    hist = [
        {
            "step": s["step"],
            "did": s["decision"].get("action"),
            "target": s["decision"].get("target_text") or s["decision"].get("text") or s["decision"].get("id"),
            "reason": s["decision"].get("reason"),
        }
        for s in history
    ]
    marks = [
        {
            "id": e["id"],
            "label": (e.get("text") or e.get("tag") or "")[:60],
            "bbox": e.get("bbox"),
        }
        for e in elements
    ]
    user = json.dumps(
        {
            "intent": intent,
            "page_title": title,
            "url": url,
            "steps_left_after_this": remaining,
            "history": hist,
            "marks": marks,
        },
        ensure_ascii=False,
    )
    system = (
        "You are a shopper looking at a screenshot of a real website. "
        "Yellow numbered boxes are click targets (bounding boxes). "
        "Read prices, titles, and product photos FROM THE IMAGE, not from guesses. "
        "Pick ONE next action. If the current product does not match, go back or click a different listing. "
        "Do not log in. Do not stop on search results. "
        "Use done only after you have a matching product and decided add-to-cart, or steps_left_after_this is 0. "
        "Return JSON only: "
        '{"action":"click","id":0,"reason":"..."} or '
        '{"action":"type","id":0,"text":"...","reason":"..."} or '
        '{"action":"back","reason":"..."} or '
        '{"action":"done","reason":"...","would_prefer":"..."}'
    )
    parsed = complete_json(system, user, image_png=image_png)
    action = str(parsed.get("action") or "done").lower()
    if action not in ("click", "type", "back", "done"):
        action = "done"
    out = {
        "action": action,
        "id": parsed.get("id"),
        "text": parsed.get("text") or "",
        "reason": parsed.get("reason") or "",
        "would_prefer": parsed.get("would_prefer") or "",
    }
    if action in ("click", "type") and not _valid_id(out.get("id"), elements):
        return _heuristic_decide(intent, elements, history)
    if action in ("click", "type"):
        match = next((e for e in elements if e["id"] == int(out["id"])), {})
        out["target_text"] = match.get("text") or ""
        out["bbox"] = match.get("bbox")
    return out


_GIVE_UP = re.compile(
    r"go back|does not meet|doesn't meet|do not meet|not (actually )?(fleece|matching)|"
    r"look for (an )?another|wrong product|not the right|failed to meet",
    re.I,
)
_INTENT_STOP = {"find", "with", "that", "this", "from", "under", "would", "could", "should", "tell", "whether", "something", "might", "buy"}


def _block_premature_stop(
    decision: dict[str, Any],
    intent: str,
    title: str,
    url: str,
    remaining: int,
    elements: list[dict[str, Any]],
    history: list[dict[str, Any]],
    opera_stamped: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if decision.get("action") != "done" or remaining <= 0:
        return decision
    blob = f"{title} {decision.get('reason','')} {decision.get('would_prefer','')}".lower()
    giving_up = bool(_GIVE_UP.search(blob))
    on_product = bool(re.search(r"/dp/|/gp/product|/gp/aw/d/", url))
    keys = [w for w in re.findall(r"[a-z0-9]+", intent.lower()) if len(w) > 3 and w not in _INTENT_STOP]
    matched = sum(1 for w in keys if w in blob)
    if giving_up or not on_product or matched < max(1, len(keys) // 2):
        if on_product:
            decision["action"] = "back"
            decision["target_text"] = "Back"
            decision["forced_continue"] = True
            decision["reason"] = (
                (decision.get("reason") or "")
                + " (continued: first mismatch is not a stop; keep looking)"
            ).strip()
            return decision
        alt = _heuristic_decide(intent, elements, history, opera_stamped)
        alt["forced_continue"] = True
        alt["reason"] = (
            (decision.get("reason") or "")
            + " (continued: first mismatch is not a stop; keep looking)"
        ).strip()
        return alt
    return decision


def _heuristic_decide(intent, elements, history, opera_stamped=None) -> dict[str, Any]:
    stamped = opera_stamped or []
    names = [item.get("name") or "" for item in stamped]
    typed_already = any(
        s["decision"].get("action") == "type"
        or (s["decision"].get("opera") or {}).get("type") == "type_and_submit"
        for s in history
    )
    intent_words = [w for w in re.findall(r"[a-z0-9]+", intent.lower()) if len(w) > 2]
    if "nav_bar.search_input" in names and intent_words and not typed_already:
        text = " ".join(intent_words[:8])
        return {
            "action": "type",
            "name": "nav_bar.search_input",
            "text": text,
            "target_text": "search",
            "reason": "Type the intent into search.",
            "opera": {"type": "type_and_submit", "name": "nav_bar.search_input", "text": text},
            "brain": "opera_heuristic",
        }
    clicked = {s["decision"].get("name") for s in history}
    for name in names:
        if name.startswith("product_") and name not in clicked:
            return {
                "action": "click",
                "name": name,
                "target_text": name,
                "reason": "Open a search result.",
                "opera": {"type": "click", "name": name},
                "brain": "opera_heuristic",
            }
    for name in names:
        if "add_to_cart" in name:
            return {
                "action": "click",
                "name": name,
                "target_text": name,
                "reason": "Add the current product to cart.",
                "opera": {"type": "click", "name": name},
                "brain": "opera_heuristic",
            }
    if len(history) >= 5:
        return {
            "action": "done",
            "reason": "Reached step limit (heuristic stop).",
            "would_prefer": "Open the most relevant product and add it to cart if the price looks fair.",
        }
    # Screenshot-id fallback when nothing was stamped.
    search = next(
        (
            e
            for e in elements
            if e.get("tag") in ("input", "textarea")
            and e.get("type") in ("", "text", "search")
        ),
        None,
    )
    if search and intent_words and not typed_already:
        return {
            "action": "type",
            "id": search["id"],
            "text": " ".join(intent_words[:6]),
            "target_text": search.get("text") or "search",
            "reason": "Type the intent into search.",
        }

    def score(el: dict[str, Any]) -> int:
        blob = f"{el.get('text','')} {el.get('href','')}".lower()
        s = 0
        for w in intent_words:
            if w in blob:
                s += 3
        for w in ("product", "shop", "buy", "cart", "view", "book", "item"):
            if w in blob:
                s += 1
        return s

    ranked = sorted(elements, key=score, reverse=True)
    clicked_ids = {s["decision"].get("id") for s in history}
    visited = " ".join(s.get("url") or "" for s in history)
    seen_text = {s["decision"].get("target_text") for s in history}
    for el in ranked:
        if el["id"] in clicked_ids:
            continue
        if el.get("target_text") in seen_text or el.get("text") in seen_text:
            continue
        if el.get("href") and el["href"].split("?")[0] in visited:
            continue
        if el.get("tag") == "input":
            continue
        if score(el) <= 0 and not el.get("href"):
            continue
        return {
            "action": "click",
            "id": el["id"],
            "target_text": el.get("text") or el.get("href") or "",
            "reason": "Clicked the most intent-related visible control.",
        }
    return {
        "action": "done",
        "reason": "No obvious next control.",
        "would_prefer": "Scan featured products and open the first one that matches the intent.",
    }


def _valid_id(value: Any, elements: list[dict[str, Any]]) -> bool:
    try:
        idx = int(value)
    except (TypeError, ValueError):
        return False
    return any(el["id"] == idx for el in elements)


async def _apply(page, decision: dict[str, Any]) -> None:
    action = decision.get("action")
    if action == "back":
        try:
            await page.go_back(wait_until="domcontentloaded", timeout=15000)
            await page.wait_for_timeout(1400)
        except Exception as exc:
            decision["reason"] = (decision.get("reason") or "") + f" ({exc})"
        return
    if decision.get("name") or (decision.get("opera") or {}).get("name"):
        applied = await apply_opera_action(page, decision)
        if not applied.get("ok"):
            decision["reason"] = (decision.get("reason") or "") + f" ({applied.get('detail')})"
        await page.wait_for_timeout(1400)
        return
    try:
        idx = int(decision.get("id"))
    except (TypeError, ValueError):
        return
    loc = page.locator(f"[data-sim-id='{idx}']").first
    try:
        if action == "type":
            await loc.click(timeout=4000)
            await loc.fill(str(decision.get("text") or ""))
            await loc.press("Enter")
            try:
                await page.wait_for_selector(
                    '[data-component-type="s-search-result"], [data-asin], #dp',
                    timeout=8000,
                )
            except PlaywrightTimeout:
                pass
        elif action == "click":
            await loc.click(timeout=4000)
        await page.wait_for_timeout(1400)
    except PlaywrightTimeout:
        decision["reason"] = (decision.get("reason") or "") + " (click timed out)"
    except Exception as exc:
        decision["reason"] = (decision.get("reason") or "") + f" ({exc})"


def _touch_progress(out_dir: Path, **fields: Any) -> None:
    path = out_dir / "progress.json"
    data: dict[str, Any] = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
    data.update(fields)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def mark_progress_error(out_dir: Path, message: str) -> None:
    _touch_progress(out_dir, status="error", message=message, error=message)


def _write_report(intent: str, start_url: str, steps: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    last = steps[-1]["decision"] if steps else {}
    would = last.get("would_prefer") or last.get("reason") or "Keep browsing the most relevant products."
    if llm_available() and steps:
        try:
            would_pack = complete_json(
                "Summarize a shopping walkthrough. Return JSON only: "
                '{"summary":"...","would_prefer":"what this shopper would do next or buy",'
                '"friction":["..."],"products_noticed":["..."]}',
                json.dumps({"intent": intent, "steps": steps}, ensure_ascii=False)[:12000],
            )
            if would_pack.get("would_prefer"):
                would = would_pack["would_prefer"]
            summary = would_pack.get("summary") or _default_summary(intent, steps)
            friction = would_pack.get("friction") or []
            products = would_pack.get("products_noticed") or []
        except Exception:
            summary = _default_summary(intent, steps)
            friction, products = [], []
    else:
        summary = _default_summary(intent, steps)
        friction, products = [], []
        if not llm_available():
            friction = ["Gemini was unavailable — clicks used a keyword heuristic, not a full LLM agent."]

    report = {
        "start_url": start_url,
        "intent": intent,
        "llm": llm_available(),
        "model": MODEL,
        "browser": steps[0]["browser"] if steps else "unknown",
        "summary": summary,
        "would_prefer": would,
        "friction": friction,
        "products_noticed": products,
        "steps": steps,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _default_summary(intent: str, steps: list[dict[str, Any]]) -> str:
    n = len(steps)
    last_title = steps[-1]["title"] if steps else ""
    return f"Followed “{intent}” for {n} step(s). Last page: {last_title}."
