"""Parallel listing capture + one Gemini call per shopper×listing."""

from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path
from typing import Any

from simulator.verdict import parse_listing_eval

MODEL = "gemini-2.5-flash"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
MARK_JS = """
() => {
  document.getElementById('sim-marks')?.remove();
  const skip = /log in|login|sign in|password|cookie settings|privacy|terms/i;
  const sel = [
    'a', 'button', 'input', 'textarea', 'select',
    '[role="button"]', '[role="link"]', '[role="tab"]',
    '[data-asin]', '#add-to-cart-button', '#buy-now-button',
  ].join(',');
  const vh = window.innerHeight, vw = window.innerWidth;
  const skipRoot = (el) => el.closest('#navbar, #navbar-main, #nav-belt, #nav-main, #nav-flyout-anchor, #nav-progressive-subnav, #nav-subnav, header, #gw-card-layout');
  const productRoot = document.querySelector('#ppd, #dp-container, #centerCol, #dp') || document.body;
  const seen = new Set();
  const collect = (root) => [...root.querySelectorAll(sel)].filter((el) => {
    if (seen.has(el) || skipRoot(el)) return false;
    const r = el.getBoundingClientRect();
    const style = window.getComputedStyle(el);
    if (r.width < 16 || r.height < 16) return false;
    if (r.bottom < 0 || r.top > vh || r.right < 0 || r.left > vw) return false;
    if (r.width * r.height > vw * vh * 0.45) return false;
    if (el.querySelector(sel)) return false;
    if (style.visibility === 'hidden' || style.display === 'none' || style.opacity === '0') return false;
    const text = (el.innerText || el.value || el.getAttribute('aria-label')
      || el.getAttribute('placeholder') || '').trim();
    if (skip.test(text)) return false;
    seen.add(el);
    return true;
  });
  const nodes = collect(productRoot);
  if (nodes.length < 24) nodes.push(...collect(document.body));
  nodes.sort((a, b) => {
    const ra = a.getBoundingClientRect(), rb = b.getBoundingClientRect();
    return ra.top - rb.top || ra.left - rb.left;
  });
  const picked = nodes.slice(0, 24);
  const layer = document.createElement('div');
  layer.id = 'sim-marks';
  layer.style.cssText = 'position:fixed;inset:0;z-index:2147483647;pointer-events:none;';
  picked.forEach((el, i) => {
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
  });
  document.documentElement.appendChild(layer);
  return picked.length;
}
"""


def _persona_blurb(persona: dict[str, Any]) -> str:
    extra = []
    if persona.get("prime"):
        extra.append(f"Amazon Prime: {persona['prime']}")
    if persona.get("shop_frequency"):
        extra.append(f"Shops {persona['shop_frequency']}")
    if persona.get("income"):
        extra.append(f"Household income: {persona['income']}")
    priorities = ", ".join(persona.get("priorities") or []) or "value and reviews"
    avoids = ", ".join(persona.get("avoids") or []) or "unclear listings"
    return f"""You are an OPeRA shopper: {persona.get('label')}.
{persona.get('bio') or ''}
Budget: {persona.get('budget') or 'not specified'}.
You care about: {priorities}.
You avoid: {avoids}.
{'. '.join(extra)}
Shop as this real person. Do not break character."""


async def capture_listings(urls: list[str], out_dir: Path) -> list[dict[str, Any]]:
    from playwright.async_api import async_playwright

    out_dir.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as pw:
        launch_kwargs: dict[str, Any] = {
            "headless": True,
            "args": ["--disable-blink-features=AutomationControlled"],
        }
        try:
            browser = await pw.chromium.launch(channel="chrome", **launch_kwargs)
        except Exception:
            browser = await pw.chromium.launch(**launch_kwargs)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent=USER_AGENT,
            locale="en-US",
            extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
        )

        async def dismiss_interstitial(page, url: str) -> None:
            btn = page.get_by_role("button", name="Continue shopping")
            if await btn.count():
                try:
                    await btn.first.click(timeout=4000)
                    await page.wait_for_timeout(1200)
                except Exception:
                    pass
            if await page.locator("#productTitle").count():
                return
            try:
                await page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=30000,
                    referer="https://www.google.com/",
                )
                await page.wait_for_timeout(2500)
            except Exception:
                pass
            try:
                await page.wait_for_selector("#productTitle", timeout=8000)
            except Exception:
                pass

        async def one(index: int, url: str) -> dict[str, Any]:
            page = await context.new_page()
            title = ""
            text = ""
            marks = 0
            shot_name = f"listing_{index}.png"
            dest = out_dir / shot_name
            try:
                try:
                    await page.goto("https://www.google.com/", wait_until="domcontentloaded", timeout=15000)
                except Exception:
                    pass
                await page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=30000,
                    referer="https://www.google.com/",
                )
                await page.wait_for_timeout(2500)
                await dismiss_interstitial(page, url)
                if await page.locator("#productTitle").count():
                    title = (await page.locator("#productTitle").first.inner_text()).strip()
                else:
                    title = (await page.title()).strip()
                chunks: list[str] = []
                for sel in ("#productTitle", "#corePrice_feature_div", "#acrPopover", "#feature-bullets"):
                    if await page.locator(sel).count():
                        chunks.append((await page.locator(sel).first.inner_text()).strip())
                text = "\n".join(c for c in chunks if c)[:3500]
                if not text:
                    text = ((await page.inner_text("body")) or "")[:3500]
                try:
                    marks = int(await page.evaluate(MARK_JS) or 0)
                except Exception:
                    marks = 0
                await page.screenshot(path=str(dest), type="png", full_page=False)
            except Exception as exc:
                text = text or f"(page capture failed: {exc})"
                if not dest.exists():
                    try:
                        await page.screenshot(path=str(dest), type="png", full_page=False)
                    except Exception:
                        pass
            finally:
                await page.close()
            if not dest.exists() or dest.stat().st_size < 80_000:
                raise RuntimeError(f"Screenshot for {url} was empty or too small ({dest.stat().st_size if dest.exists() else 0} bytes)")
            if not _product_title_ok(title):
                raise RuntimeError(f"Did not reach product page for {url} (title={title!r})")
            return {
                "url": url,
                "title": title,
                "text": text,
                "screenshot": shot_name,
                "marks": marks,
                "index": index,
            }

        try:
            rows = await asyncio.gather(*[one(i, url) for i, url in enumerate(urls)])
        finally:
            await context.close()
            await browser.close()
    return list(rows)


def _product_title_ok(title: str) -> bool:
    t = (title or "").strip().lower()
    if not t:
        return False
    if t in {"amazon.com", "amazon.com. spend less. smile more."}:
        return False
    return True


def _gemini_json(prompt: str, image_path: Path | None) -> str:
    key = os.environ.get("GOOGLE_API_KEY") or ""
    if not key:
        raise RuntimeError("GOOGLE_API_KEY is missing")
    import urllib.request

    parts: list[dict[str, Any]] = [{"text": prompt}]
    if image_path and image_path.exists():
        parts.append(
            {
                "inline_data": {
                    "mime_type": "image/png",
                    "data": base64.b64encode(image_path.read_bytes()).decode("ascii"),
                }
            }
        )
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"},
    }
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    out_parts = (((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or [])
    return "".join(str(part.get("text") or "") for part in out_parts)


def _judge_sync(
    persona: dict[str, Any],
    listing: dict[str, Any],
    brief: str,
    role: str,
    out_dir: Path,
) -> dict[str, Any]:
    role_line = (
        "This is the LISTING TO EVALUATE (the seller's product)."
        if role == "product"
        else "This is a COMPETITOR listing. Judge it on its own, as this shopper."
    )
    prompt = f"""{_persona_blurb(persona)}

RESEARCH BRIEF: {brief}

{role_line}
URL: {listing.get('url')}
Visible title: {listing.get('title') or '(unknown)'}
Visible text:
{(listing.get('text') or '')[:3000]}

Look at the screenshot if provided. In one pass, decide whether YOU would buy THIS listing.

Return JSON only:
{{
  "listing_title": "short product name",
  "would_buy_this": "buy" | "maybe" | "no",
  "appeal": 0-100,
  "confidence": 0-100,
  "rationale": "2-4 sentences in this shopper's voice",
  "price_perception": "cheap | fair | expensive, plus a short why",
  "trust_concerns": ["..."],
  "conversion_blockers": ["..."]
}}
"""
    shot = listing.get("screenshot") or ""
    image_path = out_dir / shot if shot else None
    try:
        raw = _gemini_json(prompt, image_path)
        parsed = parse_listing_eval(raw)
    except Exception as exc:
        parsed = parse_listing_eval("")
        parsed["rationale"] = f"Could not judge this listing ({exc})."
        parsed["would_buy_this"] = "maybe"
    if not parsed.get("listing_title"):
        parsed["listing_title"] = listing.get("title") or ""
    parsed["url"] = listing.get("url")
    parsed["role"] = role
    parsed["source"] = "gemini"
    return parsed


async def judge_listing(
    persona: dict[str, Any],
    listing: dict[str, Any],
    brief: str,
    role: str,
    out_dir: Path,
) -> dict[str, Any]:
    return await asyncio.to_thread(_judge_sync, persona, listing, brief, role, out_dir)
