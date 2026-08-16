"""OPeRA shopper personas and tasks for panel runs."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data" / "opera_catalog.json"


@lru_cache(maxsize=1)
def load_catalog() -> dict[str, Any]:
    if not CATALOG_PATH.exists():
        return {"personas": [], "tasks": []}
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def all_personas() -> list[dict[str, Any]]:
    return list(load_catalog().get("personas") or [])


def all_tasks() -> list[dict[str, Any]]:
    return list(load_catalog().get("tasks") or [])


PERSONAS = all_personas()
PERSONA_BY_ID = {p["id"]: p for p in PERSONAS}
TASK_BY_ID = {t["id"]: t for t in all_tasks()}
DEFAULT_PERSONA_IDS = [p["id"] for p in PERSONAS[:1]]


def get_personas(ids: list[str] | None) -> list[dict[str, Any]]:
    people = all_personas()
    by_id = {p["id"]: p for p in people}
    defaults = [p["id"] for p in people[:1]]
    chosen = ids or defaults
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pid in chosen:
        persona = by_id.get((pid or "").strip())
        if persona and persona["id"] not in seen:
            out.append(persona)
            seen.add(persona["id"])
    if out:
        return out
    return [by_id[i] for i in defaults if i in by_id]


def get_task(task_id: str | None) -> dict[str, Any] | None:
    if not task_id:
        return None
    return TASK_BY_ID.get(task_id) or next((t for t in all_tasks() if t["id"] == task_id), None)


def shopper_task(
    persona: dict[str, Any],
    product_url: str,
    competitor_urls: list[str],
    brief: str,
) -> str:
    competitors = "\n".join(f"- {u}" for u in competitor_urls) or "- (none provided; search Amazon for close alternatives)"
    brief_line = (brief or "Would you buy this product?").strip()
    priorities = ", ".join(persona.get("priorities") or []) or "value and reviews"
    avoids = ", ".join(persona.get("avoids") or []) or "unclear listings"
    extra = []
    if persona.get("prime"):
        extra.append(f"Amazon Prime: {persona['prime']}")
    if persona.get("shop_frequency"):
        extra.append(f"Shops {persona['shop_frequency']}")
    if persona.get("income"):
        extra.append(f"Household income: {persona['income']}")
    extra_line = ". ".join(extra)
    return f"""You are an OPeRA shopper: {persona.get('label')}.
{persona.get('bio') or ''}
Budget: {persona.get('budget') or 'not specified'}.
You care about: {priorities}.
You avoid: {avoids}.
{extra_line}

Shop as this real person. Do not break character. Do not log in or check out.

YOUR LISTING TO EVALUATE:
{product_url}

COMPETITOR LISTINGS TO ALSO OPEN:
{competitors}

RESEARCH BRIEF:
{brief_line}

Do this:
1. Open the listing. Read title, price, unit price if shown, star rating, review count, bullets, delivery, and a few reviews.
2. Open each competitor the same way.
3. Score the chance YOU personally buy the FIRST listing today, 0-100. This is not "is fish oil useful" — it is "would I add THIS listing to cart."
4. Be stingy. A fine generic option is usually 40-65. Reserve 80+ for a clear best-value winner with no blockers. 90+ only if you would buy it now without hesitation.
5. If you would buy a competitor instead, the first listing's likelihood should drop.
6. You may add the first product to cart only if this persona would actually buy it.

When finished, call done. The done result MUST be ONLY this JSON object — no preamble, no markdown:
{{
  "buy_likelihood": 0-100,
  "verdict": "buy" | "maybe" | "no",
  "product_selected": "short name of what you would buy, or none",
  "product_url": "url you would buy, or empty",
  "confidence": 0-100,
  "rationale": "2-4 sentences in this shopper's voice: why this likelihood vs the alternatives you saw",
  "price_perception": "cheap | fair | expensive, plus a short why",
  "trust_concerns": ["..."],
  "conversion_blockers": ["what would stop this shopper from buying the first listing"]
}}
Use verdict buy if buy_likelihood >= 70, maybe if 40-69, no if under 40.
If you are ready to decide, put that JSON in the done result now. Do not write "here's my decision" without the JSON.
"""


def shopper_listing_task(persona: dict[str, Any], listing_url: str, brief: str) -> str:
    brief_line = (brief or "Would you buy this product?").strip()
    priorities = ", ".join(persona.get("priorities") or []) or "value and reviews"
    avoids = ", ".join(persona.get("avoids") or []) or "unclear listings"
    extra = []
    if persona.get("prime"):
        extra.append(f"Amazon Prime: {persona['prime']}")
    if persona.get("shop_frequency"):
        extra.append(f"Shops {persona['shop_frequency']}")
    if persona.get("income"):
        extra.append(f"Household income: {persona['income']}")
    extra_line = ". ".join(extra)
    return f"""You are an OPeRA shopper: {persona.get('label')}.
{persona.get('bio') or ''}
Budget: {persona.get('budget') or 'not specified'}.
You care about: {priorities}.
You avoid: {avoids}.
{extra_line}

Shop as this real person. Do not break character. Do not log in or check out.

Open ONLY this listing and judge it:
{listing_url}

RESEARCH BRIEF:
{brief_line}

Do this:
1. Go to that URL. If Amazon shows Continue shopping, click it and stay on the listing.
2. Read title, price, unit price if shown, star rating, review count, bullets, delivery, and a few reviews.
3. Score the chance YOU personally buy THIS exact listing today, 0-100. Not whether the category is useful.
4. Be stingy. Average decent listings are 40-65. 80+ only if you would add it to cart now. Mention price, reviews, delivery, and any blockers.
5. Do not open other products. Do not search.

When finished, call done. The done result MUST be ONLY this JSON object — no preamble, no markdown:
{{
  "listing_title": "short product name",
  "buy_likelihood": 0-100,
  "would_buy_this": "buy" | "maybe" | "no",
  "verdict": "buy" | "maybe" | "no",
  "appeal": 0-100,
  "confidence": 0-100,
  "rationale": "2-4 sentences in this shopper's voice explaining the likelihood",
  "price_perception": "cheap | fair | expensive, plus a short why",
  "trust_concerns": ["..."],
  "conversion_blockers": ["..."]
}}
would_buy_this / verdict: buy if buy_likelihood >= 70, maybe if 40-69, no if under 40. appeal should match buy_likelihood.
"""
