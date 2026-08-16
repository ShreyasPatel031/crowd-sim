"""Find OPeRA shoppers similar to a free-text target description or seed persona."""

from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from simulator.personas import all_personas, load_catalog

EXAMPLES_PATH = Path(__file__).resolve().parents[1] / "data" / "example_product_personas.json"

LEVEL = {"Low": 0.25, "Medium": 0.5, "High": 0.75, "Extremely high": 1.0}

# Seed users for the seven shopper archetypes surfaced in product research.
ARCHETYPE_SEEDS: dict[str, dict[str, str]] = {
    "spec_researcher": {
        "seed_id": "85aeec61-2d4e-489d-93bf-76c928d2d795",
        "title": "Meticulous spec researcher",
        "hint": "reads reviews, compares specs, avoids impulse buys",
    },
    "budget_health_grad": {
        "seed_id": "3c426651-5067-498d-9b54-a1109090b85f",
        "title": "Budget-conscious grad student",
        "hint": "PhD stipend, quality supplements, hunts value",
    },
    "home_furnisher_premium": {
        "seed_id": "03a4f3ea-bfb7-4ba9-a7e5-5033f6958ce1",
        "title": "Frequent home furnisher",
        "hint": "shops often, premium bedding and home goods",
    },
    "smart_home_tinkerer": {
        "seed_id": "3a65c43e-e4e3-4280-8ea3-526cc99b9553",
        "title": "Smart-home / maker tinkerer",
        "hint": "heavy Amazon user, compares Ring vs Aqara, PC parts",
    },
    "cross_brand_comparer": {
        "seed_id": "3539f4ff-2781-4b5b-af33-6ace9752c557",
        "title": "Cross-brand deal hunter",
        "hint": "Oral-B vs Sonicare, low monthly spend, reads ads",
    },
    "speed_buyer_high_earner": {
        "seed_id": "f2778310-f9e8-4208-a73f-c098e00cacdc",
        "title": "Fast high-earner",
        "hint": "Bay Area engineer, buys first good option, high budget",
    },
    "exact_model_hunter": {
        "seed_id": "7f0c8207-6a6f-49cd-9d7e-17987cfafcb9",
        "title": "Exact-model brand hunter",
        "hint": "searches full product name, premium brands, cyclist",
    },
}


def _budget_num(persona: dict[str, Any]) -> int:
    text = persona.get("budget") or ""
    match = re.search(r"\$([\d,]+)", text.replace(",", ""))
    if match:
        return int(match.group(1))
    lowered = text.lower()
    if "50-150" in text:
        return 100
    if "100-500" in text:
        return 300
    if "about 50" in lowered:
        return 50
    if "20-50" in text:
        return 35
    return 150


def _persona_document(persona: dict[str, Any]) -> str:
    parts = [
        persona.get("label") or "",
        persona.get("name") or "",
        persona.get("bio") or "",
        persona.get("city") or "",
        persona.get("employment") or "",
        persona.get("education") or "",
        persona.get("gender") or "",
        persona.get("income") or "",
        persona.get("budget") or "",
        persona.get("shop_frequency") or "",
        persona.get("prime") or "",
        " ".join(persona.get("priorities") or []),
        " ".join(persona.get("avoids") or []),
        " ".join(f"{key} {value}" for key, value in (persona.get("personality") or {}).items()),
    ]
    for meta in ARCHETYPE_SEEDS.values():
        if meta["seed_id"] == persona.get("id"):
            parts.extend([meta["title"], meta["hint"]])
    return " ".join(parts).lower()


def _tokenize(text: str) -> list[str]:
    return [tok for tok in re.findall(r"[a-z0-9]+", text.lower()) if len(tok) > 2]


def _text_overlap(query: str, persona: dict[str, Any]) -> float:
    tokens = _tokenize(query)
    if not tokens:
        return 0.0
    doc = _persona_document(persona)
    hits = sum(1 for token in tokens if token in doc)
    return hits / len(tokens)


def structured_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    """Profile similarity from shopping prefs, budget, personality, activity."""
    lp, rp = set(left.get("priorities") or []), set(right.get("priorities") or [])
    la, ra = set(left.get("avoids") or []), set(right.get("avoids") or [])
    priority = len(lp & rp) / max(len(lp | rp), 1)
    avoid = len(la & ra) / max(len(la | ra), 1)
    employment = (
        1.0
        if left.get("employment") == right.get("employment")
        else 0.4
        if (left.get("employment") or "")[:6] == (right.get("employment") or "")[:6]
        else 0.0
    )
    prime = 1.0 if left.get("prime") == right.get("prime") else 0.0
    frequency = 1.0 if left.get("shop_frequency") == right.get("shop_frequency") else 0.5
    lb, rb = _budget_num(left), _budget_num(right)
    budget = 1.0 - min(abs(math.log10(lb + 1) - math.log10(rb + 1)) / 2, 1.0)
    lp_p, rp_p = left.get("personality") or {}, right.get("personality") or {}
    keys = set(lp_p) & set(rp_p)
    personality = (
        sum(
            1 - abs(LEVEL.get(lp_p[key], 0.5) - LEVEL.get(rp_p[key], 0.5))
            for key in keys
        )
        / max(len(keys), 1)
        if keys
        else 0.5
    )
    sessions = 1.0 - min(abs(left.get("session_count", 0) - right.get("session_count", 0)) / 30, 1.0)
    return (
        0.32 * priority
        + 0.12 * avoid
        + 0.10 * employment
        + 0.06 * prime
        + 0.08 * frequency
        + 0.12 * budget
        + 0.12 * personality
        + 0.08 * sessions
    )


def _sample_queries(user_id: str, limit: int = 4) -> list[str]:
    catalog = load_catalog()
    queries: list[str] = []
    for task in catalog.get("tasks") or []:
        if user_id not in (task.get("user_ids") or []):
            continue
        query = (task.get("query") or "").strip()
        if query and query not in queries:
            queries.append(query)
        if len(queries) >= limit:
            break
    return queries


def _public_card(persona: dict[str, Any], score: float, *, match_reason: str = "") -> dict[str, Any]:
    bio = (persona.get("bio") or "").split("###")[0].strip()
    return {
        "id": persona["id"],
        "label": persona.get("label") or "",
        "name": persona.get("name") or "",
        "city": persona.get("city") or "",
        "budget": persona.get("budget") or "",
        "shop_frequency": persona.get("shop_frequency") or "",
        "prime": persona.get("prime") or "",
        "session_count": persona.get("session_count") or 0,
        "priorities": persona.get("priorities") or [],
        "avoids": persona.get("avoids") or [],
        "bio": bio[:240],
        "sample_searches": _sample_queries(persona["id"]),
        "score": round(score, 3),
        "match_reason": match_reason,
    }


def similar_personas(persona_id: str, *, limit: int = 5) -> list[dict[str, Any]]:
    people = all_personas()
    by_id = {person["id"]: person for person in people}
    seed = by_id.get(persona_id)
    if not seed:
        return []
    ranked = sorted(
        (
            (
                structured_similarity(seed, person),
                person,
            )
            for person in people
            if person["id"] != persona_id
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    return [
        _public_card(person, score, match_reason="similar shopping profile")
        for score, person in ranked[:limit]
    ]


def search_personas(query: str, *, limit: int = 8) -> list[dict[str, Any]]:
    """Match a free-text target shopper description to real OPeRA personas."""
    people = all_personas()
    if not people:
        return []

    query = (query or "").strip()
    if not query:
        return [_public_card(person, 0.0) for person in people[:limit]]

    # If the query names an archetype slug, anchor on its seed persona.
    anchor: dict[str, Any] | None = None
    lowered = query.lower().replace(" ", "_")
    for key, meta in ARCHETYPE_SEEDS.items():
        if key in lowered or meta["title"].lower() in query.lower():
            anchor = next((p for p in people if p["id"] == meta["seed_id"]), None)
            break

    ranked: list[tuple[float, dict[str, Any], str]] = []
    for person in people:
        text = _text_overlap(query, person)
        profile = structured_similarity(anchor, person) if anchor else 0.0
        score = 0.55 * text + 0.45 * profile if anchor else text
        if anchor and person["id"] == anchor["id"]:
            score = max(score, 0.99)
        reason_parts = []
        if text >= 0.25:
            reason_parts.append("keyword overlap")
        if profile >= 0.55:
            reason_parts.append("similar shopping profile")
        if not reason_parts:
            reason_parts.append("partial match")
        ranked.append((score, person, ", ".join(reason_parts)))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [_public_card(person, score, match_reason=reason) for score, person, reason in ranked[:limit]]


def list_archetypes() -> list[dict[str, Any]]:
    people = all_personas()
    by_id = {person["id"]: person for person in people}
    out: list[dict[str, Any]] = []
    for key, meta in ARCHETYPE_SEEDS.items():
        seed = by_id.get(meta["seed_id"])
        if not seed:
            continue
        out.append(
            {
                "archetype": key,
                "title": meta["title"],
                "hint": meta["hint"],
                "seed": _public_card(seed, 1.0, match_reason="archetype seed"),
                "similar": similar_personas(seed["id"], limit=5),
            }
        )
    return out


@lru_cache(maxsize=1)
def load_example_products() -> dict[str, Any]:
    if not EXAMPLES_PATH.exists():
        return {"persona_type_definitions": {}, "examples": []}
    return json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))


def list_example_products() -> list[dict[str, Any]]:
    """Curated product × persona-type pairs for panel tests."""
    return list(load_example_products().get("examples") or [])
