#!/usr/bin/env python3
"""Build a local catalog of OPeRA personas and shopping tasks."""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import load_dataset

CACHE = ROOT / "data" / "cache"
OUT = ROOT / "data" / "opera_catalog.json"

AGREE = {"strongly agree", "somewhat agree"}
DISAGREE = {"strongly disagree", "somewhat disagree"}


def _load_json(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _priorities(statements: dict[str, str]) -> tuple[list[str], list[str]]:
    mapping = {
        "I usually do a lot research (e.g. reading online reviews) before making purchase.": "reviews",
        "I prioritize delivery speed and delivery fee of the product.": "delivery speed",
        "Getting high-quality online products is very important for me.": "quality",
        "The more expensive online product brands are usually my choice.": "premium brands",
        "I look carefully to find the best value for money when shopping online.": "best value",
        "I shop quickly for online products, buying the first product or brand I find that seems good enough.": "speed / good-enough",
        "Once I find a brand I like, I stick with it.": "brand loyalty",
        "Online Ads attract my attention and are a good source of information.": "ads / sponsored",
    }
    priorities: list[str] = []
    avoids: list[str] = []
    for statement, trait in mapping.items():
        answer = (statements.get(statement) or "").strip().lower()
        if answer in AGREE:
            priorities.append(trait)
        elif answer in DISAGREE:
            avoids.append(trait)
    return priorities[:6], avoids[:6]


def persona_from_user(row: dict) -> dict:
    survey = _load_json(row.get("survey"))
    demo = survey.get("Demographic Information") or {}
    desc = survey.get("Self Description") or {}
    pref = survey.get("Shopping Preference") or {}
    statements = pref.get("To what extent do you agree with the following statements") or {}
    personality = (survey.get("Personality") or {}).get("Big Five Scores") or {}
    priorities, avoids = _priorities(statements)
    uid = str(row.get("user_id") or "")
    age = demo.get("Age") or ""
    job = demo.get("Employment status") or "Shopper"
    city = demo.get("City") or ""
    spend = str(pref.get("Monthly online shopping spend $") or "").strip()
    bio = (desc.get("Two sentence description") or "").strip()
    interview = (row.get("interview_transcript_processed") or "").strip()
    if interview:
        bio = (bio + " " + interview[:400]).strip()
    label_parts = [p for p in (job, age.split(" years")[0] if age else "") if p]
    return {
        "id": uid,
        "source": "opera",
        "name": f"OPeRA {job or 'shopper'} in {city or 'US'}".strip(),
        "label": " · ".join(label_parts) or "OPeRA shopper",
        "age": age,
        "city": city,
        "gender": demo.get("Gender") or "",
        "income": demo.get("Yearly household income or stipend") or "",
        "education": demo.get("Education level") or "",
        "employment": job,
        "prime": pref.get("Amazon Prime membership") or "",
        "shop_frequency": pref.get("Online shopping frequency") or "",
        "budget": f"About ${spend}/month online" if spend else "Not specified",
        "bio": bio or f"{job} shopper, {age}, {city}.".strip(", "),
        "priorities": priorities or ["reviews", "value"],
        "avoids": avoids or ["unclear listings"],
        "personality": personality,
        "session_count": 0,
    }


def _english_query(text: str) -> bool:
    if not text or len(text) < 3 or len(text) > 80:
        return False
    letters = sum(ch.isalpha() and ch.isascii() for ch in text)
    return letters >= 3 and letters / max(len(text), 1) > 0.35


def build_catalog() -> dict:
    users = load_dataset("NEU-HAI/OPeRA", "filtered_user", cache_dir=str(CACHE))
    actions = load_dataset("NEU-HAI/OPeRA", "filtered_action", cache_dir=str(CACHE))
    personas: dict[str, dict] = {}
    for split in users.values():
        for row in split:
            uid = row["user_id"]
            if uid not in personas:
                personas[uid] = persona_from_user(row)

    sessions: dict[str, list] = defaultdict(list)
    for split in actions.values():
        for row in split:
            sessions[row["session_id"]].append(row)

    tasks: dict[str, dict] = {}
    for sid, rows in sessions.items():
        uid = sid.split("_", 1)[0]
        if uid in personas:
            personas[uid]["session_count"] += 1
        rows = sorted(rows, key=lambda r: r.get("timestamp") or "")
        query = ""
        titles: list[str] = []
        asins: list[str] = []
        for row in rows:
            if not query and row.get("action_type") == "input":
                text = (row.get("input_text") or "").strip()
                if _english_query(text):
                    query = text
            if row.get("products"):
                try:
                    prods = json.loads(row["products"])
                except json.JSONDecodeError:
                    prods = []
                if isinstance(prods, dict):
                    prods = [prods]
                for prod in prods or []:
                    if not isinstance(prod, dict):
                        continue
                    title = (prod.get("title") or "").strip()
                    asin = (prod.get("asin") or "").strip()
                    if title and title not in titles:
                        titles.append(title)
                    if asin and asin not in asins:
                        asins.append(asin)
        if not query:
            continue
        key = re.sub(r"\s+", " ", query.lower()).strip()
        task = tasks.setdefault(
            key,
            {
                "id": f"task_{len(tasks)+1:03d}",
                "query": query,
                "brief": (
                    f"Shop for '{query}' the way this person actually shops. "
                    "Open the given listing, compare nearby alternatives, and say buy, maybe, or no."
                ),
                "search_url": f"https://www.amazon.com/s?k={query.replace(' ', '+')}",
                "example_products": [],
                "asins": [],
                "session_ids": [],
                "user_ids": [],
            },
        )
        task["session_ids"].append(sid)
        if uid not in task["user_ids"]:
            task["user_ids"].append(uid)
        for title in titles[:3]:
            if title not in task["example_products"]:
                task["example_products"].append(title)
        for asin in asins[:4]:
            if asin not in task["asins"]:
                task["asins"].append(asin)

    ranked_tasks = sorted(tasks.values(), key=lambda t: (-len(t["session_ids"]), t["query"]))
    ranked_people = sorted(personas.values(), key=lambda p: (-p["session_count"], p["label"]))
    return {
        "personas": ranked_people,
        "tasks": ranked_tasks,
        "n_personas": len(ranked_people),
        "n_tasks": len(ranked_tasks),
    }


def main() -> None:
    catalog = build_catalog()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    print(f"Wrote {catalog['n_personas']} personas and {catalog['n_tasks']} tasks → {OUT}")
    print("Sample personas:")
    for p in catalog["personas"][:5]:
        print(f"  {p['session_count']:3d} sess  {p['label']}  {p['budget']}")
    print("Sample tasks:")
    for t in catalog["tasks"][:8]:
        print(f"  {len(t['session_ids']):2d} sess  {t['query']}")


if __name__ == "__main__":
    main()
