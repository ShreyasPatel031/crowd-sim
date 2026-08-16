"""Terac human-study client (general population)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

TERAC_BASE = os.environ.get("TERAC_API_BASE", "https://terac.com/api/external/v2")


def terac_available() -> bool:
    return bool(os.environ.get("TERAC_API_KEY"))


def _headers() -> dict[str, str]:
    key = os.environ.get("TERAC_API_KEY") or ""
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }


def _request(method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{TERAC_BASE}{path}",
        data=data,
        headers=_headers(),
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Terac {exc.code}: {body[:500]}") from exc


def ensure_project() -> str:
    existing = os.environ.get("TERAC_PROJECT_ID")
    if existing:
        return existing
    created = _request(
        "POST",
        "/projects",
        {"name": "Shopper panel (hackathon)", "description": "Synthetic vs human Amazon listing tests"},
    )
    return str(created.get("id") or created.get("project_id") or "")


TERAC_ATELIER = os.environ.get(
    "TERAC_ATELIER_URL",
    "https://terac.com/atelier-msuo0cw0",
)


def find_audience_matches(query: str, *, limit: int = 8) -> dict[str, Any]:
    """Match a free-text shopper description to OPeRA-style hints + Terac feasibility."""
    from simulator.persona_search import search_personas

    q = (query or "").strip()
    opera_hits = search_personas(q, limit=limit) if q else []
    payload: dict[str, Any] = {
        "query": q,
        "opera_matches": opera_hits,
        "atelier_url": TERAC_ATELIER,
        "status": "local_match",
        "message": "Matched OPeRA shopper profiles. Use Terac to recruit real people with the same traits.",
    }
    if not terac_available() or not q:
        payload["status"] = "opera_only" if opera_hits else "empty"
        if not terac_available():
            payload["message"] = "Set TERAC_API_KEY to request a priced human panel on Terac."
        return payload
    try:
        feasibility = request_audience_feasibility(q, n=max(5, min(limit * 3, 25)))
        payload.update(feasibility)
        payload["status"] = feasibility.get("status") or "requested"
    except Exception as exc:
        payload["status"] = "error"
        payload["error"] = str(exc)[:400]
    return payload


def request_audience_feasibility(description: str, *, n: int = 10) -> dict[str, Any]:
    """Ask Terac to price recruiting participants matching a shopper description."""
    if not terac_available():
        return {"status": "skipped", "reason": "TERAC_API_KEY missing"}
    project_id = ensure_project()
    payload = {
        "project_id": project_id,
        "description": description[:4000],
        "target_participants": max(1, min(n, 50)),
        "estimated_duration_minutes": 5,
        "business_type": "b2c",
    }
    try:
        created = _request("POST", "/feasibility-requests", payload)
    except Exception:
        created = _request(
            "POST",
            "/feasibility_requests",
            {
                "projectId": project_id,
                "prompt": description[:4000],
                "participantCount": max(1, min(n, 50)),
            },
        )
    req_id = str(created.get("id") or created.get("feasibility_request_id") or "")
    return {
        "feasibility_id": req_id,
        "feasibility": created,
        "status": created.get("status") or "requested",
        "message": "Terac is pricing this audience — poll feasibility until RESPONDED.",
        "dashboard": TERAC_ATELIER,
    }


def launch_listing_study(
    *,
    product_url: str,
    brief: str,
    task_url: str,
    n: int = 10,
) -> dict[str, Any]:
    """Create a general-population draft that asks real people the same buy/maybe/no question."""
    if not terac_available():
        return {"status": "skipped", "reason": "TERAC_API_KEY missing"}
    project_id = ensure_project()
    title = "Would you buy this Amazon product?"
    description = (
        f"Open the listing, skim price/reviews, and say if you would buy it.\n\n"
        f"Product: {product_url}\nBrief: {brief}\n"
        "Answer buy, maybe, or no, then one short reason."
    )
    payload = {
        "title": title[:200],
        "internal_title": "Hackathon shopper panel — general population",
        "description": description[:5000],
        "project_id": project_id,
        "num_participants": max(1, min(n, 25)),
        "business_type": "b2c",
        "unrestricted_audience": True,
        "estimated_duration_minutes": 5,
        "screening_questions": [
            {
                "key": "amazon_shopper",
                "text": "Have you bought something on Amazon in the last 12 months?",
                "pick": "one",
                "answers": [
                    {"text": "Yes", "qualify_logic": "must"},
                    {"text": "No", "qualify_logic": "reject"},
                ],
            }
        ],
        "tasks": [
            {
                "sequence": 1,
                "task_type": "survey",
                "review_type": "auto_approve",
                "task_url": task_url,
                "title": "Rate this listing",
                "description": "Visit the page, then submit buy / maybe / no plus a short reason.",
                "duration_minutes": 5,
            }
        ],
    }
    draft = _request("POST", "/opportunities", payload)
    opportunity_id = str(draft.get("id") or "")
    launched = None
    if opportunity_id:
        try:
            launched = _request("POST", f"/opportunities/{opportunity_id}/launch", {})
        except Exception as exc:
            launched = {"status": "draft_only", "error": str(exc)[:300]}
    return {
        "status": "created",
        "project_id": project_id,
        "opportunity": draft,
        "launch": launched,
        "dashboard": ((draft.get("links") or {}).get("dashboard") or {}),
    }
