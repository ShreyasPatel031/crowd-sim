#!/usr/bin/env python3
"""Run the Amazon shopper on Browser Use Cloud (their overlays + live session)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from browser_use_sdk import BrowserUse

TASK = (
    "Go to https://www.amazon.com. Find a men's fleece bathrobe under $25. "
    "Open a product page, check the price and whether it is actually fleece, "
    "and say whether you would add it to cart. Do not log in or check out."
)


def main() -> None:
    if not os.environ.get("BROWSER_USE_API_KEY"):
        raise SystemExit("BROWSER_USE_API_KEY is missing")

    client = BrowserUse()
    session = client.sessions.create(
        start_url="https://www.amazon.com",
        browser_screen_width=1280,
        browser_screen_height=800,
        proxy_country_code="us",
    )
    live = session.live_url
    dash = f"https://cloud.browser-use.com/sessions"
    print(f"session: {session.id}", flush=True)
    print(f"live: {live}", flush=True)
    print(f"dashboard: {dash}", flush=True)
    if live:
        subprocess.run(["open", live], check=False)

    created = client.tasks.create(
        TASK,
        session_id=str(session.id),
        llm="browser-use-2.0",
        highlight_elements=True,
        vision=True,
        max_steps=15,
    )
    print(f"task: {created.id}", flush=True)

    seen = 0
    view = None
    while True:
        view = client.tasks.get(str(created.id))
        status = view.status.value if hasattr(view.status, "value") else str(view.status)
        for step in view.steps[seen:]:
            print(f"  step {step.number}: {step.next_goal} — {step.url[:90]}", flush=True)
            seen = step.number
        if status in ("finished", "failed", "stopped"):
            print(f"status: {status}", flush=True)
            break
        time.sleep(3)

    run_id = uuid.uuid4().hex[:10]
    out = ROOT / "data" / "sim_runs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": run_id,
        "browser": "browser-use-cloud",
        "model": view.llm if view else "gemini-2.5-flash",
        "task_id": str(created.id),
        "session_id": str(session.id),
        "live_url": live,
        "status": status,
        "output": view.output if view else None,
        "is_success": view.is_success if view else None,
        "cost": str(view.cost.root) if view and view.cost else None,
        "steps": [
            {
                "number": s.number,
                "url": s.url,
                "next_goal": s.next_goal,
                "screenshot_url": s.screenshot_url,
            }
            for s in (view.steps if view else [])
        ],
    }
    (out / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"id": run_id, "status": status, "output": payload["output"]}, indent=2))
    print(f"Report: {out / 'report.json'}")


if __name__ == "__main__":
    main()
