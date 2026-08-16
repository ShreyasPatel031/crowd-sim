#!/usr/bin/env python3
"""Open-source Browser Use + your Google AI Studio Gemini key."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from browser_use import Agent, Browser, ChatGoogle

DEFAULT_TASK = (
    "Go to https://www.amazon.com. Find a men's fleece bathrobe under $25. "
    "Open a product page, check the price and whether it is actually fleece, "
    "and say whether you would add it to cart. Do not log in or check out."
)


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--max-steps", type=int, default=30)
    args = parser.parse_args()

    if not os.environ.get("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY is missing")

    llm = ChatGoogle(model="gemini-2.5-flash")
    browser = Browser(headless=False, highlight_elements=True, use_cloud=False)
    agent = Agent(task=args.task, llm=llm, browser=browser)
    history = await agent.run(max_steps=args.max_steps)

    run_id = uuid.uuid4().hex[:10]
    out = ROOT / "data" / "sim_runs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    final = history.final_result() if hasattr(history, "final_result") else str(history)
    payload = {
        "id": run_id,
        "browser": "local-chromium",
        "model": "gemini-2.5-flash",
        "task": args.task,
        "output": final,
        "is_done": history.is_done() if hasattr(history, "is_done") else None,
    }
    (out / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Report: {out / 'report.json'}")


if __name__ == "__main__":
    asyncio.run(main())
