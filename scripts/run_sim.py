#!/usr/bin/env python3
"""CLI: python scripts/run_sim.py --url https://books.toscrape.com --intent 'cheap travel book'"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from simulator.agent import run_simulation


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--intent", default="Find something you might buy.")
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--max-steps", type=int, default=30)
    args = parser.parse_args()
    run_id = uuid.uuid4().hex[:10]
    out = ROOT / "data" / "sim_runs" / run_id
    report = await run_simulation(args.url, args.intent, out, max_steps=args.max_steps, headed=args.headed)
    print(json.dumps({
        "id": run_id,
        "browser": report.get("browser"),
        "llm": report.get("llm"),
        "would_prefer": report["would_prefer"],
        "steps": len(report["steps"]),
    }, indent=2))
    print(f"Report: {out / 'report.json'}")


if __name__ == "__main__":
    asyncio.run(main())
