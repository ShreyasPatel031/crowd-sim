#!/usr/bin/env python3
"""Replay converted OPeRA HTML through the live Playwright loop with a dummy oracle.

The dummy adapter returns the gold next action. Hands click/type by name=.
This checks whether the live agent can consume OPeRA-format observations.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from opera_repro.converter import ConverterConfig, session_to_examples
from opera_repro.data import load_fixture
from simulator.agent import run_simulation

DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "mini_sessions.json"
DEFAULT_SESSION = "sess_shoes"


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    parser.add_argument("--session-id", default=DEFAULT_SESSION)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--max-steps", type=int, default=8)
    args = parser.parse_args()

    rows = [row for row in load_fixture(args.fixture) if row["session_id"] == args.session_id]
    if not rows:
        raise SystemExit(f"No rows for session {args.session_id} in {args.fixture}")
    examples = session_to_examples(args.session_id, rows, "train", ConverterConfig())
    run_id = uuid.uuid4().hex[:10]
    out = ROOT / "data" / "sim_runs" / run_id
    report = await run_simulation(
        url="https://example.com",
        intent="opera-dummy",
        out_dir=out,
        max_steps=args.max_steps,
        headed=args.headed,
        opera_dummy_examples=examples,
    )
    print(
        json.dumps(
            {
                "id": run_id,
                "brain": report.get("brain"),
                "all_applied": report.get("all_applied"),
                "n_applied_ok": report.get("n_applied_ok"),
                "n_steps": report.get("n_steps"),
                "steps": [
                    {
                        "gold": s["gold_action"],
                        "applied_ok": s["applied_ok"],
                        "exact_match": s["exact_match"],
                        "detail": s["decision"].get("applied", {}).get("detail"),
                    }
                    for s in report["steps"]
                ],
            },
            indent=2,
        )
    )
    print(f"Report: {out / 'report.json'}")


if __name__ == "__main__":
    asyncio.run(main())
