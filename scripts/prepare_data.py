#!/usr/bin/env python3
"""Download OPeRA-filtered and convert sessions into next-action SFT examples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from opera_repro.config import converter_config_from_yaml, load_config
from opera_repro.converter import convert_rows, split_stats, write_jsonl
from opera_repro.data import load_fixture, load_opera_filtered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--fixture", default=None, help="Use a local JSON fixture instead of HuggingFace")
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    converter_cfg = converter_config_from_yaml(cfg)
    if args.fixture:
        rows = load_fixture(args.fixture)
        print(f"Loaded fixture {args.fixture}: {len(rows)} rows")
    else:
        print("Downloading NEU-HAI/OPeRA filtered_action from HuggingFace...")
        rows = load_opera_filtered(cache_dir=args.cache_dir)
        print(f"Loaded {len(rows)} filtered actions")

    examples = convert_rows(rows, converter_cfg)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = split_stats(examples)
    for split, split_rows in examples.items():
        path = out_dir / f"{split}.jsonl"
        n = write_jsonl(path, split_rows)
        print(f"Wrote {n:5d} examples → {path}")

    report_path = out_dir / "split_stats.json"
    report_path.write_text(json.dumps({"config": converter_cfg.__dict__, "stats": stats}, indent=2), encoding="utf-8")
    print("\nSplit stats")
    print(json.dumps(stats, indent=2))
    print(f"\nSaved {report_path}")
    _check_leakage(examples)


def _check_leakage(examples: dict[str, list[dict]]) -> None:
    ids = {split: {row["session_id"] for row in rows} for split, rows in examples.items()}
    leak = (ids["train"] & ids["test"]) | (ids["train"] & ids["val"]) | (ids["val"] & ids["test"])
    if leak:
        raise SystemExit(f"Session leakage detected: {sorted(leak)[:5]}")
    print("Session leakage check: OK")


if __name__ == "__main__":
    main()
