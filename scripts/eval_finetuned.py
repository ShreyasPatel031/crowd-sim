#!/usr/bin/env python3
"""Evaluate a fine-tuned Vertex endpoint (or local LoRA adapter) on held-out OPeRA."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENDPOINT = "projects/347838016394/locations/us-central1/endpoints/1173650047569494016"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--data", default=str(ROOT / "data" / "processed" / "test.jsonl"))
    parser.add_argument("--adapter", default=None, help="Local LoRA dir, or a Vertex endpoint resource")
    parser.add_argument("--endpoint", default=None, help="Vertex tuned endpoint resource")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--out", default=str(ROOT / "data" / "eval" / "gemini_sft_e1_honest.json"))
    args = parser.parse_args()

    endpoint = args.endpoint or (
        args.adapter if args.adapter and args.adapter.startswith("projects/") else None
    )
    argv = [
        str(ROOT / "scripts" / "eval_baseline.py"),
        "--config",
        args.config,
        "--data",
        args.data,
        "--out",
        args.out,
        "--workers",
        str(args.workers),
    ]
    if endpoint:
        argv.extend(["--endpoint", endpoint])
    elif args.adapter:
        argv.extend(["--adapter", args.adapter, "--backend", "local"])
    else:
        argv.extend(["--endpoint", DEFAULT_ENDPOINT])
    if args.limit is not None:
        argv.extend(["--limit", str(args.limit)])
    sys.argv = argv
    runpy.run_path(str(ROOT / "scripts" / "eval_baseline.py"), run_name="__main__")


if __name__ == "__main__":
    main()
