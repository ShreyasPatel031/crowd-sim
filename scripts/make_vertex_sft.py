#!/usr/bin/env python3
"""Convert processed OPeRA splits into Vertex supervised-tuning JSONL.

Vertex wants one JSON object per line:

    {"systemInstruction": {"parts": [{"text": ...}]},
     "contents": [{"role": "user",  "parts": [{"text": ...}]},
                  {"role": "model", "parts": [{"text": ...}]}]}

Two hard limits drive the checks below: 131,072 tokens per example and 1GB per
training file. Validation is capped at 256 examples, the limit the tuning API
reference gives.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Vertex limits for gemini-2.5-flash supervised tuning.
MAX_TOKENS_PER_EXAMPLE = 131_072
MAX_TRAIN_FILE_BYTES = 1_000_000_000
MAX_VAL_EXAMPLES = 256

# countTokens on the largest examples in this dataset came out at 2.68-2.70
# chars/token — the candidate list is dense, dot-separated slugs, so it tokenizes
# far worse than prose. Guard below the observed floor so the local screen never
# passes an example the API would reject.
CHARS_PER_TOKEN = 2.6


def to_vertex(record: dict) -> dict:
    """Reshape one processed record into a Vertex tuning example."""
    by_role = {m["role"]: m["content"] for m in record["messages"]}
    return {
        "systemInstruction": {"parts": [{"text": by_role["system"]}]},
        "contents": [
            {"role": "user", "parts": [{"text": by_role["user"]}]},
            {"role": "model", "parts": [{"text": by_role["assistant"]}]},
        ],
    }


def example_chars(example: dict) -> int:
    system = example["systemInstruction"]["parts"][0]["text"]
    turns = sum(len(c["parts"][0]["text"]) for c in example["contents"])
    return len(system) + turns


def convert(src: Path, dst: Path, limit: int | None, seed: int) -> dict:
    rows = [json.loads(line) for line in src.open()]
    if limit is not None and len(rows) > limit:
        random.Random(seed).shuffle(rows)
        rows = rows[:limit]

    kept, skipped, sizes = [], [], []
    for row in rows:
        example = to_vertex(row)
        chars = example_chars(example)
        if chars / CHARS_PER_TOKEN > MAX_TOKENS_PER_EXAMPLE:
            skipped.append((row["session_id"], row["action_id"], chars))
            continue
        kept.append(example)
        sizes.append(chars)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as fh:
        for example in kept:
            fh.write(json.dumps(example, ensure_ascii=False) + "\n")

    sizes.sort()
    return {
        "source": str(src),
        "output": str(dst),
        "examples": len(kept),
        "skipped_oversize": len(skipped),
        "skipped_detail": skipped[:5],
        "bytes": dst.stat().st_size,
        "chars_median": sizes[len(sizes) // 2] if sizes else 0,
        "chars_max": sizes[-1] if sizes else 0,
        "est_tokens_total": int(sum(sizes) / 2.7),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default=str(ROOT / "data/processed/train.jsonl"))
    parser.add_argument("--val", default=str(ROOT / "data/processed/val.jsonl"))
    parser.add_argument("--outdir", default=str(ROOT / "data/vertex"))
    parser.add_argument("--val-limit", type=int, default=MAX_VAL_EXAMPLES)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    report = {
        "train": convert(Path(args.train), outdir / "train_sft.jsonl", args.train_limit, args.seed),
        "val": convert(Path(args.val), outdir / "val_sft.jsonl", args.val_limit, args.seed),
    }

    for split, info in report.items():
        gb = info["bytes"] / 1e9
        print(f"{split}:")
        print(f"  examples        {info['examples']:,} (skipped {info['skipped_oversize']} oversize)")
        print(f"  file size       {gb:.3f} GB")
        print(f"  chars median    {info['chars_median']:,}   max {info['chars_max']:,}")
        print(f"  est. tokens     {info['est_tokens_total']:,}")
        for sid, aid, chars in info["skipped_detail"]:
            print(f"    oversize: {sid} step {aid} — {chars:,} chars")

    train_bytes = report["train"]["bytes"]
    if train_bytes > MAX_TRAIN_FILE_BYTES:
        raise SystemExit(f"train file is {train_bytes/1e9:.2f} GB, over the 1GB limit")
    if report["val"]["examples"] > MAX_VAL_EXAMPLES:
        raise SystemExit(f"val has {report['val']['examples']} examples, over {MAX_VAL_EXAMPLES}")

    (outdir / "convert_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {outdir}/convert_report.json")


if __name__ == "__main__":
    main()
