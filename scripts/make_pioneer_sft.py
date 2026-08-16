#!/usr/bin/env python3
"""Convert processed OPeRA splits into Pioneer decoder SFT JSONL.

Pioneer wants one chat object per line:

    {"messages": [{"role": "system", ...}, {"role": "user", ...},
                  {"role": "assistant", ...}]}

Qwen3-8B on Pioneer qualifies 40,960 input tokens. OPeRA observations are dense
dot-separated slugs that tokenize at roughly 2.6 chars/token, far worse than
prose, so the budget below is deliberately conservative.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MAX_INPUT_TOKENS = 40_960
CHARS_PER_TOKEN = 2.6
# Headroom for the chat template and the generated action.
CHAR_BUDGET = int(MAX_INPUT_TOKENS * CHARS_PER_TOKEN) - 6_000


def example_chars(messages: list[dict]) -> int:
    return sum(len(m["content"]) for m in messages)


def convert(src: Path, dst: Path, limit: int | None) -> dict:
    kept: list[dict] = []
    oversize = 0
    sizes: list[int] = []

    for line in src.open():
        row = json.loads(line)
        messages = row["messages"]
        chars = example_chars(messages)
        if chars > CHAR_BUDGET:
            oversize += 1
            continue
        kept.append({"messages": messages})
        sizes.append(chars)
        if limit is not None and len(kept) >= limit:
            break

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as fh:
        for example in kept:
            fh.write(json.dumps(example, ensure_ascii=False) + "\n")

    sizes.sort()
    return {
        "output": str(dst),
        "examples": len(kept),
        "skipped_oversize": oversize,
        "bytes": dst.stat().st_size,
        "chars_median": sizes[len(sizes) // 2] if sizes else 0,
        "chars_max": sizes[-1] if sizes else 0,
        "est_tokens_max": int(sizes[-1] / CHARS_PER_TOKEN) if sizes else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default=str(ROOT / "data/processed/train.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "data/pioneer/train_sft.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    info = convert(Path(args.src), Path(args.out), args.limit)
    print(f"examples        {info['examples']:,} (skipped {info['skipped_oversize']} oversize)")
    print(f"file size       {info['bytes']:,} bytes")
    print(f"chars median    {info['chars_median']:,}   max {info['chars_max']:,}")
    print(f"est max tokens  {info['est_tokens_max']:,} / {MAX_INPUT_TOKENS:,}")
    print(f"wrote           {info['output']}")


if __name__ == "__main__":
    main()
