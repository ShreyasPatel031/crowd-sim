#!/usr/bin/env python3
"""Evaluate a model (or gold/dummy predictions) with session-macro exact match."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from opera_repro.actions import Action, actions_equal, parse_action
from opera_repro.config import load_config
from opera_repro.converter import iter_jsonl
from opera_repro.evaluate import evaluate_predictions, format_report
from opera_repro.prompts import REASONING_RESPONSE_SCHEMA, REASONING_SYSTEM, SYSTEM_PROMPT

load_dotenv(ROOT / ".env")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--data", default=str(ROOT / "data" / "processed" / "test.jsonl"))
    parser.add_argument("--adapter", default=None, help="Optional LoRA adapter dir")
    parser.add_argument("--model", default=None, help="Override base model name")
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Vertex tuned endpoint resource (projects/.../endpoints/...). Implies --backend gemini.",
    )
    parser.add_argument("--backend", choices=("local", "gemini"), default="local")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=24, help="Parallel Gemini requests")
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=128,
        help="Raise this when targets carry a reason field, or long answers get truncated",
    )
    parser.add_argument("--oracle", action="store_true", help="Score gold labels (sanity check → 100%)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    records = list(iter_jsonl(args.data))
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise SystemExit(f"No examples in {args.data}. Run scripts/prepare_data.py first.")

    if args.endpoint:
        os.environ["GEMINI_ENDPOINT"] = args.endpoint
        args.backend = "gemini"

    if args.oracle:
        preds = [json.dumps(row["gold_action"], ensure_ascii=False) for row in records]
    elif args.backend == "gemini":
        preds = _generate_gemini(records, workers=args.workers, max_output_tokens=args.max_output_tokens)
    else:
        preds = _generate(records, cfg, args.model, args.adapter)

    result = evaluate_predictions(records, preds)
    print(format_report(result))
    if args.out:
        payload = result.as_dict()
        payload["preds"] = [
            {
                "session_id": row["session_id"],
                "gold": row["gold_action"],
                "pred": parse_action(pred).to_dict() if parse_action(pred) else pred[:120],
                "hit": actions_equal(parse_action(pred), Action(**row["gold_action"])),
            }
            for row, pred in zip(records, preds)
        ]
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.out}")


def _generate(records, cfg, model_name, adapter):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "PyTorch/transformers are required for model eval.\n"
            "Install with: .venv/bin/pip install -r requirements-train.txt\n"
            "Or run a metric sanity check: python scripts/eval_baseline.py --oracle"
        ) from exc

    name = model_name or cfg["model"]["name"]
    eval_cfg = cfg.get("eval", {})
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    preds = []
    max_new = int(eval_cfg.get("max_new_tokens", 128))
    for i, row in enumerate(records, start=1):
        messages = [m for m in row["messages"] if m["role"] != "assistant"]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        preds.append(text)
        parsed = parse_action(text)
        gold = row["gold_action"]
        print(f"[{i}/{len(records)}] gold={gold} pred={parsed.to_dict() if parsed else text[:80]!r}")
    return preds


def _generate_gemini(records, workers: int = 24, max_output_tokens: int = 128):
    from simulator.llm import complete_json, current_model

    print(
        f"Gemini backend: {current_model()} workers={workers} max_output_tokens={max_output_tokens}",
        flush=True,
    )
    preds = [""] * len(records)

    def one(i: int, row: dict) -> tuple[int, str]:
        system = next(m["content"] for m in row["messages"] if m["role"] == "system")
        user = next(m["content"] for m in row["messages"] if m["role"] == "user")
        last_err = ""
        for attempt in range(6):
            try:
                parsed_json = complete_json(
                    system,
                    user,
                    temperature=0.0,
                    max_output_tokens=max_output_tokens,
                    thinking_budget=0,
                    verbose=False,
                    response_schema=(REASONING_RESPONSE_SCHEMA if "Never omit \"reason\"" in system else None),
                )
                return i, json.dumps(parsed_json, ensure_ascii=False) if parsed_json else ""
            except Exception as exc:
                last_err = str(exc)
                if any(code in last_err for code in ("429", "500", "503")):
                    time.sleep(min(2**attempt, 16) + random.random())
                    continue
                break
        print(f"  fail [{i}] {last_err[:160]}", flush=True)
        return i, ""

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, i, row) for i, row in enumerate(records)]
        for fut in as_completed(futures):
            i, text = fut.result()
            preds[i] = text
            done += 1
            if done == len(records) or done % 50 == 0:
                print(f"  [{done}/{len(records)}]", flush=True)
    return preds


if __name__ == "__main__":
    # SYSTEM_PROMPT imported so the eval path stays aligned with training prompts.
    _ = SYSTEM_PROMPT
    main()
