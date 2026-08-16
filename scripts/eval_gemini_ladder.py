#!/usr/bin/env python3
"""Prompt ladder on a small official OPeRA-test slice.

  1. vanilla Gemini
  2. Gemini + OPeRA action/tag prompt
  3. Gemini + tag prompt + few-shot train examples

Same records, greedy decode, session-macro exact match. No fine-tune.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from opera_repro.actions import Action, actions_equal, parse_action
from opera_repro.config import converter_config_from_yaml, load_config
from opera_repro.converter import convert_rows, write_jsonl
from opera_repro.data import KEEP_FIELDS
from opera_repro.evaluate import evaluate_predictions, format_report
from opera_repro.prompts import TAGS_SYSTEM, VANILLA_SYSTEM, fewshot_block


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=6)
    parser.add_argument("--max-examples", type=int, default=48)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--shots", type=int, default=3)
    parser.add_argument("--same-user", action="store_true", help="Only run same-user few-shot and merge into existing results")
    parser.add_argument("--slice", default=str(ROOT / "data" / "processed" / "test_slice.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "data" / "eval" / "gemini_ladder.json"))
    args = parser.parse_args()

    if args.same_user:
        _run_same_user(args)
        return

    records, shots = _materialize_slice(args.sessions, args.max_examples, args.shots)
    print(
        f"slice: {len(records)} examples, {len({r['session_id'] for r in records})} sessions; "
        f"{len(shots)} few-shot examples from train",
        flush=True,
    )

    oracle_preds = [json.dumps(row["gold_action"], ensure_ascii=False) for row in records]
    oracle = evaluate_predictions(records, oracle_preds)
    print(format_report(oracle, "oracle (must be 100%)"))
    if oracle.session_macro_accuracy < 1.0:
        raise SystemExit("oracle failed — converter/scorer bug, stop")

    fewshot_system = TAGS_SYSTEM + fewshot_block(shots)
    ladders = [
        ("vanilla", VANILLA_SYSTEM),
        ("tags", TAGS_SYSTEM),
        ("fewshot", fewshot_system),
    ]
    results = {"oracle": oracle.as_dict(), "slice": _slice_meta(records), "ladders": []}
    for name, system in ladders:
        t0 = time.perf_counter()
        preds = _generate_gemini(records, system, workers=args.workers)
        elapsed = time.perf_counter() - t0
        scored = evaluate_predictions(records, preds)
        print(format_report(scored, f"Gemini 2.5 Flash — {name} ({elapsed:.1f}s)"))
        results["ladders"].append(_ladder_row(name, elapsed, scored, records, preds))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


def _ladder_row(name, elapsed, scored, records, preds):
    return {
        "name": name,
        "elapsed_s": elapsed,
        "metrics": scored.as_dict(),
        "preds": [
            {
                "session_id": row["session_id"],
                "gold": row["gold_action"],
                "pred": parse_action(pred).to_dict() if parse_action(pred) else pred[:120],
                "hit": actions_equal(
                    parse_action(pred),
                    Action(**row["gold_action"]),
                ),
            }
            for row, pred in zip(records, preds)
        ],
    }


def _run_same_user(args) -> None:
    from opera_repro.converter import iter_jsonl

    slice_path = Path(args.slice)
    if not slice_path.exists():
        raise SystemExit(f"missing slice {slice_path} — run the full ladder first")
    records = list(iter_jsonl(str(slice_path)))
    train_examples = _load_train()
    train_by_user: dict[str, list] = {}
    for row in train_examples:
        train_by_user.setdefault(user_key(row["session_id"]), []).append(row)
    users = {user_key(r["session_id"]) for r in records}
    print(
        f"same-user few-shot on {len(records)} examples / {len(users)} users",
        flush=True,
    )

    systems = []
    shot_meta = []
    for row in records:
        user = user_key(row["session_id"])
        shots = _pick_same_user_shots(
            train_by_user.get(user, []), user, row["session_id"], args.shots
        )
        systems.append(TAGS_SYSTEM + fewshot_block(shots))
        shot_meta.append(
            {
                "user": user,
                "n_shots": len(shots),
                "shot_sessions": [s["session_id"] for s in shots],
                "shot_golds": [s["gold_action"] for s in shots],
            }
        )
    n_with_shots = sum(1 for m in shot_meta if m["n_shots"])
    print(f"  examples with same-user train shots: {n_with_shots}/{len(records)}", flush=True)
    print(f"  first example shots: {shot_meta[0]['shot_golds']}", flush=True)

    oracle_preds = [json.dumps(row["gold_action"], ensure_ascii=False) for row in records]
    oracle = evaluate_predictions(records, oracle_preds)
    print(format_report(oracle, "oracle (must be 100%)"))
    if oracle.session_macro_accuracy < 1.0:
        raise SystemExit("oracle failed — converter/scorer bug, stop")

    t0 = time.perf_counter()
    preds = _generate_gemini_per_system(records, systems, workers=args.workers)
    elapsed = time.perf_counter() - t0
    scored = evaluate_predictions(records, preds)
    print(format_report(scored, f"Gemini 2.5 Flash — fewshot_same_user ({elapsed:.1f}s)"))

    out = Path(args.out)
    if out.exists():
        results = json.loads(out.read_text(encoding="utf-8"))
    else:
        results = {"oracle": oracle.as_dict(), "slice": _slice_meta(records), "ladders": []}
    results["ladders"] = [row for row in results.get("ladders", []) if row.get("name") != "fewshot_same_user"]
    row = _ladder_row("fewshot_same_user", elapsed, scored, records, preds)
    row["shot_meta"] = {
        "n_with_shots": n_with_shots,
        "users": sorted(users),
        "example_shots": shot_meta[0],
    }
    results["ladders"].append(row)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


def _load_train():
    from opera_repro.converter import iter_jsonl

    train_path = ROOT / "data" / "processed" / "train.jsonl"
    if train_path.exists():
        print(f"loading {train_path}…", flush=True)
        return list(iter_jsonl(str(train_path)))

    from datasets import load_dataset

    cfg = converter_config_from_yaml(load_config())
    cache = str(ROOT / "data" / "cache")
    print("loading OPeRA train for same-user shots…", flush=True)
    train_ds = load_dataset("NEU-HAI/OPeRA", "filtered_action", split="train", cache_dir=cache)

    rows = []
    for raw in train_ds:
        row = {field: raw.get(field) for field in KEEP_FIELDS}
        row["hf_split"] = "train"
        rows.append(row)
    return convert_rows(rows, cfg)["train"]


def _slice_meta(records):
    types: dict[str, int] = {}
    for row in records:
        t = row["gold_action"]["type"]
        types[t] = types.get(t, 0) + 1
    return {
        "n_examples": len(records),
        "n_sessions": len({r["session_id"] for r in records}),
        "by_type": types,
    }


def _materialize_slice(max_sessions: int, max_examples: int, n_shots: int):
    from datasets import load_dataset

    cfg = converter_config_from_yaml(load_config())
    cache = str(ROOT / "data" / "cache")
    print("loading OPeRA filtered_action…", flush=True)
    test_ds = load_dataset("NEU-HAI/OPeRA", "filtered_action", split="test", cache_dir=cache)
    train_ds = load_dataset("NEU-HAI/OPeRA", "filtered_action", split="train", cache_dir=cache)

    def to_rows(ds, split: str):
        rows = []
        for raw in ds:
            row = {field: raw.get(field) for field in KEEP_FIELDS}
            row["hf_split"] = split
            rows.append(row)
        return rows

    test_examples = convert_rows(to_rows(test_ds, "test"), cfg)["test"]
    train_examples = convert_rows(to_rows(train_ds, "train"), cfg)["train"]
    slice_rows = _take_sessions(test_examples, max_sessions, max_examples)
    write_jsonl(ROOT / "data" / "processed" / "test_slice.jsonl", slice_rows)
    shots = _pick_shots(train_examples, n_shots)
    return slice_rows, shots


def _take_sessions(records, max_sessions: int, max_examples: int):
    by_sid: OrderedDict[str, list] = OrderedDict()
    for row in records:
        by_sid.setdefault(row["session_id"], []).append(row)
    out: list = []
    n_sess = 0
    for rows in by_sid.values():
        if n_sess >= max_sessions:
            break
        if out and len(out) + len(rows) > max_examples:
            break
        out.extend(rows)
        n_sess += 1
    return out


def user_key(session_id: str) -> str:
    sid = session_id or ""
    return sid.split("_20")[0] if "_20" in sid else sid.split("_")[0]


def _pick_same_user_shots(train_rows, user: str, exclude_session: str, n_shots: int):
    wanted = ["type_and_submit", "click", "terminate"]
    pool = [
        row
        for row in train_rows
        if user_key(row["session_id"]) == user and row["session_id"] != exclude_session
    ]
    picked, seen = [], set()
    for row in pool:
        kind = row["gold_action"]["type"]
        if kind in seen or kind not in wanted:
            continue
        if len(row.get("prompt_text") or "") > 8000:
            continue
        picked.append(row)
        seen.add(kind)
        if len(picked) >= n_shots:
            break
    if len(picked) < n_shots:
        for row in pool:
            if row in picked:
                continue
            picked.append(row)
            if len(picked) >= n_shots:
                break
    return picked


def _pick_shots(train_rows, n_shots: int):
    wanted = ["type_and_submit", "click", "terminate"]
    picked = []
    seen_types = set()
    for row in train_rows:
        kind = row["gold_action"]["type"]
        if kind in seen_types or kind not in wanted:
            continue
        if len(row.get("prompt_text") or "") > 2500:
            continue
        picked.append(row)
        seen_types.add(kind)
        if len(picked) >= n_shots:
            break
    return picked


def _generate_gemini(records, system: str, workers: int) -> list[str]:
    return _generate_gemini_per_system(records, [system] * len(records), workers)


def _generate_gemini_per_system(records, systems: list[str], workers: int) -> list[str]:
    from simulator.llm import MODEL, complete_json

    print(
        f"Gemini {MODEL} workers={workers} system_chars={len(systems[0]) if systems else 0}",
        flush=True,
    )
    preds = [""] * len(records)

    def one(i: int, row: dict) -> tuple[int, str]:
        user = next(m["content"] for m in row["messages"] if m["role"] == "user")
        last_err = ""
        for attempt in range(6):
            try:
                parsed = complete_json(
                    systems[i],
                    user,
                    temperature=0.0,
                    max_output_tokens=128,
                    thinking_budget=0,
                    verbose=False,
                )
                return i, json.dumps(parsed, ensure_ascii=False) if parsed else ""
            except Exception as exc:
                last_err = str(exc)
                if "429" in last_err or "500" in last_err or "503" in last_err:
                    time.sleep(min(2**attempt, 16) + random.random())
                    continue
                break
        print(f"  fail [{i}] {last_err[:160]}", flush=True)
        return i, ""

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(one, i, row) for i, row in enumerate(records)]
        for fut in as_completed(futs):
            i, text = fut.result()
            preds[i] = text
            done += 1
            parsed = parse_action(text)
            gold = records[i]["gold_action"]
            hit = actions_equal(parsed, Action(**gold))
            if done == len(records) or done % 10 == 0:
                print(
                    f"  [{done}/{len(records)}] last={'HIT' if hit else 'MISS'} "
                    f"gold={gold} pred={parsed.to_dict() if parsed else text[:60]!r}",
                    flush=True,
                )
    return preds


if __name__ == "__main__":
    main()
