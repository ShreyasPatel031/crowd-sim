#!/usr/bin/env python3
"""Parallel Pioneer chat eval for OPeRA next-action JSONL."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from opera_repro.actions import Action, actions_equal, parse_action
from opera_repro.evaluate import evaluate_predictions, format_report

PIONEER_URL = "https://api.pioneer.ai/v1/chat/completions"
CHARS_PER_TOKEN = 2.6
MAX_INPUT_TOKENS = 40_960
CHAR_BUDGET = int(MAX_INPUT_TOKENS * CHARS_PER_TOKEN) - 4_000


def _load_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def _truncate_messages(messages: list[dict]) -> list[dict]:
    prompt = [m for m in messages if m.get("role") != "assistant"]
    total = sum(len(m.get("content") or "") for m in prompt)
    if total <= CHAR_BUDGET:
        return prompt
    out = []
    remaining = CHAR_BUDGET
    for msg in prompt:
        content = msg.get("content") or ""
        if len(content) <= remaining:
            out.append(msg)
            remaining -= len(content)
            continue
        # Keep the tail (current observation / candidate list).
        out.append({**msg, "content": content[-remaining:]})
        remaining = 0
        break
    return out


def _load_checkpoint(path: Path) -> dict[int, dict]:
    done: dict[int, dict] = {}
    if not path.exists():
        return done
    for line in path.open():
        row = json.loads(line)
        done[int(row["i"])] = row
    return done


def _chat(key: str, model: str, messages: list[dict], timeout: int) -> tuple[str, dict]:
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 128,
            "store": False,
        }
    ).encode()
    req = urllib.request.Request(PIONEER_URL, data=payload, method="POST")
    req.add_header("X-API-Key", key)
    req.add_header("Authorization", "Bearer " + key)
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    usage = data.get("usage") or {}
    return text, usage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(ROOT / "data/processed/test.jsonl"))
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--out", default=str(ROOT / "data/pioneer/qwen3_8b_full_eval.json"))
    parser.add_argument("--ckpt", default=str(ROOT / "data/pioneer/qwen3_8b_full_eval.ckpt.jsonl"))
    args = parser.parse_args()

    _load_env()
    key = os.environ.get("PIONEER_API_KEY") or ""
    if not key:
        raise SystemExit("PIONEER_API_KEY missing")

    records = [json.loads(line) for line in Path(args.data).open()]
    ckpt_path = Path(args.ckpt)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_checkpoint(ckpt_path)
    print(
        f"n={len(records)} model={args.model} workers={args.workers} resume={len(done)}",
        flush=True,
    )

    stop = threading.Event()
    lock = threading.Lock()
    t0 = time.time()
    prompt_tokens = 0
    completion_tokens = 0
    n_ok = 0
    n_fail = 0

    def one(i: int, row: dict) -> tuple[int, dict]:
        if stop.is_set():
            return i, {"i": i, "error": "stopped", "pred": ""}
        msgs = _truncate_messages(row["messages"])
        last_err = ""
        for attempt in range(8):
            if stop.is_set():
                return i, {"i": i, "error": "stopped", "pred": ""}
            try:
                text, usage = _chat(key, args.model, msgs, args.timeout)
                return i, {
                    "i": i,
                    "pred": text,
                    "usage": usage,
                    "error": None,
                }
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_err = f"{exc.code} {body[:240]}"
                if exc.code in (402, 403):
                    stop.set()
                    return i, {"i": i, "pred": "", "error": last_err}
                if exc.code == 429:
                    retry_after = exc.headers.get("Retry-After")
                    wait = int(retry_after) if retry_after and str(retry_after).isdigit() else min(2**attempt, 16)
                    time.sleep(wait)
                    continue
                if exc.code in (500, 502, 503, 504):
                    time.sleep(min(2**attempt, 16))
                    continue
                return i, {"i": i, "pred": "", "error": last_err}
            except Exception as exc:
                last_err = str(exc)[:240]
                time.sleep(min(2**attempt, 8))
        return i, {"i": i, "pred": "", "error": last_err}

    pending = [(i, row) for i, row in enumerate(records) if i not in done]
    with ckpt_path.open("a", encoding="utf-8") as ckpt_fh, ThreadPoolExecutor(
        max_workers=args.workers
    ) as pool:
        futures = [pool.submit(one, i, row) for i, row in pending]
        for fut in as_completed(futures):
            i, payload = fut.result()
            with lock:
                if payload.get("error") and not payload.get("pred"):
                    n_fail += 1
                else:
                    n_ok += 1
                    usage = payload.get("usage") or {}
                    prompt_tokens += int(usage.get("prompt_tokens") or 0)
                    completion_tokens += int(usage.get("completion_tokens") or 0)
                done[i] = payload
                ckpt_fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
                ckpt_fh.flush()
                finished = n_ok + n_fail
                if finished == 1 or finished % 25 == 0 or finished == len(pending):
                    elapsed = time.time() - t0
                    rate = finished / elapsed if elapsed else 0
                    print(
                        f"  {len(done)}/{len(records)} ok={n_ok} fail={n_fail} "
                        f"in_tok={prompt_tokens:,} out_tok={completion_tokens:,} "
                        f"{rate:.1f} ex/s",
                        flush=True,
                    )
            if stop.is_set() and not payload.get("pred"):
                print("stopped on billing/rate error:", payload.get("error"), flush=True)

    preds = [""] * len(records)
    scored_records = []
    scored_preds = []
    for i, row in enumerate(records):
        item = done.get(i) or {"pred": "", "error": "missing"}
        preds[i] = item.get("pred") or ""
        if item.get("pred"):
            scored_records.append(row)
            scored_preds.append(item["pred"])

    result = evaluate_predictions(records, preds)
    scored = (
        evaluate_predictions(scored_records, scored_preds).as_dict()
        if scored_records
        else None
    )
    in_cost = prompt_tokens / 1e6 * 0.20
    out_cost = completion_tokens / 1e6 * 0.20
    payload = result.as_dict()
    payload.update(
        {
            "model": args.model,
            "n_ok": n_ok,
            "n_fail": n_fail,
            "n_resume": len(records) - len(pending),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "est_cost_usd": round(in_cost + out_cost, 4),
            "elapsed_s": round(time.time() - t0, 1),
            "scored_only": scored,
            "stopped": stop.is_set(),
            "preds": [
                {
                    "session_id": row["session_id"],
                    "gold": row["gold_action"],
                    "pred": (
                        parse_action(pred).to_dict()
                        if parse_action(pred)
                        else (pred[:200] if pred else (done.get(i) or {}).get("error"))
                    ),
                    "hit": actions_equal(
                        parse_action(pred),
                        Action(**row["gold_action"]),
                    ),
                    "error": (done.get(i) or {}).get("error"),
                }
                for i, (row, pred) in enumerate(zip(records, preds))
            ],
        }
    )
    out_path = Path(args.out)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n" + format_report(result))
    if scored and scored["n_examples"] != result.n_examples:
        print(f"  scored-only examples   {scored['n_examples']}")
        print(f"  scored-only session-macro {scored['session_macro_accuracy']:.2%}")
    print(
        f"tokens in={prompt_tokens:,} out={completion_tokens:,} "
        f"est_cost=${payload['est_cost_usd']:.4f} elapsed_s={payload['elapsed_s']}"
    )
    print("wrote", out_path)


if __name__ == "__main__":
    main()
