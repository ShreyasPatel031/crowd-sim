# Shopper simulation (hackathon MVP)

Paste a public URL + intent. A Playwright agent clicks through the site, screenshots each step, and writes what it would do next.

Easiest agent on this machine: **Playwright + LLM** (not `browser-use` — that needs Python 3.11; this venv is 3.10). Without `OPENAI_API_KEY` it still runs on a keyword heuristic so you can demo screenshots.

```bash
.venv/bin/pip install -r requirements-sim.txt
.venv/bin/playwright install chromium
cp .env.example .env   # add OPENAI_API_KEY for a real LLM
.venv/bin/uvicorn simulator.app:app --reload --port 8000 \
  --reload-dir simulator --reload-dir scripts \
  --reload-exclude 'data/*'
```

Open http://127.0.0.1:8000 — or:

```bash
.venv/bin/python scripts/run_sim.py --url https://books.toscrape.com --intent "cheap travel book"
```

API: `POST /v1/simulate` `{"url":"...","intent":"..."}`.

OPeRA fine-tuning is the later USP (make the path look like a real shopper). It is not in this loop yet.

# Minimal OPeRA next-action reproduction

Reproduce the *directional* ACL result: fine-tuning Qwen on OPeRA human traces should lift exact next-action accuracy from ~4% toward ~20–30%+.

This is **not** the paper's 64×H200 / 40k-context FSDP run. It is a QLoRA pipeline you can run on one 24–80GB GPU.

Paper numbers (Lu et al., ACL 2026, Table 2 on OPeRA):

| Model | Session-macro exact-match |
|---|---|
| Qwen2.5-7B base | 4.10% |
| Qwen2.5-7B fine-tuned | 32.04% |
| + reasoning | 35.14% |

Your number will not match 4.10% exactly. The split, HTML truncation, and context length are different on purpose.

## What v1 does

```
OPeRA-filtered
      ↓
session-level train/val/test split
      ↓
(history + current simplified HTML) → next action
      ↓
Qwen2.5-7B-Instruct  +  QLoRA SFT
      ↓
held-out sessions, exact next-action accuracy
```

Kept: `session_id`, simplified HTML, action type, target (`semantic_id`), input text.

Ignored for v1: persona, screenshots, human rationales, browser/Playwright.

## Exact evaluation

A prediction is correct only if **all** of these match the gold action:

1. action type (`click` / `type_and_submit` / `terminate`)
2. target name (HTML `name=` / OPeRA `semantic_id`)
3. attribute, when it exists (search text for `type_and_submit`)

Examples:

```
gold: click("reviews")      pred: click("reviews")      ✓
gold: click("reviews")      pred: click("add_to_cart")  ✗
gold: type_and_submit("nav_bar.search_input", "running shoes")
pred: type_and_submit("nav_bar.search_input", "trail shoes")   ✗
```

Unparseable output counts as wrong.

**Reported score** = mean over sessions of (correct steps / steps in that session). Long sessions do not dominate. Micro accuracy is also printed.

OPeRA `input` is mapped to the paper's `type_and_submit`.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Smoke-test the converter and metric (no GPU, no HuggingFace download):

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/prepare_data.py --fixture tests/fixtures/mini_sessions.json
.venv/bin/python scripts/eval_baseline.py --data data/processed/test.jsonl --oracle
```

The oracle run must print `session-macro exact   100.00%`.

## Day-by-day

**1. Download OPeRA-filtered** (~150MB actions, no screenshots):

```bash
.venv/bin/python scripts/prepare_data.py
```

Default split is the official HuggingFace OPeRA-test (90 sessions / 992 actions), with 10% of remaining sessions held out as val. For a fresh 80/10/10 session split, set `dataset.split_mode: random` in `configs/default.yaml`.

**2. Inspect converted examples** in `data/processed/{train,val,test}.jsonl`. Each line is:

```
INPUT:  previous (observation, action) pairs + current HTML
OUTPUT: {"type": "click", "name": "reviews"}
```

**3. Base Qwen eval** (needs a GPU + `requirements-train.txt`):

```bash
.venv/bin/pip install -r requirements-train.txt
.venv/bin/python scripts/eval_baseline.py --limit 50
.venv/bin/python scripts/eval_baseline.py
```

Expect something in the low single digits, not necessarily 4.10%.

**4. QLoRA fine-tune** (one A100/H100 80GB is plenty; 24–48GB can work):

```bash
.venv/bin/python scripts/train_qlora.py
```

Defaults: Qwen2.5-7B-Instruct, 4-bit base + bf16 LoRA, 4k context, 2 epochs, lr `2e-5`. Loss is on the action JSON, not the HTML.

**5. Evaluate the adapter:**

```bash
.venv/bin/python scripts/eval_finetuned.py --adapter outputs/qwen25-7b-opera-qlora
```

Goal: does ~4% become something closer to 20–30%+?

Stop there. Do not add reasoning, personas, or a browser agent until this lift shows up.

## Config knobs

`configs/default.yaml`

- `split_mode`: `official` (paper-like test) or `random` (80/10/10)
- `max_current_html_chars` / `max_history_html_chars`: HTML is often 70k–150k characters; we compress it
- `max_history_steps`: how much of the same session is visible
- `include_first_action`: v1 includes step 0 (it still has an observation)

## Not in v1

Reasoning-augmented SFT (paper: 32.04% → 35.14%), persona conditioning, screenshots, Playwright.
