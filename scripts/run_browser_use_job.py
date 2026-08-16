#!/usr/bin/env python3
"""Background worker: Browser Use + Gemini for the web UI."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from simulator.model_profiles import build_browser_llm, get_profile
from simulator.personas import shopper_task, shopper_listing_task
from simulator.verdict import parse_verdict

load_dotenv(ROOT / ".env")

DEFAULT_MAX_STEPS = 30
SHOPPER_MAX_STEPS = 18

RATIONALE_PROMPT = """
<choice_rationale>
Every step you MUST put a shopper-style choice rationale in `memory`.

`memory` format (two labeled parts, in this order):
Rationale: <why you chose THIS query, product, control, or stop — versus other options visible now>
Progress: <short progress note for later steps>

Rationale rules:
- Explain the decision, not the mechanics. Never restate the action ("type in the search bar", "click index 12", "call done").
- Compare what you picked against alternatives on the page: title match to the user request, price, ratings, reviews, brand, sponsored vs organic, position, missing info.
- If you skip something, say why (wrong category, too expensive, no drops, login wall, etc.).
- If you search, say why that query (and why you did not rephrase).
- If you scroll, say what you still needed to see before choosing.
- If you finish, say why this outcome is good enough for the request.

Examples:
Rationale: Query "ear ache drops" matches the request exactly, so there is no need to rephrase.
Progress: On Amazon home, about to search.
Rationale: eosera Ear Pain MD is actual earache drops, visible after a short scroll, and $4.70/fl oz looks fair vs nearby listings. Skipped sponsored gadgets that were not drops.
Progress: Found a matching product at a fair price; adding to cart.
Rationale: Cart already has the matching drops at a fair unit price, so the request is done.
Progress: Added eosera Ear Pain MD; finishing.
</choice_rationale>
"""


def validate_public_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw.startswith(("http://", "https://")):
        raw = "https://" + raw
    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("URL must be a public http(s) link")
    host = parsed.hostname or ""
    if host in ("localhost", "127.0.0.1") or host.endswith(".local"):
        raise ValueError("Local URLs are not allowed")
    return raw


def build_task(url: str, intent: str) -> str:
    clean_url = validate_public_url(url)
    clean_intent = (intent or "Explore the site and describe what you would do next.").strip()
    return f"Go to {clean_url}. {clean_intent} Do not log in or sign up unless absolutely required."


def load_persona(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data.get("id"):
        raise ValueError("persona file must be a JSON object with an id")
    return data


def parse_competitor_urls(raw: str | None) -> list[str]:
    urls: list[str] = []
    for line in (raw or "").replace(",", "\n").splitlines():
        item = line.strip()
        if not item:
            continue
        urls.append(validate_public_url(item))
        if len(urls) >= 4:
            break
    return urls


def touch_progress(out_dir: Path, **fields: Any) -> None:
    path = out_dir / "progress.json"
    data: dict[str, Any] = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
    data.update(fields)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def mark_error(out_dir: Path, message: str) -> None:
    touch_progress(out_dir, status="error", message=message, error=message)


def _save_screenshot(out_dir: Path, step_num: int, state: Any) -> str:
    shot_name = f"step_{step_num:02d}.png"
    dest = out_dir / shot_name
    src = getattr(state, "screenshot_path", None)
    if src and Path(src).exists():
        shutil.copy2(src, dest)
        return shot_name
    raw = getattr(state, "screenshot", None)
    if raw:
        dest.write_bytes(base64.b64decode(raw))
        return shot_name
    return ""


VIEWPORT_MARKS_JS = """(() => {
  const vw = window.innerWidth, vh = window.innerHeight;
  const skip = /log in|login|sign in|password|cookie settings|privacy|terms/i;
  const sel = [
    'a', 'button', 'input', 'textarea', 'select',
    '[role="button"]', '[role="link"]', '[role="tab"]',
    '[data-asin]', '#add-to-cart-button', '#buy-now-button',
  ].join(',');
  const seen = new Set();
  const marks = [];
  for (const el of document.querySelectorAll(sel)) {
    if (seen.has(el)) continue;
    const r = el.getBoundingClientRect();
    const style = getComputedStyle(el);
    if (r.width < 8 || r.height < 8) continue;
    if (r.bottom < 0 || r.top > vh || r.right < 0 || r.left > vw) continue;
    if (r.width * r.height > vw * vh * 0.45) continue;
    if (el.querySelector(sel)) continue;
    if (style.visibility === 'hidden' || style.display === 'none' || style.opacity === '0') continue;
    const text = (el.innerText || el.value || el.getAttribute('aria-label')
      || el.getAttribute('placeholder') || '').trim();
    if (skip.test(text)) continue;
    seen.add(el);
    marks.push({
      x: r.x, y: r.y, w: r.width, h: r.height,
      text: text.replace(/\\s+/g, ' ').slice(0, 60),
    });
  }
  marks.sort((a, b) => a.y - b.y || a.x - b.x);
  return { vw, vh, dpr: window.devicePixelRatio || 1, marks };
})()"""


def _screenshot_b64(state: Any) -> str:
    raw = getattr(state, "screenshot", None)
    if raw:
        return raw
    src = getattr(state, "screenshot_path", None)
    if src and Path(src).exists():
        return base64.b64encode(Path(src).read_bytes()).decode("utf-8")
    return ""


async def _live_viewport_marks(agent: Any) -> dict[str, Any]:
    session = agent.browser_session
    cdp = await session.get_or_create_cdp_session()
    result = await cdp.cdp_client.send.Runtime.evaluate(
        params={"expression": VIEWPORT_MARKS_JS, "returnByValue": True},
        session_id=cdp.session_id,
    )
    value = (result or {}).get("result", {}).get("value") or {}
    if not value.get("marks"):
        raise RuntimeError("no viewport marks")
    return value


def _draw_viewport_marks(image_bytes: bytes, payload: dict[str, Any]) -> bytes:
    from io import BytesIO

    from PIL import Image, ImageDraw, ImageFont

    image = Image.open(BytesIO(image_bytes)).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 12)
    except OSError:
        font = ImageFont.load_default()

    img_w, img_h = image.size
    vw = float(payload.get("vw") or img_w)
    vh = float(payload.get("vh") or img_h)
    scale_x = img_w / vw if vw else 1.0
    scale_y = img_h / vh if vh else 1.0
    yellow = (255, 212, 0, 255)
    ink = (17, 17, 17, 255)

    for i, mark in enumerate(payload.get("marks") or []):
        x1 = int(round(mark["x"] * scale_x))
        y1 = int(round(mark["y"] * scale_y))
        x2 = int(round((mark["x"] + mark["w"]) * scale_x))
        y2 = int(round((mark["y"] + mark["h"]) * scale_y))
        if x2 <= x1 or y2 <= y1:
            continue
        if x2 < 0 or y2 < 0 or x1 > img_w or y1 > img_h:
            continue
        x1 = max(0, x1 - 5)
        y1 = max(0, y1 - 5)
        x2 = min(img_w - 1, x2 + 5)
        y2 = min(img_h - 1, y2 + 5)
        draw.rectangle((x1, y1, x2, y2), outline=yellow, width=3)

        label = str(i)
        box = draw.textbbox((0, 0), label, font=font)
        tw, th = box[2] - box[0], box[3] - box[1]
        bw, bh = tw + 6, th + 2
        bx1 = min(max(x1, x2 - bw), img_w - bw)
        by1 = max(0, y1 - bh)
        draw.rectangle((bx1, by1, bx1 + bw, by1 + bh), fill=yellow)
        draw.text((bx1 + 3, by1 + 1 - box[1]), label, fill=ink, font=font)

    combined = Image.alpha_composite(image, overlay).convert("RGB")
    out = BytesIO()
    combined.save(out, format="PNG")
    return out.getvalue()


async def _save_highlighted_screenshot(
    out_dir: Path,
    step_num: int,
    state: Any,
    agent: Any,
) -> str:
    shot_name = f"step_{step_num:02d}.png"
    dest = out_dir / shot_name
    raw = b""
    if agent.browser_session:
        try:
            raw = await agent.browser_session.take_screenshot()
        except Exception:
            raw = b""
    if not raw:
        screenshot_b64 = _screenshot_b64(state)
        raw = base64.b64decode(screenshot_b64) if screenshot_b64 else b""
    if not raw:
        return _save_screenshot(out_dir, step_num, state)

    try:
        payload = await _live_viewport_marks(agent)
        dest.write_bytes(_draw_viewport_marks(raw, payload))
        return shot_name
    except Exception:
        dest.write_bytes(raw)
        return shot_name


ACTION_LABELS = {
    "click": "Clicked",
    "input": "Typed",
    "type": "Typed",
    "send_keys": "Pressed keys",
    "scroll": "Scrolled",
    "go_to_url": "Opened page",
    "navigate": "Opened page",
    "go_back": "Went back",
    "wait": "Waited",
    "done": "Finished",
    "extract": "Extracted",
    "search": "Searched",
    "select_dropdown": "Chose option",
    "dropdown_options": "Opened menu",
    "upload_file": "Uploaded",
    "switch": "Switched tab",
    "switch_tab": "Switched tab",
    "close_tab": "Closed tab",
    "think": "Thought",
}


def _human_action(name: str) -> str:
    key = (name or "act").strip()
    if key in ACTION_LABELS:
        return ACTION_LABELS[key]
    return key.replace("_", " ").strip().capitalize() or "Act"


def _clip(text: str, limit: int = 160) -> str:
    clean = " ".join(str(text).split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1] + "…"


def _action_label(output: Any) -> str:
    if not output or not getattr(output, "action", None):
        return "Thought"
    dumped = output.action[0].model_dump(exclude_none=True, mode="json")
    return _human_action(next(iter(dumped.keys()), "act"))


def _action_detail(output: Any) -> str:
    if not output or not getattr(output, "action", None):
        return ""
    dumped = output.action[0].model_dump(exclude_none=True, mode="json")
    if not dumped:
        return ""
    name = next(iter(dumped))
    params = dumped.get(name)
    if params is None or params is True:
        return ""
    if isinstance(params, dict):
        for key in ("text", "query", "url", "keys", "value", "file_name", "extracted_content"):
            if params.get(key) not in (None, "", [], {}):
                return _clip(params[key])
        skip = {"index", "xpath", "num_clicks", "new_tab", "down", "pages"}
        parts = []
        for key, value in params.items():
            if key in skip or value in (None, "", [], {}, False):
                continue
            parts.append(f"{key}={_clip(value, 80)}")
        return ", ".join(parts)
    return _clip(params)


def _extract_rationale(output: Any) -> str:
    memory = ((output.memory if output else "") or "").strip()
    match = re.search(
        r"(?is)\brationale:\s*(.+?)(?:\n\s*progress:|\Z)",
        memory,
    )
    if match:
        return _clip(match.group(1), 400)
    return ""


def _step_record(step_num: int, state: Any, output: Any, shot_name: str) -> dict[str, Any]:
    action = _action_label(output)
    evaluation = ((output.evaluation_previous_goal if output else "") or "").strip()
    memory = ((output.memory if output else "") or "").strip()
    plan = ((output.next_goal if output else "") or "").strip()
    detail = _action_detail(output)
    reason = _extract_rationale(output)
    return {
        "step": step_num,
        "url": getattr(state, "url", "") or "",
        "title": getattr(state, "title", "") or "",
        "screenshot": shot_name,
        "decision": {
            "action": action,
            "detail": detail,
            "reason": reason,
            "evaluation": evaluation,
            "memory": memory,
            "plan": plan,
            "target_text": reason,
        },
    }


def _history_to_steps(history: Any, out_dir: Path) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for i, item in enumerate(history.history, start=1):
        shot_name = _save_screenshot(out_dir, i, item.state)
        steps.append(_step_record(i, item.state, item.model_output, shot_name))
    return steps


def _build_explanation(
    intent: str,
    url: str,
    steps: list[dict[str, Any]],
    final: str | None,
    *,
    succeeded: bool | None = None,
) -> str:
    n = len(steps)
    if succeeded is True:
        lede = f"The agent finished in {n} step{'s' if n != 1 else ''}."
    elif succeeded is False:
        lede = f"The agent did not finish after {n} step{'s' if n != 1 else ''}."
    else:
        lede = f"The agent took {n} step{'s' if n != 1 else ''}."
    lines = [lede, f"Asked to {intent.strip()} starting at {url}.", ""]
    if not steps:
        lines.append("No steps were recorded.")
    for step in steps:
        decision = step.get("decision") or {}
        action = decision.get("action") or "Act"
        detail = decision.get("detail") or ""
        reason = decision.get("reason") or ""
        headline = f"{step.get('step')}. {action}"
        if detail and detail != action:
            headline += f" — {detail}"
        lines.append(headline)
        if reason:
            lines.append(f"   Why: {reason}")
    lines.append("")
    if final:
        lines.append("Outcome")
        lines.append(final.strip())
    else:
        last = (steps[-1].get("decision") or {}) if steps else {}
        lines.append("Outcome")
        lines.append(last.get("reason") or f"Stopped after {n} step(s).")
    return "\n".join(lines).strip()


def _transcript_for_verdict(steps: list[dict[str, Any]], final: str | None) -> str:
    lines: list[str] = []
    for step in steps:
        decision = step.get("decision") or {}
        lines.append(
            f"Step {step.get('step')}: {decision.get('action')} {decision.get('detail') or ''}".strip()
        )
        if decision.get("reason"):
            lines.append(f"Why: {decision['reason']}")
        if step.get("url"):
            lines.append(f"URL: {step['url']}")
    if final:
        lines.append("Final:")
        lines.append(final)
    return "\n".join(lines)


def _gemini_extract_verdict(transcript: str, *, model: str = "gemini-2.5-flash") -> dict[str, Any] | None:
    key = os.environ.get("GOOGLE_API_KEY") or ""
    if not key or not transcript.strip():
        return None
    import urllib.error
    import urllib.request

    prompt = (
        "Extract the shopper's final decision as JSON only. "
        "Keys: buy_likelihood (0-100 chance they buy THIS listing), "
        "verdict (buy if likelihood>=70, maybe if 40-69, no if <40), "
        "product_selected, product_url, confidence (0-100), rationale, price_perception, "
        "trust_concerns (array), conversion_blockers (array). "
        "Do not mark buy just because the category is useful.\n\n"
        + transcript[:12000]
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"},
    }
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None
    parts = (((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or [])
    text = "".join(str(part.get("text") or "") for part in parts)
    return parse_verdict(text)


def _vertex_sft_extract_verdict(transcript: str, endpoint: str) -> dict[str, Any] | None:
    if not transcript.strip() or not endpoint:
        return None
    try:
        import os

        prev = os.environ.get("GEMINI_ENDPOINT")
        os.environ["GEMINI_ENDPOINT"] = endpoint
        from simulator.llm import complete_json

        parsed = complete_json(
            "Extract the shopper's final decision as JSON only.",
            transcript[:12000],
            temperature=0.1,
            max_output_tokens=512,
            verbose=False,
        )
        if prev:
            os.environ["GEMINI_ENDPOINT"] = prev
        elif "GEMINI_ENDPOINT" in os.environ:
            del os.environ["GEMINI_ENDPOINT"]
        return parse_verdict(json.dumps(parsed)) if parsed else None
    except Exception:
        return None


def _resolve_verdict(
    steps: list[dict[str, Any]],
    final: str | None,
    *,
    profile_id: str = "gemini",
) -> dict[str, Any] | None:
    profile = get_profile(profile_id)
    parsed = parse_verdict(final or "")
    if parsed:
        parsed["source"] = "agent"
        return parsed
    try:
        from simulator.pioneer import extract_verdict_text, pioneer_available

        if profile.get("verdict_source") == "pioneer" and pioneer_available():
            extracted = extract_verdict_text(_transcript_for_verdict(steps, final))
            parsed = parse_verdict(extracted)
            if parsed:
                parsed["source"] = "pioneer"
                return parsed
    except Exception:
        pass
    if profile.get("verdict_source") == "vertex_sft":
        parsed = _vertex_sft_extract_verdict(
            _transcript_for_verdict(steps, final),
            str(profile.get("vertex_endpoint") or ""),
        )
        if parsed:
            parsed["source"] = "vertex_sft"
            return parsed
    parsed = _gemini_extract_verdict(
        _transcript_for_verdict(steps, final),
        model=str(profile.get("browser_model") or "gemini-2.5-flash"),
    )
    if parsed:
        parsed["source"] = "gemini"
        return parsed
    return None


def _write_report(
    out_dir: Path,
    *,
    url: str,
    intent: str,
    task: str,
    history: Any,
    steps: list[dict[str, Any]],
    persona: dict[str, Any] | None = None,
    competitor_urls: list[str] | None = None,
    profile_id: str = "gemini",
) -> dict[str, Any]:
    profile = get_profile(profile_id)
    final = history.final_result() if hasattr(history, "final_result") else None
    succeeded = history.is_successful() if hasattr(history, "is_successful") else None
    explanation = _build_explanation(intent, url, steps, final, succeeded=succeeded)
    verdict = _resolve_verdict(steps, final, profile_id=profile_id)
    report = {
        "start_url": url,
        "intent": intent,
        "task": task,
        "agent": "browser-use",
        "llm": True,
        "model": profile.get("report_model") or profile.get("browser_model"),
        "model_profile": profile_id,
        "browser": "browser-use-local-chromium",
        "summary": explanation,
        "would_prefer": explanation,
        "output": final,
        "explanation": explanation,
        "is_done": history.is_done() if hasattr(history, "is_done") else None,
        "is_success": succeeded,
        "steps": steps,
        "friction": [],
        "products_noticed": [],
        "persona": persona,
        "competitor_urls": competitor_urls or [],
        "verdict": verdict,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


async def run_browser_use_job(
    run_id: str,
    url: str,
    intent: str,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    headed: bool = False,
    persona: dict[str, Any] | None = None,
    competitor_urls: list[str] | None = None,
    listing_only: bool = False,
    model_profile: str = "gemini",
) -> None:
    profile = get_profile(model_profile)
    if profile["provider"] == "google" and not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is missing")
    if profile["provider"] == "pioneer" and not os.environ.get("PIONEER_API_KEY"):
        raise RuntimeError("PIONEER_API_KEY is missing")

    out_dir = ROOT / "data" / "sim_runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_url = validate_public_url(url)
    competitors = competitor_urls or []
    if persona and listing_only:
        task = shopper_listing_task(persona, clean_url, intent)
        intent = intent or f"Would {persona.get('name')} buy this listing?"
    elif persona:
        task = shopper_task(persona, clean_url, competitors, intent)
        intent = intent or f"Would {persona.get('name')} buy this listing?"
    else:
        task = build_task(clean_url, intent)
    steps: list[dict[str, Any]] = []
    agent_holder: dict[str, Any] = {}

    touch_progress(
        out_dir,
        status="running",
        message="Starting Browser Use agent…",
        url=clean_url,
        intent=intent,
        task=task,
        max_steps=max_steps,
        agent="browser-use",
        steps=steps,
        persona=persona,
        competitor_urls=competitors,
        model_profile=model_profile,
    )

    async def on_step(state: Any, output: Any, step_num: int) -> None:
        shot_name = await _save_highlighted_screenshot(out_dir, step_num, state, agent_holder["agent"])
        record = _step_record(step_num, state, output, shot_name)
        steps.append(record)
        action = record["decision"]["action"]
        reason = record["decision"]["reason"] or action
        touch_progress(
            out_dir,
            message=f"Step {step_num}: {action} — {reason}",
            steps=steps,
            current_url=state.url,
            current_title=state.title,
        )

    from browser_use import Agent, Browser

    llm = build_browser_llm(model_profile)
    browser = Browser(
        headless=not headed,
        highlight_elements=False,
        dom_highlight_elements=False,
        use_cloud=False,
    )
    agent = Agent(
        task=task,
        llm=llm,
        browser=browser,
        register_new_step_callback=on_step,
        extend_system_message=RATIONALE_PROMPT,
    )
    agent_holder["agent"] = agent

    touch_progress(out_dir, message="Opening browser…")
    history = await agent.run(max_steps=max_steps)
    if not steps:
        steps = _history_to_steps(history, out_dir)
    touch_progress(out_dir, message="Writing report…")
    _write_report(
        out_dir,
        url=clean_url,
        intent=intent,
        task=task,
        history=history,
        steps=steps,
        persona=persona,
        competitor_urls=competitors,
        profile_id=model_profile,
    )
    touch_progress(out_dir, status="done", message="Complete", report_ready=True)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--intent", default="Explore the site and describe what you would do next.")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--persona-file", default="")
    parser.add_argument("--competitors", default="")
    parser.add_argument("--model-profile", default=os.environ.get("SHOPPER_MODEL", "gemini"))
    args = parser.parse_args()
    out_dir = ROOT / "data" / "sim_runs" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        persona = load_persona(args.persona_file or None)
        await run_browser_use_job(
            args.run_id,
            args.url,
            args.intent,
            max_steps=args.max_steps,
            headed=args.headed,
            persona=persona,
            competitor_urls=parse_competitor_urls(args.competitors),
            model_profile=args.model_profile,
        )
    except Exception as exc:
        mark_error(out_dir, str(exc))
        raise SystemExit(1) from exc


if __name__ == "__main__":
    asyncio.run(main())
