#!/usr/bin/env python3
"""Run a panel of persona shoppers. Each shopper×listing gets its own Browser Use browser, in parallel."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from simulator.personas import get_personas
from simulator.model_profiles import get_profile
from simulator.verdict import aggregate_panel, combine_listing_evals, parse_listing_eval
from scripts.run_browser_use_job import (  # noqa: E402
    SHOPPER_MAX_STEPS,
    mark_error,
    parse_competitor_urls,
    run_browser_use_job,
    touch_progress,
    validate_public_url,
)

PANELS = ROOT / "data" / "sim_runs"
LISTING_MAX_STEPS = int(os.environ.get("PANEL_MAX_STEPS", "10"))


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _shopper_progress(personas: list[dict[str, Any]], reports: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for persona in personas:
        pid = persona["id"]
        rec = reports.get(pid) or {}
        rows.append(
            {
                "id": pid,
                "name": persona["name"],
                "label": persona["label"],
                "status": rec.get("status") or "queued",
                "message": rec.get("message") or "",
                "verdict": rec.get("verdict"),
                "run_id": rec.get("run_id"),
                "screenshot": rec.get("screenshot"),
                "steps": rec.get("steps") or [],
                "thinking": (rec.get("verdict") or {}).get("rationale") or rec.get("thinking") or "",
            }
        )
    return rows


def _eval_from_report(report: dict[str, Any], url: str) -> dict[str, Any]:
    parsed = parse_listing_eval(str(report.get("output") or ""))
    if report.get("verdict"):
        v = report["verdict"]
        if not parsed.get("rationale") and v.get("rationale"):
            parsed = parse_listing_eval(json.dumps(v))
    parsed["url"] = url
    if not parsed.get("listing_title"):
        parsed["listing_title"] = (report.get("steps") or [{}])[-1].get("title") or ""
    return parsed


def _merge_listing_steps(
    shopper_dir: Path,
    listings: list[tuple[int, str, str, str]],
) -> list[dict[str, Any]]:
    """listings: (index, url, role, child_run_id)"""
    merged: list[dict[str, Any]] = []
    n = 1
    for index, url, role, child_id in listings:
        child_dir = PANELS / child_id
        progress_path = child_dir / "progress.json"
        report_path = child_dir / "report.json"
        child: dict[str, Any] = {}
        if progress_path.exists():
            try:
                child = _read(progress_path)
            except json.JSONDecodeError:
                child = {}
        if report_path.exists() and not child.get("steps"):
            try:
                child = {**child, **_read(report_path)}
            except json.JSONDecodeError:
                pass
        prefix = "Your listing" if role == "product" else f"Competitor {index}"
        for step in child.get("steps") or []:
            shot = step.get("screenshot") or ""
            dest_name = ""
            if shot:
                src = child_dir / shot
                dest_name = f"L{index}_{shot}"
                if src.exists():
                    shutil.copy2(src, shopper_dir / dest_name)
            decision = dict(step.get("decision") or {})
            action = decision.get("action") or "Browsed"
            decision["action"] = f"{prefix}: {action}"
            merged.append(
                {
                    "step": n,
                    "url": step.get("url") or url,
                    "title": step.get("title") or "",
                    "screenshot": dest_name,
                    "decision": decision,
                }
            )
            n += 1
    return merged


async def run_panel_job(
    panel_id: str,
    product_url: str,
    brief: str,
    persona_ids: list[str],
    competitor_urls: list[str],
    *,
    max_steps: int = LISTING_MAX_STEPS,
    headed: bool = False,
    public_base: str = "",
    query: str = "",
    model_profile: str = "gemini",
) -> None:
    out_dir = PANELS / panel_id
    out_dir.mkdir(parents=True, exist_ok=True)
    personas = get_personas(persona_ids)
    profile = get_profile(model_profile)
    product_url = validate_public_url(product_url)
    shopper_state: dict[str, Any] = {
        pid: {"status": "queued", "run_id": f"{panel_id}_{pid}"} for pid in [p["id"] for p in personas]
    }
    lock = asyncio.Lock()

    def publish(**fields: Any) -> None:
        done = [s for s in shopper_state.values() if s.get("verdict")]
        summary = fields.pop("summary", None)
        if done:
            summary = aggregate_panel([{"verdict": s["verdict"]} for s in done])
        touch_progress(
            out_dir,
            kind="panel",
            product_url=product_url,
            brief=brief,
            competitor_urls=competitor_urls,
            query=query,
            personas=[{"id": p["id"], "name": p["name"], "label": p["label"]} for p in personas],
            shoppers=_shopper_progress(personas, shopper_state),
            summary=summary,
            **fields,
        )

    publish(status="running", message="Hiring shoppers…")
    if not competitor_urls:
        publish(message="Finding competitor listings…")
        try:
            from simulator.competitors import find_competitors

            found = await find_competitors(
                product_url,
                query=query or None,
                persona=personas[0] if personas else None,
            )
            competitor_urls = [item["url"] for item in found.get("competitors") or []]
            publish(competitor_urls=competitor_urls, product_title=found.get("product_title") or "")
        except Exception as exc:
            publish(message=f"Competitor search failed ({exc}). Continuing with product only.")

    all_urls = [product_url] + [u for u in competitor_urls if u != product_url]
    n_jobs = len(personas) * len(all_urls)
    publish(
        status="running",
        message=f"Opening {n_jobs} browsers in parallel ({len(personas)} shopper{'s' if len(personas) != 1 else ''} × {len(all_urls)} listing{'s' if len(all_urls) != 1 else ''})…",
    )

    async def eval_shopper(persona: dict[str, Any]) -> None:
        pid = persona["id"]
        run_id = f"{panel_id}_{pid}"
        run_dir = PANELS / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        listing_jobs = [
            (i, url, "product" if i == 0 else "competitor", f"{run_id}_L{i}")
            for i, url in enumerate(all_urls)
        ]
        stop = asyncio.Event()

        async def publish_live() -> None:
            while not stop.is_set():
                steps = _merge_listing_steps(run_dir, listing_jobs)
                running = 0
                for *_, child_id in listing_jobs:
                    path = PANELS / child_id / "progress.json"
                    if path.exists():
                        try:
                            st = _read(path).get("status")
                        except json.JSONDecodeError:
                            st = ""
                        if st == "running":
                            running += 1
                shot = ""
                if steps:
                    last = next((s for s in reversed(steps) if s.get("screenshot")), None)
                    if last:
                        shot = f"{run_id}/{last['screenshot']}"
                thinking = ""
                if steps:
                    thinking = (steps[-1].get("decision") or {}).get("reason") or ""
                touch_progress(run_dir, status="running", message=f"{running} listing browsers open", steps=steps)
                async with lock:
                    shopper_state[pid] = {
                        "status": "running",
                        "run_id": run_id,
                        "message": f"{running} listing browsers open",
                        "steps": steps,
                        "screenshot": shot,
                        "thinking": thinking,
                    }
                    publish(status="running")
                try:
                    await asyncio.wait_for(stop.wait(), timeout=0.8)
                except asyncio.TimeoutError:
                    pass

        async def run_listing(index: int, url: str, role: str, child_id: str) -> None:
            await run_browser_use_job(
                child_id,
                url,
                brief,
                max_steps=max_steps,
                headed=headed,
                persona=persona,
                competitor_urls=[],
                listing_only=True,
                model_profile=model_profile,
            )

        watcher = asyncio.create_task(publish_live())
        try:
            await asyncio.gather(
                *[run_listing(i, url, role, child_id) for i, url, role, child_id in listing_jobs]
            )
        except Exception as exc:
            stop.set()
            async with lock:
                shopper_state[pid] = {
                    "status": "error",
                    "run_id": run_id,
                    "message": str(exc)[:300],
                }
                publish(message=f"{persona['name']} failed: {exc}")
            return
        finally:
            stop.set()
            watcher.cancel()
            try:
                await watcher
            except asyncio.CancelledError:
                pass

        steps = _merge_listing_steps(run_dir, listing_jobs)
        product_eval: dict[str, Any] = {}
        comp_evals: list[dict[str, Any]] = []
        for index, url, role, child_id in listing_jobs:
            report_path = PANELS / child_id / "report.json"
            if not report_path.exists():
                continue
            try:
                report = _read(report_path)
            except json.JSONDecodeError:
                continue
            parsed = _eval_from_report(report, url)
            if role == "product":
                product_eval = parsed
            else:
                comp_evals.append(parsed)
        verdict = combine_listing_evals(product_url, product_eval, comp_evals)
        steps.append(
            {
                "step": len(steps) + 1,
                "url": product_url,
                "title": "Decision",
                "screenshot": next((s["screenshot"] for s in reversed(steps) if s.get("screenshot")), ""),
                "decision": {
                    "action": "Decided",
                    "detail": verdict.get("verdict") or "maybe",
                    "reason": verdict.get("rationale") or "",
                    "memory": verdict.get("rationale") or "",
                },
            }
        )
        shot = ""
        last = next((s for s in reversed(steps) if s.get("screenshot")), None)
        if last:
            shot = f"{run_id}/{last['screenshot']}"
        _write(
            run_dir / "report.json",
            {
                "start_url": product_url,
                "intent": brief,
                "agent": "browser-use",
                "model": profile.get("report_model") or profile.get("browser_model"),
                "model_profile": model_profile,
                "summary": verdict.get("rationale") or "",
                "output": verdict.get("rationale") or "",
                "steps": steps,
                "persona": persona,
                "competitor_urls": competitor_urls,
                "verdict": verdict,
                "listing_evals": {"product": product_eval, "competitors": comp_evals},
            },
        )
        touch_progress(run_dir, status="done", message="Finished", steps=steps, report_ready=True)
        async with lock:
            shopper_state[pid] = {
                "status": "done",
                "run_id": run_id,
                "message": "Finished",
                "verdict": verdict,
                "screenshot": shot,
                "steps": steps,
                "thinking": verdict.get("rationale") or "",
            }
            publish(status="running")

    t0 = time.monotonic()
    await asyncio.gather(*[eval_shopper(persona) for persona in personas])
    elapsed = round(time.monotonic() - t0, 1)

    shoppers = []
    for persona in personas:
        rec = shopper_state[persona["id"]]
        shoppers.append(
            {
                "persona": persona,
                "run_id": rec.get("run_id"),
                "status": rec.get("status"),
                "message": rec.get("message"),
                "verdict": rec.get("verdict"),
                "screenshot": rec.get("screenshot"),
                "steps": rec.get("steps") or [],
            }
        )
    summary = aggregate_panel([s for s in shoppers if s.get("verdict")])
    terac = {"status": "skipped", "reason": "not requested"}
    try:
        from simulator.terac import launch_listing_study, terac_available

        if terac_available() and public_base:
            task_url = f"{public_base.rstrip('/')}/human/{panel_id}"
            publish(message="Launching Terac human study…")
            terac = launch_listing_study(product_url=product_url, brief=brief, task_url=task_url, n=10)
        elif terac_available():
            terac = {"status": "skipped", "reason": "Set PUBLIC_BASE_URL so Terac participants can open the survey"}
    except Exception as exc:
        terac = {"status": "error", "error": str(exc)[:400]}

    report = {
        "kind": "panel",
        "panel_id": panel_id,
        "product_url": product_url,
        "brief": brief,
        "competitor_urls": competitor_urls,
        "summary": summary,
        "shoppers": shoppers,
        "human_responses": [],
        "terac": terac,
        "elapsed_seconds": elapsed,
        "mode": "browser-use-parallel",
        "model_profile": model_profile,
    }
    _write(out_dir / "report.json", report)
    publish(status="done", message=f"Panel complete in {elapsed}s", report_ready=True, summary=summary, terac=terac)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-id", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--brief", default="Would you buy this product? Compare it to the competitors.")
    parser.add_argument("--personas", default="")
    parser.add_argument("--competitors", default="")
    parser.add_argument("--query", default="")
    parser.add_argument("--max-steps", type=int, default=LISTING_MAX_STEPS)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--public-base", default=os.environ.get("PUBLIC_BASE_URL", ""))
    parser.add_argument("--model-profile", default=os.environ.get("SHOPPER_MODEL", "gemini"))
    args = parser.parse_args()
    out_dir = PANELS / args.panel_id
    out_dir.mkdir(parents=True, exist_ok=True)
    persona_ids = [p.strip() for p in args.personas.split(",") if p.strip()]
    try:
        await run_panel_job(
            args.panel_id,
            args.url,
            args.brief,
            persona_ids,
            parse_competitor_urls(args.competitors),
            max_steps=args.max_steps,
            headed=args.headed,
            public_base=args.public_base,
            query=args.query,
            model_profile=args.model_profile,
        )
    except Exception as exc:
        mark_error(out_dir, str(exc))
        raise SystemExit(1) from exc


if __name__ == "__main__":
    asyncio.run(main())
