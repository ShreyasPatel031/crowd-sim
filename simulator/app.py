"""Crowd Sim — shopper panel + walkthrough app."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import uuid
import urllib.parse
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from simulator.public_urls import validate_public_url
from simulator.worker_proxy import proxy_form_post, proxy_get, uses_panel_worker
from simulator.personas import all_personas, all_tasks, get_personas, get_task
from simulator.model_profiles import (
    DEFAULT_PROFILE,
    all_profiles,
    get_profile,
    is_vercel_runtime,
    panel_available_for,
    profile_available,
)
from simulator.pioneer import pioneer_available
from simulator.stripe_pay import payment_link, stripe_configured
from simulator.terac import terac_available
from simulator.verdict import aggregate_panel, normalize_verdict, prepare_panel_report

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("SIM_RUNS_DIR") or ROOT / "data" / "sim_runs")
TEMPLATES = ROOT / "simulator" / "templates"
BU_PYTHON = ROOT / ".venv-bu" / "bin" / "python"
PANEL_PYTHON = ROOT / ".venv" / "bin" / "python"
BU_JOB = ROOT / "scripts" / "run_browser_use_job.py"
PANEL_JOB = ROOT / "scripts" / "run_panel_job.py"
DEFAULT_MAX_STEPS = 30

app = FastAPI(title="Crowd Sim", version="0.2.0")
templates = Jinja2Templates(directory=str(TEMPLATES))
RUNS.mkdir(parents=True, exist_ok=True)
app.mount("/runs", StaticFiles(directory=str(RUNS)), name="runs")


def _run_dir(run_id: str) -> Path:
    return RUNS / run_id


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def browser_use_available() -> bool:
    return profile_available(DEFAULT_PROFILE) and BU_PYTHON.exists() and BU_JOB.exists()


def panel_available() -> bool:
    return panel_available_for(DEFAULT_PROFILE)


def _touch_progress(out_dir: Path, **fields) -> None:
    path = out_dir / "progress.json"
    data = {}
    if path.exists():
        try:
            data = _read_json(path)
        except json.JSONDecodeError:
            data = {}
    data.update(fields)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _mark_progress_error(out_dir: Path, message: str) -> None:
    _touch_progress(out_dir, status="error", message=message, error=message)


def _spawn_detached(cmd: list[str], log_path: Path) -> subprocess.Popen:
    """Start a worker outside the API process group so reloads do not kill it."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "ab")
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=os.environ.copy(),
            start_new_session=True,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            close_fds=True,
        )
    except Exception:
        log_f.close()
        raise
    log_f.close()
    return proc


def _split_urls(raw: str) -> list[str]:
    urls: list[str] = []
    for line in (raw or "").replace(",", "\n").splitlines():
        item = line.strip()
        if not item:
            continue
        try:
            urls.append(validate_public_url(item))
        except ValueError:
            continue
        if len(urls) >= 4:
            break
    return urls


def _listing_label(index: int, role: str, title: str = "", url: str = "") -> str:
    if role == "product":
        return title or "Your listing"
    return title or f"Competitor {index}" or url


def _read_listing_session(child_id: str, index: int, url: str, role: str) -> dict[str, Any]:
    child_dir = _run_dir(child_id)
    progress_path = child_dir / "progress.json"
    report_path = child_dir / "report.json"
    child: dict[str, Any] = {}
    if progress_path.exists():
        try:
            child = _read_json(progress_path)
        except json.JSONDecodeError:
            child = {}
    report: dict[str, Any] = {}
    if report_path.exists():
        try:
            report = _read_json(report_path)
        except json.JSONDecodeError:
            report = {}
    progress_steps = child.get("steps") or []
    report_steps = report.get("steps") or []
    steps = report_steps if len(report_steps) > len(progress_steps) else progress_steps
    last = next((s for s in reversed(steps) if s.get("screenshot")), None)
    title = child.get("current_title") or (last or {}).get("title") or ""
    status = child.get("status") or ("done" if report else ("queued" if not steps else "running"))
    return {
        "index": index,
        "role": role,
        "url": url or child.get("url") or child.get("current_url") or "",
        "label": _listing_label(index, role, title, url),
        "title": title,
        "status": status,
        "message": child.get("message") or "",
        "run_id": child_id,
        "steps": steps,
        "screenshot": f"{child_id}/{last['screenshot']}" if last else "",
        "thinking": ((last or {}).get("decision") or {}).get("reason")
        or ((last or {}).get("decision") or {}).get("memory")
        or "",
        "verdict": report.get("verdict"),
    }


def _panel_items(progress: dict) -> list[dict[str, Any]]:
    product = progress.get("product_url") or ""
    comps = [u for u in (progress.get("competitor_urls") or []) if u and u != product]
    items = [{"index": 0, "role": "product", "url": product, "label": "Your listing"}]
    for i, url in enumerate(comps, start=1):
        items.append({"index": i, "role": "competitor", "url": url, "label": f"Competitor {i}"})
    return items


def _enrich_panel_progress(panel_id: str, progress: dict) -> dict:
    items = _panel_items(progress)
    progress["items"] = items
    shoppers = list(progress.get("shoppers") or [])
    if not shoppers:
        for persona in progress.get("personas") or []:
            pid = persona.get("id") or ""
            shoppers.append(
                {
                    "id": pid,
                    "name": persona.get("name") or "",
                    "label": persona.get("label") or "",
                    "status": "queued",
                    "run_id": f"{panel_id}_{pid}" if pid else "",
                }
            )
    for shopper in shoppers:
        run_id = shopper.get("run_id") or (f"{panel_id}_{shopper.get('id')}" if shopper.get("id") else "")
        shopper["run_id"] = run_id
        listings = []
        for item in items:
            child_id = f"{run_id}_L{item['index']}"
            listings.append(
                _read_listing_session(child_id, item["index"], item.get("url") or "", item.get("role") or "competitor")
            )
        shopper["listings"] = listings
        child_progress = _run_dir(run_id) / "progress.json"
        child_report = _run_dir(run_id) / "report.json"
        if child_progress.exists():
            try:
                child = _read_json(child_progress)
            except json.JSONDecodeError:
                child = {}
            shopper["message"] = child.get("message") or shopper.get("message") or ""
            if child.get("status") == "error":
                shopper["status"] = "error"
            elif child.get("status") == "done":
                shopper["status"] = "done"
        if child_report.exists():
            try:
                report = _read_json(child_report)
            except json.JSONDecodeError:
                report = {}
            shopper["status"] = shopper.get("status") or "done"
            if report.get("verdict"):
                shopper["verdict"] = report["verdict"]
        running_listings = sum(1 for lst in listings if lst.get("status") == "running")
        done_listings = sum(1 for lst in listings if lst.get("status") in ("done", "error"))
        if shopper.get("status") not in ("error", "done"):
            if running_listings or any(lst.get("steps") for lst in listings):
                shopper["status"] = "running"
            elif done_listings == len(listings) and listings:
                shopper["status"] = "done"
        if running_listings:
            shopper["message"] = f"{running_listings} of {len(listings)} listings browsing"
        elif done_listings == len(listings) and listings:
            shopper["message"] = shopper.get("message") or "Finished"
    for item in items:
        titles = [
            lst.get("title") or ""
            for shopper in shoppers
            for lst in (shopper.get("listings") or [])
            if lst.get("index") == item["index"] and lst.get("title")
        ]
        if titles:
            item["title"] = titles[0]
            item["label"] = _listing_label(item["index"], item.get("role") or "", titles[0], item.get("url") or "")
    progress["shoppers"] = shoppers
    running = sum(1 for s in shoppers if s.get("status") == "running")
    done = sum(1 for s in shoppers if s.get("status") in ("done", "error"))
    browsers = sum(1 for s in shoppers for lst in (s.get("listings") or []) if lst.get("status") == "running")
    verdict_done = [s for s in shoppers if s.get("verdict")]
    if verdict_done:
        progress["summary"] = aggregate_panel([{"verdict": s["verdict"]} for s in verdict_done])
    if browsers:
        progress["message"] = f"{browsers} browsers running in parallel"
    elif running:
        progress["message"] = f"{running} shopper{'s' if running != 1 else ''} judging listings in parallel"
    elif done and progress.get("status") != "done":
        progress["message"] = f"{done} of {len(shoppers)} finished"
    return progress


def _enrich_panel_report(panel_id: str, report: dict[str, Any]) -> dict[str, Any]:
    """Attach panel items + per-shopper listing walkthroughs from child runs."""
    fake: dict[str, Any] = {
        "product_url": report.get("product_url") or "",
        "competitor_urls": report.get("competitor_urls") or [],
        "shoppers": [],
    }
    for row in report.get("shoppers") or []:
        persona = row.get("persona") or {}
        pid = persona.get("id") or ""
        fake["shoppers"].append(
            {
                "id": pid,
                "name": persona.get("name") or "",
                "label": persona.get("label") or "",
                "status": row.get("status") or "done",
                "run_id": row.get("run_id") or (f"{panel_id}_{pid}" if pid else ""),
            }
        )
    enriched = _enrich_panel_progress(panel_id, fake)
    listings_by_run = {
        s.get("run_id") or "": s.get("listings") or [] for s in enriched.get("shoppers") or []
    }
    shoppers_out: list[dict[str, Any]] = []
    for row in report.get("shoppers") or []:
        persona = row.get("persona") or {}
        run_id = row.get("run_id") or f"{panel_id}_{persona.get('id')}"
        shoppers_out.append({**row, "listings": listings_by_run.get(run_id) or []})
    return {**report, "items": enriched.get("items") or [], "shoppers": shoppers_out}


def _home_context(request: Request, **extra) -> dict:
    from simulator.persona_search import list_example_products

    people = all_personas()
    examples = list_example_products()
    example_ids = {pid for ex in examples for pid in (ex.get("seed_ids") or [])}
    example_task_ids = {ex.get("task_id") for ex in examples if ex.get("task_id")}
    shown_people = [p for p in people if p["id"] in example_ids]
    for person in people:
        if person["id"] not in example_ids and len(shown_people) < 16:
            shown_people.append(person)
    tasks = [t for t in all_tasks() if t["id"] in example_task_ids]
    for task in all_tasks():
        if task["id"] not in example_task_ids and len(tasks) < 40:
            tasks.append(task)
    selected = extra.get("selected") or [p["id"] for p in shown_people[:1]]
    model_profile = extra.get("model_profile") or DEFAULT_PROFILE
    on_vercel = is_vercel_runtime()
    panel_ready = panel_available_for(model_profile)
    return {
        "request": request,
        "ready": panel_ready if not on_vercel else profile_available(model_profile) or panel_ready,
        "panel_ready": panel_ready,
        "uses_worker": uses_panel_worker(),
        "walkthrough_ready": browser_use_available(),
        "pioneer": pioneer_available(),
        "terac": terac_available(),
        "model_profiles": all_profiles(),
        "model_profile": get_profile(model_profile),
        "selected_model": model_profile,
        "personas": shown_people,
        "tasks": tasks,
        "examples": examples,
        "selected": selected,
        "selected_task": extra.get("selected_task") or "",
        "stripe_link": payment_link(),
        "stripe_ready": stripe_configured(),
        "error": extra.get("error"),
        "url": extra.get("url"),
        "competitors": extra.get("competitors"),
        "brief": extra.get("brief"),
    }


async def _run_job(run_id: str, url: str, intent: str, headed: bool, wait: bool = True) -> None:
    out_dir = _run_dir(run_id)
    if not browser_use_available():
        _mark_progress_error(
            out_dir,
            "Browser Use unavailable. Set GOOGLE_API_KEY and install .venv-bu with browser-use.",
        )
        return
    cmd = [
        str(BU_PYTHON),
        str(BU_JOB),
        "--run-id",
        run_id,
        "--url",
        url,
        "--intent",
        intent,
        "--max-steps",
        str(DEFAULT_MAX_STEPS),
    ]
    if headed:
        cmd.append("--headed")
    proc = _spawn_detached(cmd, out_dir / "worker.log")
    if not wait:
        await asyncio.sleep(0.4)
        code = proc.poll()
        if code is not None:
            tail = (out_dir / "worker.log").read_text(encoding="utf-8", errors="replace")[-800:]
            _mark_progress_error(out_dir, f"Worker exited immediately ({code}). {tail}")
        return
    code = await asyncio.to_thread(proc.wait)
    if code != 0 and not (out_dir / "report.json").exists():
        progress_path = out_dir / "progress.json"
        if progress_path.exists() and _read_json(progress_path).get("status") == "error":
            return
        _mark_progress_error(out_dir, f"Browser Use worker exited with code {code}")


async def _run_panel(
    panel_id: str,
    url: str,
    brief: str,
    personas: list[str],
    competitors: str,
    headed: bool,
    public_base: str,
    query: str = "",
    wait: bool = True,
    model_profile: str = DEFAULT_PROFILE,
) -> None:
    out_dir = _run_dir(panel_id)
    if not panel_available_for(model_profile):
        if is_vercel_runtime():
            _mark_progress_error(
                out_dir,
                "Live browser panels need a worker machine. Set PANEL_WORKER_URL or run locally on port 8000.",
            )
        else:
            _mark_progress_error(
                out_dir,
                f"Model profile {model_profile!r} unavailable. Check API keys and .venv-bu.",
            )
        return
    cmd = [
        str(BU_PYTHON),
        str(PANEL_JOB),
        "--panel-id",
        panel_id,
        "--url",
        url,
        "--brief",
        brief,
        "--personas",
        ",".join(personas),
        "--competitors",
        competitors,
        "--query",
        query,
        "--model-profile",
        model_profile,
    ]
    if headed:
        cmd.append("--headed")
    if public_base:
        cmd.extend(["--public-base", public_base])
    cmd.extend(["--max-steps", os.environ.get("PANEL_MAX_STEPS", "10")])
    proc = _spawn_detached(cmd, out_dir / "worker.log")
    if not wait:
        await asyncio.sleep(0.4)
        code = proc.poll()
        if code is not None:
            tail = (out_dir / "worker.log").read_text(encoding="utf-8", errors="replace")[-800:]
            _mark_progress_error(out_dir, f"Panel worker exited immediately ({code}). {tail}")
        return
    code = await asyncio.to_thread(proc.wait)
    if code != 0 and not (out_dir / "report.json").exists():
        progress_path = out_dir / "progress.json"
        if progress_path.exists() and _read_json(progress_path).get("status") == "error":
            return
        _mark_progress_error(out_dir, f"Panel worker exited with code {code}")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    ctx = _home_context(request)
    if not ctx["ready"] and not is_vercel_runtime():
        profile = ctx["model_profile"]
        ctx["error"] = ctx.get("error") or (
            f"{profile['label']} unavailable — set the API key for this model or run the worker locally."
        )
    return templates.TemplateResponse(request, "home.html", ctx)


@app.get("/pay")
async def pay():
    link = payment_link()
    if not link:
        raise HTTPException(
            status_code=503,
            detail="Stripe not configured. Set STRIPE_PAYMENT_LINK or run scripts/setup_stripe.py",
        )
    return RedirectResponse(link, status_code=302)


@app.post("/panel")
async def start_panel(
    request: Request,
    url: str = Form(...),
    brief: str = Form("Would you buy this product? Compare it to the competitors."),
    competitors: str = Form(""),
    launch_terac: str = Form(""),
    model_profile: str = Form(DEFAULT_PROFILE),
    task_id: str = Form(""),
    personas: list[str] = Form(default=[]),
):
    try:
        validate_public_url(url)
    except ValueError as exc:
        return templates.TemplateResponse(
            request,
            "home.html",
            _home_context(
                request,
                error=str(exc),
                url=url,
                competitors=competitors,
                brief=brief,
                selected=personas,
            ),
            status_code=400,
        )
    if not panel_available_for(model_profile):
        profile = get_profile(model_profile)
        msg = (
            f"{profile['label']} unavailable — "
            + (
                "set PANEL_WORKER_URL on Vercel or run locally."
                if is_vercel_runtime()
                else "set GOOGLE_API_KEY / PIONEER_API_KEY and install .venv-bu."
            )
        )
        return templates.TemplateResponse(
            request,
            "home.html",
            _home_context(
                request,
                error=msg,
                url=url,
                competitors=competitors,
                brief=brief,
                selected=personas,
                model_profile=model_profile,
            ),
            status_code=503,
        )
    if uses_panel_worker():
        status, location, _body = await asyncio.to_thread(
            proxy_form_post,
            "/panel",
            {
                "url": url,
                "brief": brief,
                "competitors": competitors,
                "launch_terac": launch_terac,
                "model_profile": model_profile,
                "task_id": task_id,
                "personas": personas,
            },
        )
        if location:
            if location.startswith("http"):
                path = urllib.parse.urlparse(location).path or "/"
            else:
                path = location
            return RedirectResponse(path, status_code=303)
        raise HTTPException(status_code=status or 502, detail="Panel worker did not redirect")
    chosen_people = get_personas(personas)
    chosen = [p["id"] for p in chosen_people]
    task = get_task(task_id)
    if task:
        brief = (brief or "").strip() or task["brief"]
        if not url or url.rstrip("/") in ("https://amazon.com", "https://www.amazon.com"):
            if task.get("asins"):
                url = f"https://www.amazon.com/dp/{task['asins'][0]}"
            else:
                url = task.get("search_url") or url
    panel_id = uuid.uuid4().hex[:10]
    out_dir = _run_dir(panel_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    competitor_urls = _split_urls(competitors)
    shoppers = [
        {
            "id": person["id"],
            "name": person["name"],
            "label": person["label"],
            "status": "queued",
            "message": "Waiting to start…",
            "run_id": f"{panel_id}_{person['id']}",
        }
        for person in chosen_people
    ]
    _touch_progress(
        out_dir,
        kind="panel",
        status="running",
        message="Starting shopper worker…",
        product_url=url,
        brief=brief,
        competitor_urls=competitor_urls,
        personas=[{"id": p["id"], "name": p["name"], "label": p["label"]} for p in chosen_people],
        shoppers=shoppers,
        model_profile=model_profile,
    )
    headed = os.environ.get("SIM_HEADED", "").lower() in ("1", "true", "yes")
    public_base = ""
    if launch_terac:
        public_base = os.environ.get("PUBLIC_BASE_URL") or str(request.base_url).rstrip("/")
    await _run_panel(
        panel_id,
        url,
        brief,
        chosen,
        competitors,
        headed,
        public_base,
        (task or {}).get("query") or "",
        False,
        model_profile,
    )
    return RedirectResponse(f"/p/{panel_id}", status_code=303)


@app.get("/competitors")
async def competitors_api(url: str, query: str = "", persona: str = ""):
    try:
        url = validate_public_url(url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if uses_panel_worker():
        params = urllib.parse.urlencode(
            {k: v for k, v in {"url": url, "query": query, "persona": persona}.items() if v}
        )
        return await asyncio.to_thread(proxy_get, f"/competitors?{params}")
    from simulator.competitors import find_competitors
    from simulator.personas import get_personas as _get

    people = _get([persona] if persona else None)
    try:
        result = await find_competitors(url, query=query or None, persona=people[0] if people else None)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not find competitors: {exc}") from exc
    return JSONResponse(result)


@app.get("/api/terac/audience")
async def terac_audience_api(q: str = "", limit: int = 8):
    from simulator.persona_search import search_personas
    from simulator.terac import find_audience_matches

    safe_limit = max(1, min(limit, 20))
    opera = search_personas(q, limit=safe_limit)
    terac = find_audience_matches(q, limit=safe_limit)
    return JSONResponse({"query": q, "opera": opera, "terac": terac})


@app.get("/api/models")
async def models_api():
    return JSONResponse(
        {
            "profiles": all_profiles(),
            "default": DEFAULT_PROFILE,
        }
    )


@app.get("/api/personas/search")
async def personas_search_api(q: str = "", limit: int = 8):
    from simulator.persona_search import search_personas

    safe_limit = max(1, min(limit, 20))
    return JSONResponse({"query": q, "results": search_personas(q, limit=safe_limit)})


@app.get("/api/personas/archetypes")
async def personas_archetypes_api():
    from simulator.persona_search import list_archetypes

    return JSONResponse({"archetypes": list_archetypes()})


@app.get("/api/examples")
async def example_products_api():
    from simulator.persona_search import list_example_products, load_example_products

    payload = load_example_products()
    return JSONResponse(
        {
            "persona_type_definitions": payload.get("persona_type_definitions") or {},
            "examples": list_example_products(),
        }
    )


@app.get("/api/personas/{persona_id}/similar")
async def personas_similar_api(persona_id: str, limit: int = 5):
    from simulator.persona_search import similar_personas

    safe_limit = max(1, min(limit, 20))
    return JSONResponse({"persona_id": persona_id, "results": similar_personas(persona_id, limit=safe_limit)})


@app.post("/simulate")
async def simulate(
    request: Request,
    url: str = Form(...),
    intent: str = Form("Explore the site and describe what you would do next."),
):
    try:
        validate_public_url(url)
    except ValueError as exc:
        return templates.TemplateResponse(
            request,
            "home.html",
            _home_context(request, error=str(exc), url=url, brief=intent),
            status_code=400,
        )
    if not browser_use_available():
        return templates.TemplateResponse(
            request,
            "home.html",
            _home_context(
                request,
                error="Browser Use unavailable. Set GOOGLE_API_KEY in .env.",
                url=url,
                brief=intent,
            ),
            status_code=503,
        )
    run_id = uuid.uuid4().hex[:10]
    out_dir = _run_dir(run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    _touch_progress(
        out_dir,
        status="running",
        message="Queued Browser Use agent…",
        url=url,
        intent=intent,
        max_steps=DEFAULT_MAX_STEPS,
        agent="browser-use",
        steps=[],
    )
    headed = os.environ.get("SIM_HEADED", "").lower() in ("1", "true", "yes")
    await _run_job(run_id, url, intent, headed, False)
    return RedirectResponse(f"/r/{run_id}", status_code=303)


@app.post("/v1/simulate")
async def simulate_api(payload: dict):
    if not browser_use_available():
        raise HTTPException(status_code=503, detail="Browser Use unavailable (GOOGLE_API_KEY / .venv-bu)")
    url = str(payload.get("url") or "")
    intent = str(payload.get("intent") or "Explore the site and describe what you would do next.")
    try:
        validate_public_url(url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    run_id = uuid.uuid4().hex[:10]
    headed = os.environ.get("SIM_HEADED", "").lower() in ("1", "true", "yes")
    await _run_job(run_id, url, intent, headed)
    report_path = _run_dir(run_id) / "report.json"
    if not report_path.exists():
        progress = _read_json(_run_dir(run_id) / "progress.json")
        raise HTTPException(status_code=500, detail=progress.get("error") or "Run failed")
    report = _read_json(report_path)
    report["id"] = run_id
    report["report_url"] = f"/r/{run_id}"
    return JSONResponse(report)


@app.get("/r/{run_id}/status")
async def run_status(run_id: str):
    if uses_panel_worker():
        return await asyncio.to_thread(proxy_get, f"/r/{run_id}/status")
    progress_path = _run_dir(run_id) / "progress.json"
    if not progress_path.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    return JSONResponse(_read_json(progress_path))


@app.get("/r/{run_id}", response_class=HTMLResponse)
async def show_report(request: Request, run_id: str):
    if uses_panel_worker():
        return await asyncio.to_thread(proxy_get, f"/r/{run_id}")
    out_dir = _run_dir(run_id)
    report_path = out_dir / "report.json"
    if report_path.exists():
        report = _read_json(report_path)
        if report.get("kind") == "panel":
            report = prepare_panel_report(report)
            report = _enrich_panel_report(run_id, report)
            return templates.TemplateResponse(
                request,
                "panel.html",
                {
                    "request": request,
                    "report": report,
                    "panel_id": run_id,
                    "stripe_link": payment_link(),
                    "stripe_ready": stripe_configured(),
                },
            )
        return templates.TemplateResponse(
            request,
            "report.html",
            {"request": request, "report": report, "run_id": run_id},
        )
    progress_path = out_dir / "progress.json"
    if progress_path.exists():
        progress = _read_json(progress_path)
        if progress.get("kind") == "panel":
            return templates.TemplateResponse(
                request,
                "panel_live.html",
                {"request": request, "panel_id": run_id, "progress": progress},
            )
        return templates.TemplateResponse(
            request,
            "live.html",
            {"request": request, "run_id": run_id, "progress": progress},
        )
    raise HTTPException(status_code=404, detail="Run not found")


@app.get("/p/{panel_id}/status")
async def panel_status(panel_id: str):
    if uses_panel_worker():
        return await asyncio.to_thread(proxy_get, f"/p/{panel_id}/status")
    progress_path = _run_dir(panel_id) / "progress.json"
    if not progress_path.exists():
        raise HTTPException(status_code=404, detail="Panel not found")
    return JSONResponse(_enrich_panel_progress(panel_id, _read_json(progress_path)))


@app.get("/p/{panel_id}", response_class=HTMLResponse)
async def show_panel(request: Request, panel_id: str):
    if uses_panel_worker():
        return await asyncio.to_thread(proxy_get, f"/p/{panel_id}")
    return await show_report(request, panel_id)


@app.get("/human/{panel_id}", response_class=HTMLResponse)
async def human_form(request: Request, panel_id: str):
    report_path = _run_dir(panel_id) / "report.json"
    progress_path = _run_dir(panel_id) / "progress.json"
    if report_path.exists():
        report = _read_json(report_path)
    elif progress_path.exists():
        progress = _read_json(progress_path)
        report = {
            "brief": progress.get("brief") or "Would you buy this product?",
            "product_url": progress.get("product_url") or "#",
        }
    else:
        raise HTTPException(status_code=404, detail="Panel not found")
    return templates.TemplateResponse(
        request,
        "human.html",
        {"request": request, "panel_id": panel_id, "report": report, "thanks": False},
    )


@app.post("/human/{panel_id}", response_class=HTMLResponse)
async def human_submit(
    request: Request,
    panel_id: str,
    verdict: str = Form(...),
    rationale: str = Form(...),
):
    out_dir = _run_dir(panel_id)
    if not out_dir.exists():
        raise HTTPException(status_code=404, detail="Panel not found")
    human_path = out_dir / "human.json"
    responses = []
    if human_path.exists():
        try:
            responses = _read_json(human_path)
        except json.JSONDecodeError:
            responses = []
    entry = normalize_verdict({"verdict": verdict, "rationale": rationale, "confidence": 100})
    responses.append(entry)
    human_path.write_text(json.dumps(responses, indent=2), encoding="utf-8")
    report_path = out_dir / "report.json"
    if report_path.exists():
        report = _read_json(report_path)
        report["human_responses"] = responses
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        payload = report
    else:
        progress = _read_json(out_dir / "progress.json") if (out_dir / "progress.json").exists() else {}
        payload = {
            "brief": progress.get("brief") or "Would you buy this product?",
            "product_url": progress.get("product_url") or "#",
        }
    return templates.TemplateResponse(
        request,
        "human.html",
        {"request": request, "panel_id": panel_id, "report": payload, "thanks": True},
    )


@app.get("/r/{run_id}/shot/{name}")
async def shot(run_id: str, name: str):
    if uses_panel_worker():
        return await asyncio.to_thread(proxy_get, f"/r/{run_id}/shot/{name}")
    path = _run_dir(run_id) / name
    if not path.exists() or path.suffix.lower() != ".png":
        raise HTTPException(status_code=404)
    return FileResponse(path)
