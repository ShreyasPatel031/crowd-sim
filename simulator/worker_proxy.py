"""Proxy live panel traffic to a long-running worker (local :8000, Railway, etc.)."""

from __future__ import annotations

import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from fastapi import HTTPException
from fastapi.responses import Response

from simulator.model_profiles import is_vercel_runtime


def panel_worker_base() -> str | None:
    base = (os.environ.get("PANEL_WORKER_URL") or "").strip().rstrip("/")
    return base or None


def uses_panel_worker() -> bool:
    return bool(panel_worker_base()) and is_vercel_runtime()


def _worker_url(path: str) -> str:
    base = panel_worker_base()
    if not base:
        raise HTTPException(status_code=503, detail="PANEL_WORKER_URL is not configured")
    if not path.startswith("/"):
        path = "/" + path
    return f"{base}{path}"


def _build_opener() -> urllib.request.OpenerDirector:
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    return urllib.request.build_opener(NoRedirect())


def _request_headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    headers = {"ngrok-skip-browser-warning": "1"}
    if extra:
        headers.update(extra)
    return headers


def proxy_get(path: str, *, timeout: float = 120.0) -> Response:
    url = _worker_url(path)
    try:
        req = urllib.request.Request(url, method="GET", headers=_request_headers())
        with _build_opener().open(req, timeout=timeout) as resp:
            body = resp.read()
            headers = {k: v for k, v in resp.headers.items() if k.lower() not in ("transfer-encoding", "connection")}
            return Response(content=body, status_code=resp.status, headers=headers, media_type=resp.headers.get_content_type())
    except urllib.error.HTTPError as exc:
        body = exc.read()
        raise HTTPException(status_code=exc.code, detail=body.decode("utf-8", errors="replace")[:500]) from exc
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=502, detail=f"Worker unreachable: {exc.reason}") from exc


def proxy_form_post(path: str, form: dict[str, Any], *, timeout: float = 120.0) -> tuple[int, str | None, bytes]:
    """POST urlencoded form; returns status, Location header, body."""
    url = _worker_url(path)
    pairs: list[tuple[str, str]] = []
    for key, value in form.items():
        if value is None:
            continue
        if isinstance(value, list):
            for item in value:
                pairs.append((key, str(item)))
        else:
            pairs.append((key, str(value)))
    data = urllib.parse.urlencode(pairs).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers=_request_headers({"Content-Type": "application/x-www-form-urlencoded"}),
    )
    try:
        with _build_opener().open(req, timeout=timeout) as resp:
            location = resp.headers.get("Location")
            return resp.status, location, resp.read()
    except urllib.error.HTTPError as exc:
        location = exc.headers.get("Location")
        body = exc.read()
        if exc.code in (301, 302, 303, 307, 308) and location:
            return exc.code, location, body
        raise HTTPException(status_code=exc.code, detail=body.decode("utf-8", errors="replace")[:500]) from exc
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=502, detail=f"Worker unreachable: {exc.reason}") from exc
