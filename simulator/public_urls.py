"""Lightweight URL validation (no Playwright import)."""

from __future__ import annotations

from urllib.parse import urlparse


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
