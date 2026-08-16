"""Lightweight URL validation (no Playwright import)."""

from __future__ import annotations

from urllib.parse import urlparse, urlunparse


def normalize_amazon_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw.startswith(("http://", "https://")):
        raw = "https://" + raw
    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower()
    if host == "amazon.com":
        parsed = parsed._replace(netloc="www.amazon.com")
    elif host.endswith(".amazon.com") and not host.startswith("www."):
        # keep smile.amazon.com etc.; normalize bare regional hosts if needed
        pass
    return urlunparse(parsed)


def validate_public_url(url: str) -> str:
    raw = (url or "").strip()
    if raw.lower().startswith("file:"):
        raise ValueError("Local URLs are not allowed")
    raw = normalize_amazon_url(raw if raw.startswith(("http://", "https://")) else "https://" + raw.lstrip("/"))
    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("URL must be a public http(s) link")
    host = parsed.hostname or ""
    if host in ("localhost", "127.0.0.1") or host.endswith(".local"):
        raise ValueError("Local URLs are not allowed")
    return raw
