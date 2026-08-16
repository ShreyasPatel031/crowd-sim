"""Pioneer open-weight inference: verdict extraction + GLiNER2 labels."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

PIONEER_BASE = "https://api.pioneer.ai"
DECODER_MODEL = os.environ.get(
    "PIONEER_DECODER_MODEL",
    "Qwen/Qwen3-8B",
)
GLINER_MODEL = os.environ.get("PIONEER_GLINER_MODEL", "fastino/gliner2-base-v1")
PII_MODEL = os.environ.get(
    "PIONEER_PII_MODEL",
    "fastino/gliner2-privacy-filter-PII-multi",
)


def pioneer_available() -> bool:
    return bool(os.environ.get("PIONEER_API_KEY"))


def _headers() -> dict[str, str]:
    key = os.environ.get("PIONEER_API_KEY") or ""
    return {
        "Content-Type": "application/json",
        "X-API-Key": key,
        "Authorization": f"Bearer {key}",
    }


def _post(path: str, payload: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
    req = urllib.request.Request(
        f"{PIONEER_BASE}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers=_headers(),
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Pioneer {exc.code}: {body[:400]}") from exc


def chat(messages: list[dict[str, str]], *, model: str | None = None) -> str:
    data = _post(
        "/v1/chat/completions",
        {
            "model": model or DECODER_MODEL,
            "messages": messages,
            "temperature": 0.1,
            "store": False,
        },
    )
    choices = data.get("choices") or []
    if not choices:
        return ""
    return str((choices[0].get("message") or {}).get("content") or "")


def extract_verdict_text(transcript: str) -> str:
    if not pioneer_available() or not transcript.strip():
        return ""
    return chat(
        [
            {
                "role": "system",
                "content": (
                    "Extract the shopper's final decision as JSON only. "
                    'Keys: verdict (buy|maybe|no), product_selected, product_url, '
                    "confidence (0-100), rationale, price_perception, "
                    "trust_concerns (array), conversion_blockers (array)."
                ),
            },
            {"role": "user", "content": transcript[:12000]},
        ]
    )


def classify_shopper_text(text: str) -> dict[str, Any]:
    if not pioneer_available() or not text.strip():
        return {}
    try:
        return _post(
            "/inference",
            {
                "model_id": GLINER_MODEL,
                "text": text[:4000],
                "schema": {
                    "classifications": [
                        {"task": "verdict", "labels": ["buy", "maybe", "no"]},
                        {
                            "task": "price_feel",
                            "labels": ["cheap", "fair", "expensive"],
                        },
                    ]
                },
                "threshold": 0.3,
            },
        )
    except Exception:
        return {}


def scrub_pii(text: str) -> dict[str, Any]:
    if not pioneer_available() or not text.strip():
        return {}
    try:
        return _post(
            "/v1/chat/completions",
            {
                "model": PII_MODEL,
                "messages": [{"role": "user", "content": text[:4000]}],
                "schema": {"entities": ["person", "email", "phone_number", "address"]},
                "include_confidence": True,
            },
        )
    except Exception:
        return {}
