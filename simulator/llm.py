"""Vertex LLM: Gemini 2.5 Flash (default) or Qwen MaaS (LLM_PROVIDER=qwen)."""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from typing import Any

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "project-amer-scs-sandbox")
PROVIDER = os.environ.get("LLM_PROVIDER", "gemini").strip().lower()
if PROVIDER == "qwen":
    LOCATION = os.environ.get("QWEN_LOCATION", "us-south1")
    MODEL = os.environ.get("QWEN_MODEL", "qwen3-235b-a22b-instruct-2507-maas")
    PUBLISHER = "qwen"
else:
    LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")
    MODEL = os.environ.get("GEMINI_ENDPOINT") or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    PUBLISHER = "google"


def current_model() -> str:
    if PROVIDER == "qwen":
        return os.environ.get("QWEN_MODEL", MODEL)
    return os.environ.get("GEMINI_ENDPOINT") or os.environ.get("GEMINI_MODEL", MODEL)

LAST_LATENCY_S = 0.0
_TOKEN = ""
_TOKEN_AT = 0.0
_TOKEN_LOCK = threading.Lock()


def llm_available() -> bool:
    try:
        _access_token()
        return True
    except Exception:
        return False


def complete_json(
    system: str,
    user: str,
    image_png: bytes | None = None,
    *,
    temperature: float = 0.2,
    max_output_tokens: int = 512,
    thinking_budget: int | None = None,
    verbose: bool = True,
    response_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parts: list[dict[str, Any]] = [{"text": user}]
    if image_png and PROVIDER != "qwen":
        parts.append(
            {
                "inlineData": {
                    "mimeType": "image/png",
                    "data": base64.b64encode(image_png).decode("ascii"),
                }
            }
        )
    generation_config: dict[str, Any] = {
        "temperature": temperature,
        "responseMimeType": "application/json",
        "maxOutputTokens": max_output_tokens,
    }
    if thinking_budget is not None and PROVIDER != "qwen":
        generation_config["thinkingConfig"] = {"thinkingBudget": thinking_budget}
    if response_schema and PROVIDER != "qwen":
        generation_config["responseSchema"] = response_schema
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": generation_config,
    }
    text = _generate(payload, verbose=verbose)
    return _parse_json(text)
    parts: list[dict[str, Any]] = [{"text": user}]
    if image_png and PROVIDER != "qwen":
        parts.append(
            {
                "inlineData": {
                    "mimeType": "image/png",
                    "data": base64.b64encode(image_png).decode("ascii"),
                }
            }
        )
    generation_config: dict[str, Any] = {
        "temperature": temperature,
        "responseMimeType": "application/json",
        "maxOutputTokens": max_output_tokens,
    }
    if thinking_budget is not None and PROVIDER != "qwen":
        generation_config["thinkingConfig"] = {"thinkingBudget": thinking_budget}
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": generation_config,
    }
    text = _generate(payload, verbose=verbose)
    return _parse_json(text)


def _generate_url(model: str) -> str:
    host = "aiplatform.googleapis.com" if LOCATION == "global" else f"{LOCATION}-aiplatform.googleapis.com"
    if model.startswith("projects/"):
        return f"https://{host}/v1/{model}:generateContent"
    return (
        f"https://{host}/v1/projects/{PROJECT}"
        f"/locations/{LOCATION}/publishers/{PUBLISHER}/models/{model}:generateContent"
    )


def _generate(payload: dict[str, Any], verbose: bool = True) -> str:
    global LAST_LATENCY_S
    token = _access_token()
    model = current_model()
    url = _generate_url(model)
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode()[:800]
        raise RuntimeError(f"Vertex {PROVIDER} error {exc.code}: {detail}") from exc
    LAST_LATENCY_S = time.perf_counter() - t0
    if verbose:
        print(f"  llm {PROVIDER}/{model} {LAST_LATENCY_S:.2f}s", flush=True)
    parts = data["candidates"][0]["content"].get("parts") or []
    return "".join(p.get("text") or "" for p in parts)


def _access_token() -> str:
    global _TOKEN, _TOKEN_AT
    now = time.time()
    with _TOKEN_LOCK:
        if _TOKEN and now - _TOKEN_AT < 3000:
            return _TOKEN
        _TOKEN = subprocess.check_output(
            ["gcloud", "auth", "print-access-token"],
            text=True,
        ).strip()
        _TOKEN_AT = now
        return _TOKEN


def _parse_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}
