"""Shopper model profiles for Browser Use + verdict extraction."""

from __future__ import annotations

import os
from typing import Any

PIONEER_BASE = "https://api.pioneer.ai/v1"
DEFAULT_SFT_ENDPOINT = (
    "projects/347838016394/locations/us-central1/endpoints/1173650047569494016"
)

PROFILES: dict[str, dict[str, Any]] = {
    "gemini": {
        "id": "gemini",
        "label": "Gemini 2.5 Flash",
        "hint": "Base model · 16.1% micro exact on OPeRA honest test",
        "provider": "google",
        "browser_model": "gemini-2.5-flash",
        "report_model": "gemini-2.5-flash",
        "verdict_source": "gemini",
    },
    "gemini_sft": {
        "id": "gemini_sft",
        "label": "Gemini + OPeRA SFT",
        "hint": "opera-flash-sft-e1 · 26.9% micro (+10.8 vs base)",
        "provider": "google",
        "browser_model": "gemini-2.5-flash",
        "report_model": "opera-flash-sft-e1",
        "verdict_source": "vertex_sft",
        "vertex_endpoint": os.environ.get("GEMINI_SFT_ENDPOINT", DEFAULT_SFT_ENDPOINT),
    },
    "qwen": {
        "id": "qwen",
        "label": "Qwen3-8B",
        "hint": "Pioneer hosted · 6.9% session-macro zero-shot",
        "provider": "pioneer",
        "browser_model": os.environ.get("PIONEER_DECODER_MODEL", "Qwen/Qwen3-8B"),
        "report_model": os.environ.get("PIONEER_DECODER_MODEL", "Qwen/Qwen3-8B"),
        "verdict_source": "pioneer",
    },
    "frontier": {
        "id": "frontier",
        "label": "Frontier (Pioneer auto)",
        "hint": "Routes to best Pioneer frontier model per step",
        "provider": "pioneer",
        "browser_model": "pioneer/auto",
        "report_model": "pioneer/auto",
        "verdict_source": "pioneer",
    },
}

DEFAULT_PROFILE = os.environ.get("SHOPPER_MODEL", "gemini").strip().lower() or "gemini"


def is_vercel_runtime() -> bool:
    return bool(os.environ.get("VERCEL") or os.environ.get("VERCEL_ENV") or os.environ.get("VERCEL_URL"))


def all_profiles() -> list[dict[str, Any]]:
    return list(PROFILES.values())


def get_profile(profile_id: str | None) -> dict[str, Any]:
    key = (profile_id or DEFAULT_PROFILE).strip().lower()
    return PROFILES.get(key) or PROFILES["gemini"]


def profile_available(profile_id: str | None) -> bool:
    profile = get_profile(profile_id)
    if profile["provider"] == "google":
        return bool(os.environ.get("GOOGLE_API_KEY"))
    return bool(os.environ.get("PIONEER_API_KEY"))


def panel_available_for(profile_id: str | None) -> bool:
    if is_vercel_runtime():
        return bool((os.environ.get("PANEL_WORKER_URL") or "").strip())
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    bu_python = root / ".venv-bu" / "bin" / "python"
    bu_job = root / "scripts" / "run_browser_use_job.py"
    if not bu_python.exists() or not bu_job.exists():
        return False
    return profile_available(profile_id)


def build_browser_llm(profile_id: str | None):
    profile = get_profile(profile_id)
    if profile["provider"] == "google":
        from browser_use import ChatGoogle

        return ChatGoogle(model=profile["browser_model"])
    from browser_use import ChatOpenAI

    key = os.environ.get("PIONEER_API_KEY") or ""
    if not key:
        raise RuntimeError("PIONEER_API_KEY is missing for this model profile")
    return ChatOpenAI(
        model=profile["browser_model"],
        api_key=key,
        base_url=PIONEER_BASE,
        temperature=0.2,
    )
