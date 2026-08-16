"""Parse a shopper's structured buy/maybe/no verdict."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

VERDICT_VALUES = ("buy", "maybe", "no")


def band_from_likelihood(likelihood: int) -> str:
    if likelihood >= 70:
        return "buy"
    if likelihood >= 40:
        return "maybe"
    return "no"


def _int_score(value: Any, default: int = 50) -> int:
    try:
        return max(0, min(100, int(float(value))))
    except (TypeError, ValueError):
        return default


def buy_likelihood_of(data: dict[str, Any] | None) -> int:
    data = data or {}
    if data.get("buy_likelihood") not in (None, ""):
        return _int_score(data.get("buy_likelihood"), 50)
    if data.get("appeal") not in (None, ""):
        return _int_score(data.get("appeal"), 50)
    verdict = str(data.get("verdict") or data.get("would_buy_this") or "").strip().lower()
    confidence = _int_score(data.get("confidence"), 50)
    if verdict == "buy":
        return confidence
    if verdict == "no":
        return max(0, 100 - confidence)
    if verdict == "maybe":
        return min(65, max(35, round(40 + (confidence - 50) * 0.3)))
    return confidence


def _clip_list(items: Any, limit: int = 6) -> list[str]:
    if not isinstance(items, list):
        if items:
            return [str(items)[:240]]
        return []
    out: list[str] = []
    for item in items:
        text = str(item).strip()
        if text:
            out.append(text[:240])
        if len(out) >= limit:
            break
    return out


def normalize_verdict(raw: dict[str, Any] | None) -> dict[str, Any]:
    data = raw or {}
    likelihood = buy_likelihood_of(data)
    verdict = str(data.get("verdict") or data.get("would_buy_this") or "").strip().lower()
    if verdict not in VERDICT_VALUES:
        if verdict in ("pass", "skip", "wouldn't", "would not"):
            verdict = "no"
        elif "maybe" in verdict or "unsure" in verdict:
            verdict = "maybe"
        else:
            verdict = band_from_likelihood(likelihood)
    confidence = _int_score(data.get("confidence"), likelihood)
    return {
        "verdict": verdict,
        "buy_likelihood": likelihood,
        "product_selected": str(data.get("product_selected") or "").strip()[:200],
        "product_url": str(data.get("product_url") or "").strip()[:500],
        "confidence": confidence,
        "rationale": str(data.get("rationale") or "").strip()[:1200],
        "price_perception": str(data.get("price_perception") or "").strip()[:400],
        "trust_concerns": _clip_list(data.get("trust_concerns")),
        "conversion_blockers": _clip_list(data.get("conversion_blockers")),
    }


def extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        return None
    blob = match.group(0)
    try:
        parsed = json.loads(blob)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        blob = re.sub(r",\s*}", "}", blob)
        blob = re.sub(r",\s*]", "]", blob)
        try:
            parsed = json.loads(blob)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None
    return None


def parse_verdict(text: str | None) -> dict[str, Any] | None:
    parsed = extract_json_object(text or "")
    if not parsed:
        return None
    if "verdict" not in parsed and "rationale" not in parsed and "buy_likelihood" not in parsed:
        return None
    return normalize_verdict(parsed)


def parse_listing_eval(text: str | None) -> dict[str, Any]:
    parsed = extract_json_object(text or "") or {}
    likelihood = buy_likelihood_of(parsed)
    would = str(parsed.get("would_buy_this") or parsed.get("verdict") or "").strip().lower()
    if would not in VERDICT_VALUES:
        would = band_from_likelihood(likelihood)
    return {
        "would_buy_this": would,
        "verdict": would,
        "buy_likelihood": likelihood,
        "appeal": _int_score(parsed.get("appeal"), likelihood),
        "listing_title": str(parsed.get("listing_title") or parsed.get("product_selected") or "").strip()[:200],
        "confidence": _int_score(parsed.get("confidence"), likelihood),
        "rationale": str(parsed.get("rationale") or "").strip()[:1200],
        "price_perception": str(parsed.get("price_perception") or "").strip()[:400],
        "trust_concerns": _clip_list(parsed.get("trust_concerns")),
        "conversion_blockers": _clip_list(parsed.get("conversion_blockers")),
        "url": str(parsed.get("url") or "").strip()[:500],
    }


def _real_competitors(product_url: str, product: dict[str, Any], competitors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from simulator.competitors import extract_asin, same_listing

    seed_title = product.get("listing_title") or product.get("product_selected") or ""
    seed_asin = extract_asin(product_url) or extract_asin(str(product.get("url") or ""))
    out: list[dict[str, Any]] = []
    for item in competitors:
        if not item:
            continue
        cand_title = item.get("listing_title") or item.get("product_selected") or ""
        cand_asin = extract_asin(str(item.get("url") or ""))
        if seed_asin and cand_asin and seed_asin == cand_asin:
            continue
        if seed_title and cand_title and same_listing(seed_title, "", cand_title, ""):
            continue
        out.append(item)
    return out


def combine_listing_evals(
    product_url: str,
    product: dict[str, Any] | None,
    competitors: list[dict[str, Any]],
) -> dict[str, Any]:
    product = product or {}
    comps = _real_competitors(product_url, product, competitors)
    product_l = buy_likelihood_of(product)
    best: dict[str, Any] | None = None
    for item in comps:
        if best is None or buy_likelihood_of(item) > buy_likelihood_of(best):
            best = item
    best_l = buy_likelihood_of(best) if best else 0
    product_name = product.get("listing_title") or "this listing"
    best_name = (best or {}).get("listing_title") or "a competitor"

    likelihood = product_l
    selected = product_name
    selected_url = product_url
    if best and best_l > product_l + 8:
        likelihood = min(product_l, max(8, product_l - (best_l - product_l)))
        selected = best_name
        selected_url = str(best.get("url") or "")
        lead = (
            f"{likelihood}% likely to buy the first listing. "
            f"Would pick {best_name} instead ({best_l}% vs {product_l}%)."
        )
    elif product_l >= 70 and product_l >= best_l - 5:
        lead = f"{likelihood}% likely to buy {product_name}."
        if best:
            lead += f" It beat {best_name} ({product_l}% vs {best_l}%)."
    else:
        lead = f"{likelihood}% likely to buy {product_name}."
        if best and best_l >= product_l:
            selected = best_name
            selected_url = str(best.get("url") or product_url)
            lead += f" Close with {best_name} ({best_l}%)."

    bits = [lead, str(product.get("rationale") or "").strip()]
    if best and best_l > product_l:
        bits.append(str(best.get("rationale") or "").strip())
    rationale = " ".join(b for b in bits if b)
    verdict = band_from_likelihood(likelihood)
    out = normalize_verdict(
        {
            "verdict": verdict,
            "buy_likelihood": likelihood,
            "product_selected": selected,
            "product_url": selected_url,
            "confidence": _int_score(product.get("confidence"), likelihood),
            "rationale": rationale,
            "price_perception": product.get("price_perception") or "",
            "trust_concerns": product.get("trust_concerns") or [],
            "conversion_blockers": product.get("conversion_blockers") or [],
        }
    )
    out["source"] = "parallel"
    return out


def _panel_decision(avg_likelihood: int) -> tuple[str, str]:
    if avg_likelihood >= 65:
        return "likely", "Likely to convert"
    if avg_likelihood >= 40:
        return "uncertain", "Uncertain — iterate"
    return "unlikely", "Unlikely to convert"


def _panel_why(shoppers: list[dict[str, Any]], avg_likelihood: int, label: str) -> str:
    n = len(shoppers)
    who = f"{n} shopper" if n == 1 else f"{n} shoppers"
    parts = [
        f"{label}: {who} put the average chance of buying this listing at {avg_likelihood}%."
    ]
    preferred = [
        (s.get("verdict") or {}).get("product_selected") or ""
        for s in shoppers
        if (s.get("verdict") or {}).get("verdict") == "no"
    ]
    preferred = [p for p in preferred if p]
    if preferred:
        top = Counter(preferred).most_common(1)[0][0]
        parts.append(f"Several would buy {top} instead.")
    blockers: list[str] = []
    for s in shoppers:
        blockers.extend((s.get("verdict") or {}).get("conversion_blockers") or [])
    if blockers:
        uniq = list(dict.fromkeys(blockers))[:3]
        parts.append("Main blockers: " + "; ".join(uniq) + ".")
    rationales = [
        str((s.get("verdict") or {}).get("rationale") or "").strip()
        for s in shoppers
    ]
    rationales = [r for r in rationales if r]
    if rationales:
        sample = rationales[0]
        if len(sample) > 280:
            sample = sample[:277].rsplit(" ", 1)[0] + "…"
        parts.append(sample)
    return " ".join(parts)


def aggregate_panel(shoppers: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {"buy": 0, "maybe": 0, "no": 0}
    likelihoods: list[int] = []
    confidences: list[int] = []
    blockers: list[str] = []
    normalized: list[dict[str, Any]] = []
    for shopper in shoppers:
        raw = shopper.get("verdict") or {}
        verdict = normalize_verdict(raw) if raw else {}
        if verdict:
            shopper = {**shopper, "verdict": {**raw, **verdict}}
        normalized.append(shopper)
        key = (shopper.get("verdict") or {}).get("verdict")
        if key in counts:
            counts[key] += 1
        if shopper.get("verdict"):
            likelihoods.append(buy_likelihood_of(shopper["verdict"]))
        if isinstance((shopper.get("verdict") or {}).get("confidence"), int):
            confidences.append(shopper["verdict"]["confidence"])
        blockers.extend((shopper.get("verdict") or {}).get("conversion_blockers") or [])
    n = max(len(shoppers), 1)
    avg_likelihood = round(sum(likelihoods) / len(likelihoods)) if likelihoods else 0
    decision, label = _panel_decision(avg_likelihood)
    return {
        "n": len(shoppers),
        "counts": counts,
        "buy_rate": round(100 * counts["buy"] / n),
        "maybe_rate": round(100 * counts["maybe"] / n),
        "no_rate": round(100 * counts["no"] / n),
        "avg_confidence": round(sum(confidences) / len(confidences)) if confidences else None,
        "avg_buy_likelihood": avg_likelihood,
        "panel_verdict": decision,
        "panel_label": label,
        "why": _panel_why(normalized, avg_likelihood, label),
        "top_blockers": blockers[:12],
    }


def prepare_panel_report(report: dict[str, Any]) -> dict[str, Any]:
    shoppers = []
    for row in report.get("shoppers") or []:
        verdict = row.get("verdict")
        if verdict:
            row = {**row, "verdict": normalize_verdict(verdict)}
        shoppers.append(row)
    report = {**report, "shoppers": shoppers}
    report["summary"] = aggregate_panel(shoppers)
    return report
