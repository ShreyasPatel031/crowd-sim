"""Find Amazon competitor listings for a product + optional persona."""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote_plus, urlparse

ASIN_RE = re.compile(r"(?:/dp/|/gp/product/|/gp/aw/d/)([A-Z0-9]{10})", re.I)
SEARCH_RESULT_RE = re.compile(
    r'data-component-type="s-search-result"[^>]*data-asin="([A-Z0-9]{10})"[\s\S]{0,4000}?'
    r'<h2[^>]*>[\s\S]*?(?:<span[^>]*>)?\s*([^<]{8,240})\s*(?:</span>)?',
    re.I,
)
PRODUCT_TITLE_RE = re.compile(r'id="productTitle"[^>]*>\s*([^<]+?)\s*<', re.I)
OG_TITLE_RE = re.compile(r'property="og:title"\s+content="([^"]+)"', re.I)
BYLINE_RE = re.compile(r'id="bylineInfo"[^>]*>([^<]+)<', re.I)
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
CATALOG_PATH = Path(__file__).resolve().parents[1] / "data" / "opera_catalog.json"

STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "your", "you",
    "set", "pack", "size", "amazon", "brand", "visit", "store",
}

GENERIC_BRAND_WORDS = {
    "organic", "premium", "best", "new", "generic", "value", "daily",
    "extra", "strength", "triple", "mini",
}

SIZE_WORDS = {
    "mg", "mcg", "iu", "ct", "count", "oz", "lb", "ml", "fl", "capsules",
    "softgels", "tablets", "pills", "pieces", "pack", "bottle", "bottles",
}


def extract_asin(url: str) -> str:
    match = ASIN_RE.search(url or "")
    if match:
        return match.group(1).upper()
    parsed = urlparse(url or "")
    for key in ("asin", "ASIN"):
        vals = parse_qs(parsed.query).get(key) or []
        if vals and re.fullmatch(r"[A-Z0-9]{10}", vals[0].upper()):
            return vals[0].upper()
    return ""


def product_url_for_asin(asin: str) -> str:
    return f"https://www.amazon.com/dp/{asin}"


def _tokens(text: str) -> set[str]:
    return {
        w
        for w in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(w) >= 3 and w not in STOPWORDS
    }


def _normalize_brand(text: str) -> str:
    raw = (text or "").strip()
    raw = re.sub(r"(?i)^visit the\s+", "", raw)
    raw = re.sub(r"(?i)^brand:\s*", "", raw)
    raw = re.sub(r"(?i)\s+store$", "", raw)
    words = [w.lower() for w in re.findall(r"[A-Za-z0-9]+", raw) if len(w) > 1]
    words = [w for w in words if w not in STOPWORDS and w not in GENERIC_BRAND_WORDS]
    if not words:
        return ""
    if len(words) >= 2 and words[0] not in {"a", "an"}:
        return f"{words[0]} {words[1]}"
    return words[0]


def brand_from_title(title: str) -> str:
    words = [w.lower() for w in re.findall(r"[A-Za-z0-9]+", title or "") if len(w) > 1]
    words = [w for w in words if w not in STOPWORDS and w not in GENERIC_BRAND_WORDS]
    if not words:
        return ""
    if len(words) >= 2:
        return f"{words[0]} {words[1]}"
    return words[0]


@lru_cache(maxsize=1)
def _catalog() -> dict[str, Any]:
    if not CATALOG_PATH.exists():
        return {"personas": [], "tasks": []}
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def catalog_product_for_asin(asin: str) -> dict[str, str]:
    needle = (asin or "").upper()
    if not needle:
        return {}
    for task in _catalog().get("tasks") or []:
        asins = [str(a).upper() for a in (task.get("asins") or [])]
        if needle not in asins:
            continue
        idx = asins.index(needle)
        titles = task.get("example_products") or []
        title = titles[idx] if idx < len(titles) else (titles[0] if titles else "")
        return {
            "asin": needle,
            "title": title,
            "query": task.get("query") or "",
            "brand": brand_from_title(title),
        }
    return {}


def search_query_for_persona(title: str, persona: dict[str, Any] | None = None) -> str:
    text = title or ""
    asin = extract_asin(text) if "amazon." in text.lower() or "/dp/" in text.lower() else ""
    if asin:
        meta = catalog_product_for_asin(asin)
        text = meta.get("query") or meta.get("title") or ""
    words = [
        w
        for w in re.findall(r"[A-Za-z0-9]+", text)
        if len(w) > 1 and w.lower() not in STOPWORDS and w.lower() not in SIZE_WORDS and not w.isdigit()
    ]
    query = " ".join(words[:6]) or "similar product"
    if not persona:
        return query
    blob = " ".join(
        [
            str(persona.get("label") or ""),
            str(persona.get("bio") or ""),
            " ".join(persona.get("priorities") or []),
            " ".join(persona.get("avoids") or []),
            str(persona.get("budget") or ""),
        ]
    ).lower()
    if any(k in blob for k in ("value", "cheap", "budget", "under $", "/month")) and "premium brands" not in blob:
        query += " best value"
    elif "quality" in blob or "premium" in blob:
        query += " highly rated"
    return query


def resolve_search_query(
    *,
    explicit: str | None,
    meta: dict[str, str],
    seed_title: str,
    persona: dict[str, Any] | None = None,
) -> str:
    explicit_q = (explicit or "").strip()
    if explicit_q:
        return explicit_q
    title = (seed_title or meta.get("title") or "").strip()
    title_q = search_query_for_persona(title, persona)
    meta_q = (meta.get("query") or "").strip()
    if title and meta_q:
        overlap = len(_tokens(title) & _tokens(meta_q))
        if overlap < 2:
            return title_q
    return meta_q or title_q or "similar product"


def _fetch_html(url: str, *, timeout: float = 18.0) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def fetch_amazon_product_http(product_url: str) -> dict[str, str]:
    """Best-effort title + brand without Playwright."""
    asin = extract_asin(product_url)
    if not asin:
        return {}
    url = normalize_amazon_url(product_url)
    try:
        html = _fetch_html(url)
    except Exception:
        return {"asin": asin}
    if len(html) < 5000 or "captcha" in html.lower():
        return {"asin": asin}
    title_match = PRODUCT_TITLE_RE.search(html) or OG_TITLE_RE.search(html)
    title = (title_match.group(1) if title_match else "").strip()
    title = re.sub(r"\s*:\s*Amazon\.[^:]*$", "", title).strip()
    byline = BYLINE_RE.search(html)
    brand = _normalize_brand(byline.group(1) if byline else "") or brand_from_title(title)
    return {"asin": asin, "title": title, "brand": brand}


def normalize_amazon_url(url: str) -> str:
    from simulator.public_urls import normalize_amazon_url as _norm

    return _norm(url)


def competitors_from_amazon_http(
    query: str,
    skip: set[str],
    limit: int = 4,
    *,
    seed_title: str = "",
    seed_brand: str = "",
    timeout: float = 20.0,
) -> list[dict[str, str]]:
    q = (query or "").strip()
    if not q:
        return []
    search_url = "https://www.amazon.com/s?" + urllib.parse.urlencode({"k": q})
    try:
        html = _fetch_html(search_url, timeout=timeout)
    except Exception:
        return []
    if len(html) < 8000 or "captcha" in html.lower():
        return []
    out: list[dict[str, str]] = []
    seen = set(skip)
    for match in SEARCH_RESULT_RE.finditer(html):
        asin = match.group(1).upper()
        title = re.sub(r"\s+", " ", match.group(2)).strip()
        if not asin or asin in seen or asin == "0000000000":
            continue
        item = {
            "asin": asin,
            "url": product_url_for_asin(asin),
            "title": title,
            "brand": brand_from_title(title),
            "via": "amazon",
        }
        if not _keep_item(item, seen, seed_title, seed_brand):
            seen.add(asin)
            continue
        seen.add(asin)
        out.append(item)
        if len(out) >= limit:
            break
    if out:
        return out
    asins = _collect_asins(re.findall(r'data-asin="([A-Z0-9]{10})"', html), skip)
    for asin in asins:
        meta = catalog_product_for_asin(asin)
        item = {
            "asin": asin,
            "url": product_url_for_asin(asin),
            "title": meta.get("title") or "",
            "brand": meta.get("brand") or brand_from_title(meta.get("title") or ""),
            "via": "amazon",
        }
        if not _keep_item(item, seen, seed_title, seed_brand):
            seen.add(asin)
            continue
        seen.add(asin)
        out.append(item)
        if len(out) >= limit:
            break
    return out


def same_listing(seed_title: str, seed_brand: str, cand_title: str, cand_brand: str = "") -> bool:
    """True when the candidate is the same product or the same brand's sibling SKU."""
    seed_b = _normalize_brand(seed_brand) or brand_from_title(seed_title)
    cand_b = _normalize_brand(cand_brand) or brand_from_title(cand_title)
    if seed_b and cand_b and seed_b == cand_b:
        return True
    seed_tok = _tokens(seed_title) - SIZE_WORDS
    cand_tok = _tokens(cand_title) - SIZE_WORDS
    if seed_tok and cand_tok:
        shared = seed_tok & cand_tok
        union = seed_tok | cand_tok
        if union and len(shared) / len(union) >= 0.42:
            return True
        if seed_b and seed_b == cand_b:
            return True
        if len(shared) >= 5 and seed_b and cand_b and seed_b.split()[0] == cand_b.split()[0]:
            return True
    return False


def _collect_asins(hrefs: list[str], skip: set[str]) -> list[str]:
    found: list[str] = []
    for href in hrefs:
        asin = extract_asin(urllib.parse.unquote(href or ""))
        if not asin or asin in skip or asin in found:
            continue
        found.append(asin)
    return found


def _keep_item(
    item: dict[str, str],
    skip: set[str],
    seed_title: str,
    seed_brand: str,
) -> bool:
    asin = (item.get("asin") or "").upper()
    if not asin or asin in skip:
        return False
    title = item.get("title") or ""
    if not title:
        meta = catalog_product_for_asin(asin)
        title = meta.get("title") or ""
        if title:
            item["title"] = title
    if title and same_listing(seed_title, seed_brand, title, item.get("brand") or ""):
        return False
    return True


def _opera_task_matches(query_tokens: set[str], task_tokens: set[str]) -> bool:
    if not query_tokens or not task_tokens:
        return False
    shared = query_tokens & task_tokens
    if not shared:
        return False
    if query_tokens == task_tokens:
        return True
    if query_tokens <= task_tokens:
        return True
    # "fish" must not match "fish oil"; require a real phrase overlap.
    if task_tokens < query_tokens and len(task_tokens) < 2:
        return False
    return len(shared) >= min(2, len(query_tokens))


def competitors_from_opera(
    query: str,
    skip: set[str],
    limit: int = 4,
    *,
    seed_title: str = "",
    seed_brand: str = "",
) -> list[dict[str, str]]:
    if not CATALOG_PATH.exists() or not query:
        return []
    qtok = _tokens(query)
    if not qtok:
        return []
    scored: list[tuple[int, dict[str, str]]] = []
    for task in _catalog().get("tasks") or []:
        ttok = _tokens(task.get("query") or "")
        query_hit = _opera_task_matches(qtok, ttok)
        titles = task.get("example_products") or []
        asins = task.get("asins") or []
        for i, asin in enumerate(asins):
            asin = str(asin).upper()
            title = titles[i] if i < len(titles) else (titles[0] if titles else task.get("query") or "")
            title_shared = len(qtok & _tokens(title))
            if not query_hit and title_shared < min(2, len(qtok)):
                continue
            score = max(len(qtok & ttok), title_shared)
            scored.append(
                (
                    score,
                    {"asin": asin, "url": product_url_for_asin(asin), "title": title, "via": "opera"},
                )
            )
    scored.sort(key=lambda item: (-item[0], item[1]["asin"]))
    out: list[dict[str, str]] = []
    seen: set[str] = set(skip)
    for _, item in scored:
        if not _keep_item(item, seen, seed_title, seed_brand):
            seen.add(item["asin"])
            continue
        seen.add(item["asin"])
        out.append(item)
        if len(out) >= limit:
            return out
    return out


def competitors_from_duckduckgo(
    query: str,
    skip: set[str],
    limit: int = 4,
    *,
    seed_title: str = "",
    seed_brand: str = "",
) -> list[dict[str, str]]:
    q = f"{query} site:amazon.com/dp"
    if seed_brand:
        q = f'{query} -"{seed_brand}" site:amazon.com/dp'
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": q})
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return []
    hrefs = re.findall(r"https?://[^\s\"']*amazon\.com[^\s\"']*", html, re.I)
    hrefs += re.findall(r"uddg=([^&\"']+)", html)
    decoded = [urllib.parse.unquote(h) for h in hrefs]
    asins = _collect_asins(decoded, skip)
    out: list[dict[str, str]] = []
    seen = set(skip)
    for asin in asins:
        item = {"asin": asin, "url": product_url_for_asin(asin), "title": "", "via": "search"}
        if not _keep_item(item, seen, seed_title, seed_brand):
            seen.add(asin)
            continue
        seen.add(asin)
        out.append(item)
        if len(out) >= limit:
            break
    return out


async def _from_amazon(
    product_url: str,
    query: str,
    skip: set[str],
    limit: int,
    *,
    seed_title: str = "",
    seed_brand: str = "",
) -> dict[str, Any]:
    from playwright.async_api import async_playwright

    info: dict[str, Any] = {
        "title": seed_title,
        "brand": seed_brand,
        "canonical_asin": extract_asin(product_url),
        "parent_asin": "",
        "variation_asins": [],
        "candidates": [],
    }
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page(
            viewport={"width": 1280, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
            ),
        )
        try:
            await page.goto(product_url, wait_until="domcontentloaded", timeout=25000)
            await page.wait_for_timeout(1600)
            html = await page.content()
            blocked = len(html) < 20000 or "captcha" in html.lower()
            if not blocked:
                identity = await page.evaluate(
                    """() => {
                      const text = (sel) => (document.querySelector(sel)?.innerText || "").trim();
                      const brandRaw = text("#bylineInfo") || text("a#bylineInfo") || text("#brand");
                      const parentEl = document.querySelector("[data-parent-asin]");
                      const html = document.documentElement.innerHTML;
                      const parentMatch = html.match(/"parentAsin"\\s*:\\s*"([A-Z0-9]{10})"/i);
                      const canonical = (location.pathname.match(/\\/dp\\/([A-Z0-9]{10})/i) || [])[1] || "";
                      const variations = [...document.querySelectorAll(
                        "#twister li[data-defaultasin], #twister li[data-asin], #twister_feature_div [data-defaultasin], #variation_size_name li[data-defaultasin], #variation_color_name li[data-defaultasin], #variation_style_name li[data-defaultasin]"
                      )].map(el => el.getAttribute("data-defaultasin") || el.getAttribute("data-asin") || "")
                        .filter(Boolean);
                      const compareHrefs = [...document.querySelectorAll(
                        "#HLCXComparisonTable a[href*='/dp/'], #comparison-table a[href*='/dp/'], #similarities_feature_div a[href*='/dp/']"
                      )].map(a => a.href);
                      return {
                        title: text("#productTitle"),
                        brand: brandRaw,
                        parent: (parentEl && parentEl.getAttribute("data-parent-asin")) || (parentMatch && parentMatch[1]) || "",
                        canonical,
                        variations,
                        compareHrefs,
                      };
                    }"""
                )
                info["title"] = identity.get("title") or info["title"]
                info["brand"] = _normalize_brand(identity.get("brand") or "") or info["brand"] or brand_from_title(info["title"])
                info["canonical_asin"] = (identity.get("canonical") or info["canonical_asin"] or "").upper()
                info["parent_asin"] = (identity.get("parent") or "").upper()
                info["variation_asins"] = [str(a).upper() for a in (identity.get("variations") or []) if a]
                local_skip = set(skip) | {
                    info["canonical_asin"],
                    info["parent_asin"],
                    *info["variation_asins"],
                }
                local_skip.discard("")
                compare_asins = _collect_asins(identity.get("compareHrefs") or [], local_skip)
                for asin in compare_asins:
                    info["candidates"].append({"asin": asin, "title": "", "brand": "", "via": "amazon"})

            search_q = query or info["title"]
            if search_q:
                await page.goto(
                    f"https://www.amazon.com/s?k={quote_plus(search_q)}",
                    wait_until="domcontentloaded",
                    timeout=25000,
                )
                await page.wait_for_timeout(1600)
                html = await page.content()
                if len(html) > 20000 and "captcha" not in html.lower():
                    rows = await page.evaluate(
                        """() => {
                          const cards = [...document.querySelectorAll('div[data-component-type="s-search-result"][data-asin]')];
                          const fromCards = cards.slice(0, 24).map(el => ({
                            asin: (el.getAttribute("data-asin") || "").toUpperCase(),
                            title: (el.querySelector("h2")?.innerText || el.querySelector("h2 span")?.innerText || "").trim(),
                          }));
                          if (fromCards.length) return fromCards;
                          return [...document.querySelectorAll('a[href*="/dp/"]')].slice(0, 40).map(a => {
                            const m = (a.href || "").match(/\\/dp\\/([A-Z0-9]{10})/i);
                            return {
                              asin: (m && m[1] || "").toUpperCase(),
                              title: (a.innerText || a.getAttribute("aria-label") || "").trim(),
                            };
                          }).filter(row => row.asin);
                        }"""
                    )
                    local_skip = set(skip) | {
                        info.get("canonical_asin") or "",
                        info.get("parent_asin") or "",
                        *(info.get("variation_asins") or []),
                        *(c["asin"] for c in info["candidates"]),
                    }
                    local_skip.discard("")
                    for row in rows or []:
                        asin = (row.get("asin") or "").upper()
                        if not asin or asin in local_skip:
                            continue
                        info["candidates"].append(
                            {
                                "asin": asin,
                                "title": row.get("title") or "",
                                "brand": brand_from_title(row.get("title") or ""),
                                "via": "amazon",
                            }
                        )
                        local_skip.add(asin)
        except Exception:
            pass
        finally:
            await browser.close()

    seed_title = info.get("title") or seed_title
    seed_brand = info.get("brand") or seed_brand or brand_from_title(seed_title)
    skip_all = set(skip) | {
        info.get("canonical_asin") or "",
        info.get("parent_asin") or "",
        *(info.get("variation_asins") or []),
    }
    skip_all.discard("")
    kept: list[dict[str, str]] = []
    seen = set(skip_all)
    for cand in info["candidates"]:
        item = {
            "asin": cand["asin"],
            "url": product_url_for_asin(cand["asin"]),
            "title": cand.get("title") or "",
            "brand": cand.get("brand") or "",
            "via": cand.get("via") or "amazon",
        }
        if not _keep_item(item, seen, seed_title, seed_brand):
            seen.add(cand["asin"])
            continue
        seen.add(cand["asin"])
        kept.append(item)
        if len(kept) >= limit:
            break
    info["competitors"] = kept
    return info


async def find_competitors(
    product_url: str,
    *,
    query: str | None = None,
    persona: dict[str, Any] | None = None,
    limit: int = 4,
) -> dict[str, Any]:
    from simulator.model_profiles import is_vercel_runtime

    product_url = normalize_amazon_url(product_url)
    seed_asin = extract_asin(product_url)
    if not seed_asin:
        return {
            "product_url": product_url,
            "product_title": "",
            "query": "",
            "source": "none",
            "competitors": [],
        }
    skip = {seed_asin}
    meta = catalog_product_for_asin(seed_asin) if seed_asin else {}
    on_vercel = is_vercel_runtime()
    http_meta: dict[str, str] = {}
    if seed_asin and not (on_vercel and meta.get("title")):
        http_meta = fetch_amazon_product_http(product_url)
    seed_title = http_meta.get("title") or meta.get("title") or ""
    seed_brand = http_meta.get("brand") or meta.get("brand") or brand_from_title(seed_title)
    q = resolve_search_query(explicit=query, meta=meta, seed_title=seed_title, persona=persona)
    title = seed_title
    source = "none"
    items: list[dict[str, str]] = []

    if on_vercel:
        extra_opera = competitors_from_opera(
            q,
            skip,
            limit,
            seed_title=seed_title,
            seed_brand=seed_brand,
        )
        if extra_opera:
            items.extend(extra_opera)
            source = "opera"
        if len(items) < limit:
            extra_http = competitors_from_amazon_http(
                q,
                skip | {i["asin"] for i in items},
                limit - len(items),
                seed_title=seed_title,
                seed_brand=seed_brand,
                timeout=6.0,
            )
            if extra_http:
                items.extend(extra_http)
                source = "opera+amazon" if source == "opera" else "amazon"
        return {
            "product_url": product_url,
            "product_title": title or q,
            "query": q,
            "source": source,
            "competitors": items[:limit],
        }

    if len(items) < limit:
        extra_http = competitors_from_amazon_http(
            q,
            skip | {i["asin"] for i in items},
            limit - len(items),
            seed_title=seed_title,
            seed_brand=seed_brand,
        )
        if extra_http:
            items.extend(extra_http)
            source = "amazon"

    if len(items) < limit:
        try:
            amazon = await _from_amazon(product_url, q, skip | {i["asin"] for i in items}, limit - len(items), seed_title=seed_title, seed_brand=seed_brand)
            title = amazon.get("title") or title
            seed_brand = amazon.get("brand") or seed_brand or brand_from_title(title)
            skip |= {
                amazon.get("canonical_asin") or "",
                amazon.get("parent_asin") or "",
                *(amazon.get("variation_asins") or []),
            }
            skip.discard("")
            pw_items = list(amazon.get("competitors") or [])
            if pw_items:
                items.extend(pw_items)
                source = "amazon" if source == "none" else source + "+amazon"
        except Exception:
            pass

    if len(items) < limit:
        extra = competitors_from_duckduckgo(
            q,
            skip | {i["asin"] for i in items},
            limit - len(items),
            seed_title=title,
            seed_brand=seed_brand,
        )
        if extra:
            items.extend(extra)
            source = "search" if source == "none" else source + "+search"

    if len(items) < limit and not on_vercel:
        extra = competitors_from_opera(
            q,
            skip | {i["asin"] for i in items},
            limit - len(items),
            seed_title=title,
            seed_brand=seed_brand,
        )
        if extra:
            items.extend(extra)
            source = "opera" if source == "none" else source + "+opera"

    return {
        "product_url": product_url,
        "product_title": title or q,
        "query": q,
        "source": source,
        "competitors": items[:limit],
    }
