"""Compress OPeRA simplified HTML into a model observation.

Two builders live here:

`render_candidates` is the one used for training and eval. It lists every
element on the page that carries a `name=` attribute, so the full action
space is always present and the observation never depends on knowing which
element the human picked.

`compress_html` keeps a window around a named element. It is only safe for
*history* steps, where the action is already disclosed in the prompt. Using
it for the current step leaks the label into the input.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser

_STORES_RE = re.compile(
    r'<div name="nav_bar\.stores">.*?</div>',
    re.DOTALL,
)
_NAME_RE = re.compile(r'name="([^"]+)"')

# Attributes worth showing when an element has no visible text of its own.
_LABEL_ATTRS = ("aria-label", "title", "value", "alt", "placeholder")
_VOID_TAGS = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
)


def compress_html(
    html: str | None,
    max_chars: int,
    must_keep_name: str | None = None,
) -> str:
    text = html or ""
    if not text:
        return ""
    text = _STORES_RE.sub("", text, count=1)
    if len(text) <= max_chars:
        return text

    kept: list[str] = []
    title = _extract_title(text)
    if title:
        kept.append(f"<title>{title}</title>")

    if must_keep_name:
        window = _window_around_name(text, must_keep_name, window=min(1200, max_chars // 3))
        if window:
            kept.append(window)

    remaining = max_chars - sum(len(part) for part in kept) - 32
    if remaining <= 0:
        return _clip("".join(kept), max_chars)

    head_budget = max(remaining // 2, min(remaining, 1500))
    head = text[:head_budget]
    tail_budget = remaining - len(head)
    tail = text[-tail_budget:] if tail_budget > 200 else ""
    if tail and tail in head:
        tail = ""

    pieces = kept + [head]
    if tail:
        pieces.append("\n...[truncated]...\n")
        pieces.append(tail)
    else:
        pieces.append("\n...[truncated]...")
    return _clip("".join(pieces), max_chars)


def named_targets(html: str | None) -> list[str]:
    return _NAME_RE.findall(html or "")


@dataclass
class Candidate:
    name: str
    tag: str
    label: str
    #  False while `label` still holds an attribute fallback, so the first
    #  piece of visible text is allowed to replace it.
    label_from_text: bool = False

    def render(self, max_label_chars: int) -> str:
        head = f'name="{self.name}" ({self.tag})'
        if not max_label_chars or not self.label:
            return head
        label = self.label[:max_label_chars].rstrip()
        if len(self.label) > max_label_chars:
            label += "…"
        return f"{head} {label}"


class _CandidateParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.candidates: list[Candidate] = []
        self._by_name: dict[str, Candidate] = {}
        self._stack: list[str | None] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        mapping = {key: (value or "") for key, value in attrs}
        name = mapping.get("name", "").strip()
        pushed: str | None = None
        if name and name not in self._by_name:
            fallback = ""
            for attr in _LABEL_ATTRS:
                if mapping.get(attr, "").strip():
                    fallback = _squash(mapping[attr])
                    break
            candidate = Candidate(name=name, tag=tag, label=fallback)
            self.candidates.append(candidate)
            self._by_name[name] = candidate
            pushed = name
        elif name:
            pushed = name
        if tag not in _VOID_TAGS:
            self._stack.append(pushed)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if self._stack:
            self._stack.pop()

    def handle_data(self, data: str) -> None:
        text = _squash(data)
        if not text:
            return
        for name in reversed(self._stack):
            if name is None:
                continue
            candidate = self._by_name.get(name)
            if candidate is None:
                return
            # Visible text beats an attribute fallback, but never replaces
            # text we already collected for this element.
            if not candidate.label_from_text:
                candidate.label = text
                candidate.label_from_text = True
            return


def extract_candidates(html: str | None) -> list[Candidate]:
    """Every uniquely-named element on the page, in DOM order."""
    text = _STORES_RE.sub("", html or "", count=1)
    if not text:
        return []
    parser = _CandidateParser()
    try:
        parser.feed(text)
        parser.close()
    except Exception:
        # Malformed markup: fall back to a name-only listing so the action
        # space is still complete.
        seen: dict[str, None] = {}
        for name in _NAME_RE.findall(text):
            seen.setdefault(name, None)
        return [Candidate(name=name, tag="?", label="") for name in seen]
    return parser.candidates


def render_candidates(
    html: str | None,
    max_chars: int,
    max_label_chars: int = 90,
) -> str:
    """Label-free observation: the page's whole action space, plus labels.

    Every candidate name is always emitted — that is the set the model has to
    choose from. Only the descriptive labels are shortened to hit `max_chars`,
    so the correct answer can never be truncated away.
    """
    text = _STORES_RE.sub("", html or "", count=1)
    if not text:
        return ""
    candidates = extract_candidates(text)
    if not candidates:
        return _clip(text, max_chars)

    header = _extract_title(text)
    prefix = f"<title>{header}</title>\n" if header else ""

    for budget in (max_label_chars, 60, 40, 24, 0):
        body = "\n".join(candidate.render(budget) for candidate in candidates)
        if len(prefix) + len(body) <= max_chars:
            return prefix + body
    return prefix + "\n".join(candidate.render(0) for candidate in candidates)


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _extract_title(html: str) -> str:
    match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    return re.sub(r"\s+", " ", match.group(1)).strip() if match else ""


def _window_around_name(html: str, name: str, window: int) -> str:
    needle = f'name="{name}"'
    idx = html.find(needle)
    if idx < 0:
        return ""
    start = max(0, idx - window // 3)
    end = min(len(html), idx + window)
    return html[start:end]


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 16] + "\n...[truncated]"
