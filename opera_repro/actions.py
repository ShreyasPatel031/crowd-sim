"""Canonical browser actions and exact-match scoring.

Paper action space (Lu et al., ACL 2026, ShopCART / OPeRA transfer):
  click(name)
  type_and_submit(name, text)
  terminate()

OPeRA-filtered maps onto that space as:
  click     -> click
  input     -> type_and_submit
  terminate -> terminate

The target is OPeRA `semantic_id` (the HTML `name=` attribute).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

ACTION_TYPES = ("click", "type_and_submit", "terminate")
OPERA_TO_CANONICAL = {
    "click": "click",
    "input": "type_and_submit",
    "type_and_submit": "type_and_submit",
    "terminate": "terminate",
}

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_CALL_RE = re.compile(
    r"^\s*(click|type_and_submit|terminate)\s*\((.*)\)\s*$",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class Action:
    type: str
    name: str | None = None
    text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": self.type}
        if self.type == "click":
            payload["name"] = self.name or ""
        elif self.type == "type_and_submit":
            payload["name"] = self.name or ""
            payload["text"] = self.text or ""
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def to_call(self) -> str:
        if self.type == "terminate":
            return "terminate()"
        if self.type == "click":
            return f"click({_quote(self.name or '')})"
        return f"type_and_submit({_quote(self.name or '')}, {_quote(self.text or '')})"


def _quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _norm_name(value: str | None) -> str:
    return (value or "").strip()


def _norm_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def _norm_type(value: str | None) -> str:
    raw = (value or "").strip().lower()
    return OPERA_TO_CANONICAL.get(raw, raw)


def canonicalize(action: Action) -> Action:
    action_type = _norm_type(action.type)
    if action_type not in ACTION_TYPES:
        return Action(type=action_type, name=_norm_name(action.name) or None, text=_norm_text(action.text) or None)
    if action_type == "terminate":
        return Action(type="terminate")
    if action_type == "click":
        return Action(type="click", name=_norm_name(action.name) or None)
    return Action(
        type="type_and_submit",
        name=_norm_name(action.name) or None,
        text=_norm_text(action.text) or None,
    )


def actions_equal(pred: Action | None, gold: Action) -> bool:
    """Exact match: type, target name, and search text must all agree."""
    if pred is None:
        return False
    pred_c = canonicalize(pred)
    gold_c = canonicalize(gold)
    if pred_c.type != gold_c.type:
        return False
    if gold_c.type == "terminate":
        return True
    if pred_c.name != gold_c.name:
        return False
    if gold_c.type == "type_and_submit":
        return pred_c.text == gold_c.text
    return True


def _parse_call_args(raw: str) -> list[str]:
    raw = raw.strip()
    if not raw:
        return []
    try:
        parsed = json.loads(f"[{raw}]")
        return ["" if item is None else str(item) for item in parsed]
    except json.JSONDecodeError:
        parts = [p.strip().strip("'\"") for p in raw.split(",")]
        return [p for p in parts if p != ""]


def parse_action(text: str | None) -> Action | None:
    """Parse a model string into an Action, or None if illegal/unparseable."""
    if not text or not str(text).strip():
        return None
    blob = str(text).strip()

    fenced = _JSON_FENCE_RE.search(blob)
    if fenced:
        blob = fenced.group(1).strip()

    parsed_json = _try_json(blob)
    if parsed_json is None:
        match = _JSON_OBJECT_RE.search(blob)
        if match:
            parsed_json = _try_json(match.group(0))

    if parsed_json is not None:
        return _action_from_mapping(parsed_json)

    call = _CALL_RE.match(blob.splitlines()[-1].strip())
    if call:
        kind = _norm_type(call.group(1))
        args = _parse_call_args(call.group(2))
        if kind == "terminate":
            return canonicalize(Action(type="terminate"))
        if kind == "click" and args:
            return canonicalize(Action(type="click", name=args[0]))
        if kind == "type_and_submit" and len(args) >= 2:
            return canonicalize(Action(type="type_and_submit", name=args[0], text=args[1]))
        if kind == "type_and_submit" and len(args) == 1:
            return canonicalize(Action(type="type_and_submit", name=args[0], text=""))
    return None


def _try_json(blob: str) -> dict[str, Any] | None:
    try:
        value = json.loads(blob)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _action_from_mapping(payload: dict[str, Any]) -> Action | None:
    action_type = _norm_type(str(payload.get("type") or payload.get("action_type") or ""))
    name = payload.get("name", payload.get("target", payload.get("semantic_id")))
    text = payload.get("text", payload.get("input_text", payload.get("value")))
    if action_type not in ACTION_TYPES:
        return None
    name_s = None if name is None else str(name)
    text_s = None if text is None else str(text)
    return canonicalize(Action(type=action_type, name=name_s, text=text_s))


def action_from_opera_row(row: dict[str, Any]) -> Action | None:
    """Convert one OPeRA-filtered action row into the paper's 3-way space."""
    action_type = _norm_type(str(row.get("action_type") or ""))
    if action_type not in ACTION_TYPES:
        return None
    if action_type == "terminate":
        return Action(type="terminate")

    name = _first_nonempty(
        row.get("semantic_id"),
        _element_meta_name(row.get("element_meta")),
        row.get("click_type"),
    )
    if action_type == "click":
        if not name:
            return None
        return Action(type="click", name=name)

    text = _norm_text(str(row.get("input_text") or ""))
    if not name:
        name = "nav_bar.search_input"
    return Action(type="type_and_submit", name=name, text=text)


def _element_meta_name(raw: Any) -> str | None:
    if not raw:
        return None
    if isinstance(raw, dict):
        name = raw.get("name")
        return str(name) if name else None
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    if isinstance(parsed, dict) and parsed.get("name"):
        return str(parsed["name"])
    return None


def _first_nonempty(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def is_purchase_action(row: dict[str, Any], action: Action) -> bool:
    click_type = str(row.get("click_type") or "").strip().lower()
    if click_type == "purchase":
        return True
    name = (action.name or "").lower()
    return any(token in name for token in ("buy_now", "add_to_cart", "proceed_to_checkout", "place_order"))


def action_as_dict(action: Action) -> dict[str, Any]:
    return asdict(canonicalize(action))
