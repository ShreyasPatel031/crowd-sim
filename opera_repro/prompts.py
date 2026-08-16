"""Prompts for next-action prediction (v1: no persona, no rationale)."""

from __future__ import annotations

import json

from opera_repro.actions import Action

SYSTEM_PROMPT = """You are simulating a real shopper on amazon.com.
Predict the shopper's immediate next browser action.

Action space — output ONE JSON object, nothing else:

1. click a named element
{"type": "click", "name": "element_name"}

2. type a query into a field and submit
{"type": "type_and_submit", "name": "element_name", "text": "search query"}

3. close the browser / give up
{"type": "terminate"}

Rules:
- `name` must be copied exactly from an HTML name="..." attribute in the current observation.
- Follow natural shopping sequences (search, inspect products, maybe buy, or terminate).
- Do not invent extra keys. Do not wrap the JSON in markdown.
"""

# Inference always emits reason. Training still only copies a human rationale
# when one exists — never an empty "reason": "".
REASONING_SYSTEM = """You are simulating a real shopper on amazon.com.
Predict the shopper's immediate next browser action.

Always return ONE JSON object. "reason" is required and must be a non-empty
shopper-voice sentence (under 25 words), placed FIRST:

{"reason": "this one is cheaper and well reviewed", "type": "click", "name": "element_name"}
{"reason": "I need a bike tail light", "type": "type_and_submit", "name": "element_name", "text": "search query"}
{"reason": "I'm done shopping", "type": "terminate"}

Rules:
- `name` must be copied exactly from a name="..." value in the current observation.
- Do not wrap the JSON in markdown.
- Never omit "reason". Never set it to an empty string.
"""

REASONING_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "reason": {"type": "STRING"},
        "type": {"type": "STRING"},
        "name": {"type": "STRING"},
        "text": {"type": "STRING"},
    },
    "required": ["reason", "type"],
    "propertyOrdering": ["reason", "type", "name", "text"],
}

VANILLA_SYSTEM = """Predict the next browser action. Return one JSON object only."""

TAGS_SYSTEM = SYSTEM_PROMPT + """

OPeRA name= tags you will see (copy them exactly, do not invent):

Navigation
  nav_bar.search_input  nav_bar.search_button  nav_bar.cart_button
  nav_bar.homepage  nav_bar.menu  nav_bar.order_button
  go_to_cart  cart_side_bar.go_to_cart

Search results
  product_1 … product_N
  search_results.<product_slug>  search_results.sort  pagination.2  pagination.next
  refinements.<filter_slug>

Product page
  buybox.purchase_form.add_to_cart  (also called add_to_cart)
  buybox.purchase_form.buy_now  (also called buy_now)
  product_options.color.button_list.<color>
  product_options.size.drop_down_list.open_drop_down_list
  reviews  about_this_item  coupon.checkbox

Cart
  check_out  proceed_to_checkout  cart_header.select_all_items
  active_item_list.<product_slug>.checkbox
"""


def fewshot_block(examples: list[dict]) -> str:
    """Render short (observation, gold) pairs for the system prompt."""
    parts = ["", "Worked examples (same schema):"]
    for i, row in enumerate(examples, start=1):
        user = next(m["content"] for m in row["messages"] if m["role"] == "user")
        gold = json.dumps(row["gold_action"], ensure_ascii=False)
        # Keep shots short: current observation tail only.
        marker = "Current observation:"
        obs = user.split(marker, 1)[-1].strip() if marker in user else user
        obs = obs.replace("Next action JSON:", "").strip()
        if len(obs) > 900:
            obs = obs[:900] + "\n...[truncated]"
        parts.append(f"Example {i} observation:\n{obs}\nExample {i} next action:\n{gold}")
    return "\n".join(parts)



def format_user_prompt(history: list[tuple[str, Action]], current_html: str) -> str:
    parts = [
        "Predict the next action from the session history and the current page.",
        "",
        "Session history:",
    ]
    if not history:
        parts.append("(no previous actions)")
    else:
        for i, (html, action) in enumerate(history, start=1):
            parts.append(f"[Step {i}]")
            parts.append("Observation:")
            parts.append(html)
            parts.append("Action:")
            parts.append(action.to_json())
            parts.append("")
    parts.extend(
        [
            "Current observation:",
            current_html,
            "",
            "Next action JSON:",
        ]
    )
    return "\n".join(parts)


def gold_target_json(gold: Action, rationale: str | None = None) -> str:
    """Assistant target. `reason` only when the human actually wrote one."""
    text = (rationale or "").strip()
    if not text:
        return gold.to_json()
    payload = {"reason": text}
    payload.update(gold.to_dict())
    return json.dumps(payload, ensure_ascii=False)


def build_messages(
    history: list[tuple[str, Action]],
    current_html: str,
    gold: Action,
    rationale: str | None = None,
    reasoning_prompt: bool = False,
) -> list[dict[str, str]]:
    system = REASONING_SYSTEM if reasoning_prompt else SYSTEM_PROMPT
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": format_user_prompt(history, current_html)},
        {"role": "assistant", "content": gold_target_json(gold, rationale)},
    ]
