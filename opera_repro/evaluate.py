"""Exact next-action evaluation.

Primary metric (Lu et al. Table 2 on OPeRA):
  session-macro exact-match accuracy
    1. score each step: type + target + attribute must all match
    2. average steps inside a session
    3. average those session scores

Paper numbers to beat directionally:
  Qwen2.5-7B base        4.10%
  Qwen2.5-7B fine-tuned 32.04%
  + reasoning           35.14%
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

from opera_repro.actions import Action, actions_equal, parse_action


@dataclass
class EvalResult:
    n_examples: int
    n_sessions: int
    n_correct: int
    n_illegal: int
    micro_accuracy: float
    session_macro_accuracy: float
    action_type_accuracy: float
    session_outcome_accuracy: float | None
    by_action_type: dict[str, dict[str, float]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_examples": self.n_examples,
            "n_sessions": self.n_sessions,
            "n_correct": self.n_correct,
            "n_illegal": self.n_illegal,
            "micro_accuracy": self.micro_accuracy,
            "session_macro_accuracy": self.session_macro_accuracy,
            "action_type_accuracy": self.action_type_accuracy,
            "session_outcome_accuracy": self.session_outcome_accuracy,
            "by_action_type": self.by_action_type,
        }


def evaluate_predictions(
    records: Iterable[dict[str, Any]],
    predictions: Iterable[str | Action | None],
) -> EvalResult:
    rows = list(records)
    preds = list(predictions)
    if len(rows) != len(preds):
        raise ValueError(f"Got {len(preds)} predictions for {len(rows)} examples")

    per_session: dict[str, list[bool]] = defaultdict(list)
    type_correct: dict[str, list[bool]] = defaultdict(list)
    outcome_hits: list[bool] = []
    n_correct = 0
    n_illegal = 0
    n_type = 0

    for row, raw_pred in zip(rows, preds):
        gold = Action(**row["gold_action"]) if isinstance(row["gold_action"], dict) else row["gold_action"]
        pred = raw_pred if isinstance(raw_pred, Action) or raw_pred is None else parse_action(raw_pred)
        if pred is None:
            n_illegal += 1
        hit = actions_equal(pred, gold)
        n_correct += int(hit)
        per_session[row["session_id"]].append(hit)
        gold_type = gold.type
        pred_type = pred.type if pred is not None else None
        type_hit = pred_type == gold_type
        n_type += int(type_hit)
        type_correct[gold_type].append(hit)
        if row.get("is_session_outcome"):
            outcome_hits.append(hit)

    session_scores = [sum(hits) / len(hits) for hits in per_session.values() if hits]
    by_type = {
        action_type: {
            "n": float(len(hits)),
            "exact_accuracy": sum(hits) / len(hits) if hits else 0.0,
        }
        for action_type, hits in sorted(type_correct.items())
    }
    n = len(rows)
    return EvalResult(
        n_examples=n,
        n_sessions=len(per_session),
        n_correct=n_correct,
        n_illegal=n_illegal,
        micro_accuracy=n_correct / n if n else 0.0,
        session_macro_accuracy=sum(session_scores) / len(session_scores) if session_scores else 0.0,
        action_type_accuracy=n_type / n if n else 0.0,
        session_outcome_accuracy=(sum(outcome_hits) / len(outcome_hits) if outcome_hits else None),
        by_action_type=by_type,
    )


def format_report(result: EvalResult, title: str = "OPeRA next-action eval") -> str:
    lines = [
        title,
        f"  examples              {result.n_examples}",
        f"  sessions              {result.n_sessions}",
        f"  exact correct         {result.n_correct}",
        f"  illegal / unparsed    {result.n_illegal}",
        f"  micro exact-match     {result.micro_accuracy:.2%}",
        f"  session-macro exact   {result.session_macro_accuracy:.2%}   ← paper metric",
        f"  action-type accuracy  {result.action_type_accuracy:.2%}",
    ]
    if result.session_outcome_accuracy is not None:
        lines.append(f"  last-step exact       {result.session_outcome_accuracy:.2%}")
    for action_type, stats in result.by_action_type.items():
        lines.append(f"  {action_type:18} {stats['exact_accuracy']:.2%}  (n={int(stats['n'])})")
    lines.append("  paper reference: Qwen2.5-7B 4.10% → FT 32.04%")
    return "\n".join(lines)
