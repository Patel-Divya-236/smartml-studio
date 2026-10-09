"""Combine validated planner output with the rule-based advisors.

The planner is primary; the rules are the safety net. Wherever the planner made a valid
choice it stands, and wherever it said nothing -- a column it skipped, a step it left
out, a proposal the validator refused -- the rule-based recommendation fills the gap.
So the pipeline the user is shown is always complete, whatever the planner returned.

Every recommendation leaves here tagged with `metadata["source"]`: "llm" for the
planner's, "rule" for the advisors'. The UI shows that tag on each card.
"""

from typing import Any

from config.settings import SETTINGS
from src.advisors.base import Recommendation
from src.planning.vocabulary import PREPROCESSING_LABELS, TECHNIQUE_EXPLANATIONS

RULE = "rule"


def tag(recommendations: list[Recommendation], source: str) -> list[Recommendation]:
    """Mark where recommendations came from, without overwriting an existing tag."""
    for rec in recommendations:
        rec.metadata = {**rec.metadata, "source": rec.metadata.get("source", source)}
    return recommendations


def _fallback_imputer(column: Any, profile: dict[str, Any]) -> Recommendation | None:
    """A fill for a column the planner kept but did not say how to fill.

    The rule advisor would drop such a column (that is the only imputation rule that
    collides here), but the planner has configured it -- encoded or scaled it -- which is
    a decision to keep it. Dropping it anyway would show the user "drop" next to the
    planner's own steps, the contradiction the advisors were rewritten to avoid. Median
    for numbers and mode for categories are the choices that need no further judgement.
    """
    missing = float((profile.get("missing_pct") or {}).get(column, 0.0) or 0.0)
    if missing <= 0:
        return None
    action = "median" if column in set(profile.get("numeric_columns") or []) else "mode"
    return Recommendation(
        label=PREPROCESSING_LABELS[action].format(column=column),
        confidence_score=0.6,
        reason=(f"The AI plan kept '{column}' but did not say how to fill its {missing:.1f}% "
                f"missing values; {action} is the safe default."),
        why_explanation=TECHNIQUE_EXPLANATIONS[action],
        category="imputation",
        metadata={"column": column, "action": action, "source": RULE},
    )


def merge_preprocessing(
    planned: list[Recommendation],
    rules: list[Recommendation],
    profile: dict[str, Any],
) -> list[Recommendation]:
    """Planner steps first; rule steps for every (column, step) the planner left open."""
    tag(rules, RULE)
    configured = {rec.metadata.get("column") for rec in planned}
    covered = {(rec.metadata.get("column"), rec.category) for rec in planned}
    dropped = {rec.metadata.get("column") for rec in planned if rec.metadata.get("action") == "drop"}

    merged = list(planned)
    for rec in rules:
        column = rec.metadata.get("column")
        if column in dropped or (column, rec.category) in covered:
            continue
        if rec.metadata.get("action") == "drop" and column in configured:
            replacement = _fallback_imputer(column, profile)
            if replacement is not None:
                merged.append(replacement)
            continue
        merged.append(rec)

    merged.sort(key=lambda r: r.confidence_score, reverse=True)
    return merged


def merge_models(
    planned: list[Recommendation],
    rules: list[Recommendation],
    excluded: set[str],
    minimum: int | None = None,
) -> list[Recommendation]:
    """The planner's ranking, topped up from the rules when it is too short.

    With no planned models at all, the rule ranking is used whole -- that is the
    "planner unavailable" case, and it must look exactly like the app did before the
    planner existed. Either way, a model the validator refused on safety grounds is not
    slipped back in by the top-up.
    """
    tag(rules, RULE)
    minimum = SETTINGS.MIN_RECOMMENDED_MODELS if minimum is None else minimum
    ranked_rules = sorted(rules, key=lambda r: r.confidence_score, reverse=True)

    if not planned:
        return [r for r in ranked_rules if r.metadata.get("model_name") not in excluded]

    merged = list(planned)
    chosen = {rec.metadata.get("model_name") for rec in merged}
    for rec in ranked_rules:
        if len(merged) >= minimum:
            break
        name = rec.metadata.get("model_name")
        if name in chosen or name in excluded:
            continue
        merged.append(rec)
        chosen.add(name)
    return merged
