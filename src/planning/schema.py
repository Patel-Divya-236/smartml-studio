"""The shape a plan must have, and the parser that holds a reply to it.

Parsing is deliberately item by item. A plan is dozens of independent recommendations;
one entry with a missing field says nothing about the others, so it is reported as a
rejection and the rest are kept. Rejecting the whole reply for one bad entry would hand
every column back to the rules over a single typo.

This layer checks *shape* only -- is there a column, is confidence a number. Whether a
recommendation makes sense for this dataset is the validator's job, not the parser's.
"""

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

# Top-level keys a reply may use for each section. The prompt asks for the first; the
# others are spellings a model reaches for anyway, and accepting them costs nothing.
_SECTION_KEYS: dict[str, tuple[str, ...]] = {
    "preprocessing": ("preprocessing", "preprocessing_steps", "preprocessing_plan"),
    "feature_engineering": ("feature_engineering", "features", "feature_engineering_steps"),
    "models": ("models", "candidate_models", "model_recommendations"),
}


@dataclass
class Rejection:
    """One proposed recommendation that did not reach the user, and why."""

    section: str  # "preprocessing" | "feature_engineering" | "models"
    item: str     # a short description of what was proposed
    reason: str   # why it was refused, in words a user can act on

    def to_dict(self) -> dict[str, str]:
        """Serialise for the API."""
        return asdict(self)


def _confidence(value: Any) -> float:
    """Accept 0.86, "0.86" or 86 (a percentage), and clamp into [0, 1]."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.5
    if 1.0 < number <= 100.0:
        number /= 100.0
    return max(0.0, min(1.0, number))


class _Item(BaseModel):
    """Fields every planned recommendation carries."""

    model_config = ConfigDict(extra="ignore")

    reason: str = ""
    confidence: float = 0.5

    @field_validator("reason", mode="before")
    @classmethod
    def _text(cls, value: Any) -> str:
        return "" if value is None else str(value).strip()

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp(cls, value: Any) -> float:
        return _confidence(value)


class ColumnStep(_Item):
    """One preprocessing action on one column."""

    column: str
    technique: str
    # Informational. Each technique belongs to exactly one step, so the technique decides
    # where the recommendation is filed; a mismatched step is not grounds for rejection.
    step: str | None = None

    @field_validator("column", "technique", mode="before")
    @classmethod
    def _name(cls, value: Any) -> str:
        if value is None or isinstance(value, (list, dict)):
            raise ValueError("must be a single name")
        return str(value).strip()


class FeatureStep(_Item):
    """One feature-engineering step for the whole training set."""

    technique: str
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("params", mode="before")
    @classmethod
    def _params(cls, value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}


class ModelPick(_Item):
    """One candidate model."""

    model: str

    @field_validator("model", mode="before")
    @classmethod
    def _name(cls, value: Any) -> str:
        if value is None or isinstance(value, (list, dict)):
            raise ValueError("must be a single name")
        return str(value).strip()


@dataclass
class PlanResponse:
    """A parsed plan: shape-valid items only, not yet checked against the dataset."""

    preprocessing: list[ColumnStep]
    feature_engineering: list[FeatureStep]
    models: list[ModelPick]

    @property
    def is_empty(self) -> bool:
        return not (self.preprocessing or self.feature_engineering or self.models)


def extract_json(text: str) -> dict[str, Any] | None:
    """Pull the JSON object out of a reply, or return None when there is none.

    JSON mode usually returns a bare object, but not every provider honours it: replies
    arrive wrapped in a ```json fence, or with a sentence before the brace. The outermost
    `{...}` is tried after the whole text, which covers both.
    """
    if not text:
        return None
    candidate = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", candidate, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        candidate = fence.group(1)

    for attempt in (candidate, candidate[candidate.find("{"): candidate.rfind("}") + 1]):
        if not attempt:
            continue
        try:
            parsed = json.loads(attempt)
        except (TypeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _section(raw: dict[str, Any], name: str) -> list[Any]:
    for key in _SECTION_KEYS[name]:
        value = raw.get(key)
        if isinstance(value, list):
            return value
    return []


def _describe(item: Any) -> str:
    """A short, safe description of a raw item for a rejection message."""
    if isinstance(item, dict):
        bits = [str(item.get(k)) for k in ("technique", "model", "column") if item.get(k)]
        if bits:
            return " on ".join(bits[:2]) if "column" in item else bits[0]
    text = json.dumps(item, default=str) if not isinstance(item, str) else item
    return text[:80]


def _first_error(exc: ValidationError) -> str:
    error = exc.errors()[0]
    field = ".".join(str(p) for p in error.get("loc", ())) or "item"
    return f"malformed entry: '{field}' {error.get('msg', 'is invalid').lower()}"


def parse_plan(raw: dict[str, Any]) -> tuple[PlanResponse, list[Rejection]]:
    """Parse each section item by item. Returns the valid items and the malformed ones."""
    rejected: list[Rejection] = []

    preprocessing: list[ColumnStep] = []
    for item in _section(raw, "preprocessing"):
        # A step applied to several columns at once ("columns": [...]) is expanded into
        # one step per column, since every pipeline setting is per column.
        if isinstance(item, dict) and "column" not in item and isinstance(item.get("columns"), list):
            expanded = [{**item, "column": col} for col in item["columns"]]
        else:
            expanded = [item]
        for entry in expanded:
            try:
                preprocessing.append(ColumnStep.model_validate(entry))
            except ValidationError as exc:
                rejected.append(Rejection("preprocessing", _describe(entry), _first_error(exc)))

    features: list[FeatureStep] = []
    for item in _section(raw, "feature_engineering"):
        try:
            features.append(FeatureStep.model_validate(item))
        except ValidationError as exc:
            rejected.append(Rejection("feature_engineering", _describe(item), _first_error(exc)))

    models: list[ModelPick] = []
    for item in _section(raw, "models"):
        # A bare list of names is a reasonable reply too.
        entry = {"model": item} if isinstance(item, str) else item
        try:
            models.append(ModelPick.model_validate(entry))
        except ValidationError as exc:
            rejected.append(Rejection("models", _describe(item), _first_error(exc)))

    return PlanResponse(preprocessing, features, models), rejected
