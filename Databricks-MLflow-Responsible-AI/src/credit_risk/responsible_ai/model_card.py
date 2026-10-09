"""Model card generation (Markdown via Jinja2).

The card is rendered from the same objects the gate consumed, so the numbers a
reviewer reads are the numbers that decided the promotion. ``StrictUndefined``
turns any missing field into an error instead of an empty cell in a document
people rely on.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any

import pandas as pd
from jinja2 import Environment, PackageLoader, StrictUndefined

from credit_risk.config import MODEL_FEATURES, SENSITIVE_ATTRIBUTES
from credit_risk.evaluation import EvaluationReport
from credit_risk.governance.gate import GateDecision


@dataclass(frozen=True)
class TrainingDataSummary:
    source: str
    cutoff: date
    n_train: int
    n_holdout: int
    train_default_rate: float
    holdout_default_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "cutoff": self.cutoff.isoformat()}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TrainingDataSummary:
        return cls(**{**raw, "cutoff": date.fromisoformat(raw["cutoff"])})


@dataclass(frozen=True)
class MitigationComparison:
    """One row of the baseline vs mitigation table.

    Flat and JSON-serializable because it is computed in the training task and
    read back by the gate task, which renders the card.
    """

    name: str
    deployable: bool
    roc_auc: float
    flag_rate: float
    max_false_positive_rate_difference: float
    max_equalized_odds_difference: float
    note: str

    @classmethod
    def from_report(
        cls, name: str, report: EvaluationReport, *, deployable: bool, note: str
    ) -> MitigationComparison:
        return cls(
            name=name,
            deployable=deployable,
            roc_auc=report.performance.roc_auc,
            flag_rate=report.performance.flag_rate,
            max_false_positive_rate_difference=max(
                s.false_positive_rate_difference for s in report.fairness.values()
            ),
            max_equalized_odds_difference=max(
                s.equalized_odds_difference for s in report.fairness.values()
            ),
            note=note,
        )


@dataclass(frozen=True)
class CardInputs:
    """Training-time facts; persisted on the training run and read back by the gate task."""

    training_data: TrainingDataSummary
    hyperparameters: dict[str, Any]
    mitigations: tuple[MitigationComparison, ...]
    global_importance: pd.DataFrame  # columns: feature, mean_abs_shap


@dataclass(frozen=True)
class ModelCardContext:
    model_name: str
    run_id: str
    generated_at: datetime
    inputs: CardInputs
    evaluation: EvaluationReport
    gate: GateDecision
    features: tuple[str, ...] = MODEL_FEATURES
    excluded_attributes: tuple[str, ...] = SENSITIVE_ATTRIBUTES


def _format_number(value: float) -> str:
    # Counts (group sizes) read better without decimals; rates and gaps get three.
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.3f}"


def _environment() -> Environment:
    env = Environment(
        loader=PackageLoader("credit_risk.responsible_ai", "templates"),
        undefined=StrictUndefined,
        # Output is Markdown, not HTML; escaping would corrupt tables.
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    env.filters["pct"] = lambda value: f"{float(value):.1%}"
    env.filters["num"] = _format_number
    return env


def render_model_card(context: ModelCardContext) -> str:
    template = _environment().get_template("model_card.md.j2")
    return template.render(card=context)
