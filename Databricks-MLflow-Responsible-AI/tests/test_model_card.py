from __future__ import annotations

from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import pytest
from jinja2 import UndefinedError

from credit_risk.evaluation import evaluate
from credit_risk.governance.gate import evaluate_gate
from credit_risk.governance.policy import PromotionPolicy
from credit_risk.responsible_ai.model_card import (
    CardInputs,
    MitigationComparison,
    ModelCardContext,
    TrainingDataSummary,
    render_model_card,
)


@pytest.fixture
def context(policy: PromotionPolicy) -> ModelCardContext:
    rng = np.random.default_rng(0)
    n = 2_000
    sensitive = pd.DataFrame(
        {
            "region": rng.choice(["Sul", "Nordeste"], n),
            "gender": rng.choice(["F", "M"], n, p=[0.2, 0.8]),
            "farm_size_band": rng.choice(["smallholder", "medium", "large"], n),
        }
    )
    score = rng.beta(2, 18, n)
    y = (rng.random(n) < score).astype(int)
    report = evaluate(y, score, sensitive, threshold=policy.decision_threshold)
    decision = evaluate_gate(policy, report, None, {f: 0.01 for f in policy.drift.features})
    inputs = CardInputs(
        training_data=TrainingDataSummary(
            source="cat.sch.silver_loan_applications",
            cutoff=date(2024, 5, 1),
            n_train=8_000,
            n_holdout=2_000,
            train_default_rate=0.09,
            holdout_default_rate=0.08,
        ),
        hyperparameters={"num_leaves": 15, "learning_rate": 0.031234567},
        mitigations=(
            MitigationComparison.from_report("baseline", report, deployable=True, note="none"),
        ),
        global_importance=pd.DataFrame(
            {"feature": ["rainfall_anomaly", "loan_to_revenue"], "mean_abs_shap": [0.4, 0.2]}
        ),
    )
    return ModelCardContext(
        model_name="cat.sch.rural_credit_default",
        run_id="abc123",
        generated_at=datetime(2026, 10, 1, 12, 0, tzinfo=UTC),
        inputs=inputs,
        evaluation=report,
        gate=decision,
    )


def test_card_contains_every_section(context: ModelCardContext) -> None:
    card = render_model_card(context)
    for heading in (
        "## Intended use",
        "## Out-of-scope uses",
        "## Training data",
        "## Performance",
        "## Fairness",
        "### gender",
        "## Mitigation comparison",
        "## Explainability",
        "## Promotion gate",
        "## Known limitations",
        "## Human oversight",
    ):
        assert heading in card


def test_card_reports_real_values(context: ModelCardContext) -> None:
    card = render_model_card(context)
    assert "`abc123`" in card
    assert "2024-05-01" in card
    assert f"{context.evaluation.performance.roc_auc:.3f}" in card
    assert "| learning_rate | 0.0312 |" in card
    assert "synthetic, fictitious data" in card
    assert f"Result: **{'PASSED' if context.gate.passed else 'FAILED'}**" in card


def test_card_lists_excluded_sensitive_attributes(context: ModelCardContext) -> None:
    card = render_model_card(context)
    assert "region, gender, farm_size_band" in card


def test_missing_values_fail_loudly(context: ModelCardContext) -> None:
    broken = ModelCardContext(
        model_name=context.model_name,
        run_id=context.run_id,
        generated_at=context.generated_at,
        inputs=context.inputs,
        evaluation=type(context.evaluation)(
            threshold=context.evaluation.threshold,
            n_rows=context.evaluation.n_rows,
            performance=context.evaluation.performance,
            fairness=context.evaluation.fairness,
            calibration_by_group={},  # template needs per-group calibration
        ),
        gate=context.gate,
    )
    with pytest.raises(UndefinedError):
        render_model_card(broken)


def test_card_inputs_round_trip_through_json() -> None:
    summary = TrainingDataSummary("t", date(2024, 1, 2), 1, 2, 0.1, 0.2)
    assert TrainingDataSummary.from_dict(summary.to_dict()) == summary
