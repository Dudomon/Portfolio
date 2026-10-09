"""Model training and mitigation comparison. Skipped cleanly without the ML extras."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from credit_risk.config import LABEL, LOAN_KEY
from credit_risk.data.synthetic import GeneratorConfig, generate_loans
from credit_risk.pipeline import (
    build_model_frame,
    compare_mitigations,
    evaluate_scores,
    fit_candidate,
    labeled,
    prepare_silver,
)
from credit_risk.training.model import (
    TemporalSplit,
    predict_default_proba,
    temporal_split,
)


@pytest.fixture(scope="module")
def split() -> TemporalSplit:
    raw = generate_loans(GeneratorConfig(n_loans=12_000, seed=21))
    return temporal_split(labeled(build_model_frame(prepare_silver(raw).valid)))


def test_candidate_learns_signal(split: TemporalSplit) -> None:
    candidate = fit_candidate(split.train, mitigation="none", n_trials=0)
    report = evaluate_scores(
        split.holdout, predict_default_proba(candidate.pipeline, split.holdout), 0.2
    )
    # Small holdout (about 2.4k rows): a loose floor that still catches a broken pipeline.
    assert report.performance.roc_auc > 0.65
    assert report.performance.ece < 0.05


def test_model_ignores_sensitive_columns(split: TemporalSplit) -> None:
    candidate = fit_candidate(split.train, mitigation="none", n_trials=0)
    shuffled = split.holdout.copy()
    rng = np.random.default_rng(0)
    for column in ("region", "gender", "farm_size_band"):
        shuffled[column] = rng.permutation(shuffled[column].to_numpy())
    np.testing.assert_array_equal(
        predict_default_proba(candidate.pipeline, split.holdout),
        predict_default_proba(candidate.pipeline, shuffled),
    )


def test_mitigation_comparison_has_baseline_reweighing_and_benchmark(split: TemporalSplit) -> None:
    candidate = fit_candidate(split.train, mitigation="reweighing", n_trials=0)
    rows = compare_mitigations(split, candidate, threshold=0.2)
    assert [r.name for r in rows][:2] == ["baseline", "reweighing"]
    assert rows[2].name.startswith("threshold_optimizer[")
    assert [r.deployable for r in rows] == [True, True, False]
    # Reweighing targets the designed region/farm-size base-rate gaps, so it
    # should not make the worst FPR gap worse than the unmitigated baseline.
    assert rows[1].max_false_positive_rate_difference <= rows[0].max_false_positive_rate_difference


def test_tuning_logs_each_trial_through_callback(split: TemporalSplit) -> None:
    pytest.importorskip("optuna")
    seen: list[tuple[int, dict[str, Any], float]] = []
    candidate = fit_candidate(
        split.train, mitigation="none", n_trials=2, on_trial=lambda *args: seen.append(args)
    )
    assert [number for number, _, _ in seen] == [0, 1]
    assert candidate.tuning is not None
    assert candidate.params == candidate.tuning.best_params
    assert all(0.5 < auc <= 1.0 for _, _, auc in seen)


def test_explanations_fold_one_hot_columns(split: TemporalSplit) -> None:
    pytest.importorskip("shap")
    from credit_risk.responsible_ai.explain import explain

    candidate = fit_candidate(split.train, mitigation="none", n_trials=0)
    holdout = split.holdout.head(400)
    explanations = explain(candidate.pipeline, holdout, holdout[LOAN_KEY], n_local=5, top_k=3)
    features = explanations.global_importance["feature"].tolist()
    assert "crop_type" in features
    assert not any(f.startswith("crop_type_") for f in features)
    assert explanations.global_importance["mean_abs_shap"].is_monotonic_decreasing
    assert len(explanations.local) == 5
    first = explanations.local[0]
    assert len(first["top_factors"]) == 3
    assert first["predicted_default_probability"] == pytest.approx(
        predict_default_proba(candidate.pipeline, holdout).max()
    )


def test_label_column_is_integer(split: TemporalSplit) -> None:
    assert pd.api.types.is_integer_dtype(split.train[LABEL])
