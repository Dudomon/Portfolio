"""Table-driven tests for the promotion gate and the policy loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from credit_risk.evaluation import EvaluationReport, PerformanceMetrics
from credit_risk.governance.gate import GateDecision, evaluate_gate
from credit_risk.governance.policy import PromotionPolicy, load_policy, parse_policy
from credit_risk.responsible_ai.fairness import FairnessSummary

ATTRIBUTES = ("region", "gender", "farm_size_band")


def _summary(
    attribute: str, *, dp: float, fpr_gap: float, eo: float, min_count: int
) -> FairnessSummary:
    by_group = pd.DataFrame(
        {
            "count": [float(min_count), 5_000.0],
            "selection_rate": [0.10, 0.10 + dp],
            "tpr": [0.40, 0.40],
            "fpr": [0.05, 0.05 + fpr_gap],
        },
        index=["g1", "g2"],
    )
    return FairnessSummary(attribute, dp, eo, by_group)


@dataclass(frozen=True)
class Spec:
    """Compact description of an evaluation report for table-driven cases."""

    roc_auc: float = 0.75
    ece: float = 0.01
    dp: float = 0.02
    fpr_gap: float = 0.02
    eo: float = 0.10
    min_count: int = 1_000
    overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    drop: tuple[str, ...] = ()

    def report(self) -> EvaluationReport:
        fairness = {}
        for attribute in ATTRIBUTES:
            if attribute in self.drop:
                continue
            values = {
                "dp": self.dp,
                "fpr_gap": self.fpr_gap,
                "eo": self.eo,
                "min_count": self.min_count,
                **self.overrides.get(attribute, {}),
            }
            fairness[attribute] = _summary(attribute, **values)
        return EvaluationReport(
            threshold=0.2,
            n_rows=8_000,
            performance=PerformanceMetrics(
                roc_auc=self.roc_auc,
                pr_auc=0.25,
                brier=0.07,
                ece=self.ece,
                flag_rate=0.09,
                base_rate=0.08,
            ),
            fairness=fairness,
        )


STABLE_DRIFT = {
    "loan_to_revenue": 0.01,
    "installment_to_revenue": 0.01,
    "interest_rate": 0.01,
    "credit_history_months": 0.05,
    "rainfall_anomaly": 0.02,
    "farm_size_ha": 0.01,
}

CASES = [
    # id, challenger, champion, drift, expected pass, expected failing checks
    ("first_model_passes", Spec(), None, STABLE_DRIFT, True, set()),
    (
        "below_minimum_auc",
        Spec(roc_auc=0.65),
        None,
        STABLE_DRIFT,
        False,
        {"performance.roc_auc.minimum"},
    ),
    (
        "small_drop_vs_champion_ok",
        Spec(roc_auc=0.745),
        Spec(roc_auc=0.75),
        STABLE_DRIFT,
        True,
        set(),
    ),
    (
        "large_drop_vs_champion",
        Spec(roc_auc=0.73),
        Spec(roc_auc=0.75),
        STABLE_DRIFT,
        False,
        {"performance.roc_auc.vs_champion"},
    ),
    ("miscalibrated", Spec(ece=0.08), None, STABLE_DRIFT, False, {"calibration.ece"}),
    (
        "gender_parity_gap",
        Spec(overrides={"gender": {"dp": 0.12}}),
        None,
        STABLE_DRIFT,
        False,
        {"fairness.gender.demographic_parity_difference"},
    ),
    (
        "region_fpr_gap",
        Spec(overrides={"region": {"fpr_gap": 0.07}}),
        None,
        STABLE_DRIFT,
        False,
        {"fairness.region.false_positive_rate_difference"},
    ),
    ("noisy_tpr_gap_within_backstop", Spec(eo=0.22), None, STABLE_DRIFT, True, set()),
    (
        "eo_backstop",
        Spec(overrides={"farm_size_band": {"eo": 0.3}}),
        None,
        STABLE_DRIFT,
        False,
        {"fairness.farm_size_band.equalized_odds_difference"},
    ),
    (
        "tiny_group",
        Spec(overrides={"region": {"min_count": 120}}),
        None,
        STABLE_DRIFT,
        False,
        {"fairness.region.min_group_size"},
    ),
    (
        "fairness_regression_vs_champion",
        Spec(fpr_gap=0.045),
        Spec(fpr_gap=0.01),
        STABLE_DRIFT,
        False,
        {f"fairness.{a}.fpr_difference_vs_champion" for a in ATTRIBUTES},
    ),
    (
        "attribute_missing_from_evaluation",
        Spec(drop=("gender",)),
        None,
        STABLE_DRIFT,
        False,
        {"fairness.gender"},
    ),
    (
        "significant_drift",
        Spec(),
        None,
        {**STABLE_DRIFT, "rainfall_anomaly": 0.4},
        False,
        {"drift.rainfall_anomaly.psi"},
    ),
    (
        "drift_feature_not_reported",
        Spec(),
        None,
        {k: v for k, v in STABLE_DRIFT.items() if k != "interest_rate"},
        False,
        {"drift.interest_rate.psi"},
    ),
]


@pytest.mark.parametrize(
    ("challenger", "champion", "drift", "expected_pass", "expected_failures"),
    [pytest.param(*case[1:], id=case[0]) for case in CASES],
)
def test_gate_decisions(
    policy: PromotionPolicy,
    challenger: Spec,
    champion: Spec | None,
    drift: dict[str, float],
    expected_pass: bool,
    expected_failures: set[str],
) -> None:
    decision = evaluate_gate(
        policy, challenger.report(), champion.report() if champion else None, drift
    )
    assert decision.passed is expected_pass
    assert {c.name for c in decision.failed_checks} == expected_failures
    assert decision.policy_version == policy.version


def test_first_promotion_records_comparison_as_considered(policy: PromotionPolicy) -> None:
    decision = evaluate_gate(policy, Spec().report(), None, STABLE_DRIFT)
    vs_champion = next(c for c in decision.checks if c.name.endswith("vs_champion"))
    assert vs_champion.passed
    assert "no champion" in vs_champion.detail


def test_audit_record_and_round_trip(policy: PromotionPolicy) -> None:
    decision = evaluate_gate(policy, Spec(ece=0.2).report(), None, STABLE_DRIFT)
    record = decision.to_record(model_name="c.s.m", run_id="r1", champion_version=None, actor="ci")
    assert record["passed"] is False
    assert record["failed_checks"] == ["calibration.ece"]
    assert record["policy_version"] == policy.version
    assert GateDecision.from_dict(decision.to_dict()) == decision


def test_repository_policy_loads(policy: PromotionPolicy) -> None:
    assert policy.fairness.attributes == ATTRIBUTES
    assert 0 < policy.decision_threshold < 1


def _raw_policy(project_root: Path) -> dict[str, Any]:
    raw: dict[str, Any] = yaml.safe_load(
        (project_root / "conf" / "promotion_policy.yml").read_text(encoding="utf-8")
    )
    return raw


def test_policy_rejects_unknown_threshold_names(project_root: Path) -> None:
    raw = _raw_policy(project_root)
    raw["fairness"]["max_dp_difference"] = 0.1  # typo of a real key
    with pytest.raises(ValueError, match="unknown keys"):
        parse_policy(raw)


def test_policy_rejects_missing_sections(project_root: Path) -> None:
    raw = _raw_policy(project_root)
    del raw["drift"]
    with pytest.raises(ValueError, match="Policy keys"):
        parse_policy(raw)


@pytest.mark.parametrize("threshold", [0, 1, 1.5])
def test_policy_rejects_degenerate_threshold(project_root: Path, threshold: float) -> None:
    raw = _raw_policy(project_root)
    raw["decision_threshold"] = threshold
    with pytest.raises(ValueError, match="decision_threshold"):
        parse_policy(raw)


def test_policy_version_change_is_visible(project_root: Path, tmp_path: Path) -> None:
    raw = _raw_policy(project_root)
    raw["version"] = "test-2"
    path = tmp_path / "policy.yml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    policy = load_policy(path)
    decision = evaluate_gate(policy, Spec().report(), None, STABLE_DRIFT)
    assert decision.policy_version == "test-2"
