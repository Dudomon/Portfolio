"""Champion/challenger promotion gate.

The gate is a pure function of (policy, challenger report, champion report,
drift report), which makes every decision reproducible and table-testable.
It never touches the registry; the caller acts on the decision by setting an
alias, so "promotion" has exactly one mechanism.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from credit_risk.evaluation import EvaluationReport
from credit_risk.governance.audit import normalize
from credit_risk.governance.policy import PromotionPolicy


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    observed: float | None
    threshold: float | None
    detail: str


@dataclass(frozen=True)
class GateDecision:
    policy_version: str
    passed: bool
    checks: tuple[CheckResult, ...]

    @property
    def failed_checks(self) -> tuple[CheckResult, ...]:
        return tuple(c for c in self.checks if not c.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "passed": self.passed,
            "checks": [asdict(c) for c in self.checks],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> GateDecision:
        return cls(
            policy_version=str(raw["policy_version"]),
            passed=bool(raw["passed"]),
            checks=tuple(CheckResult(**c) for c in raw["checks"]),
        )

    def to_record(
        self,
        *,
        model_name: str,
        run_id: str,
        champion_version: str | None,
        actor: str,
    ) -> dict[str, Any]:
        """Flat row for the Delta audit table."""
        return normalize(
            {
                "event_time": datetime.now(UTC),
                "event_type": "promotion_gate",
                "model_name": model_name,
                "run_id": run_id,
                "champion_version": champion_version,
                "policy_version": self.policy_version,
                "passed": self.passed,
                "failed_checks": [c.name for c in self.failed_checks],
                "checks_json": json.dumps([asdict(c) for c in self.checks]),
                "actor": actor,
            }
        )


def _at_most(name: str, observed: float, limit: float, detail: str = "") -> CheckResult:
    return CheckResult(name, observed <= limit, observed, limit, detail)


def _at_least(name: str, observed: float, limit: float, detail: str = "") -> CheckResult:
    return CheckResult(name, observed >= limit, observed, limit, detail)


def evaluate_gate(
    policy: PromotionPolicy,
    challenger: EvaluationReport,
    champion: EvaluationReport | None,
    drift_psi: Mapping[str, float],
) -> GateDecision:
    """Run every policy check; the gate passes only if all checks pass.

    ``champion`` is ``None`` on the very first run. Comparative checks then
    pass with an explicit "no champion" detail rather than being skipped, so
    the audit trail shows they were considered.
    """
    checks: list[CheckResult] = []
    perf = policy.performance
    metric = perf.primary_metric
    challenger_metric = getattr(challenger.performance, metric)

    checks.append(_at_least(f"performance.{metric}.minimum", challenger_metric, perf.min_value))
    if champion is None:
        checks.append(
            CheckResult(
                f"performance.{metric}.vs_champion",
                True,
                challenger_metric,
                None,
                "no champion registered; first promotion",
            )
        )
    else:
        champion_metric = getattr(champion.performance, metric)
        floor = champion_metric - perf.max_degradation_vs_champion
        checks.append(
            _at_least(
                f"performance.{metric}.vs_champion",
                challenger_metric,
                floor,
                f"champion={champion_metric:.4f}",
            )
        )

    checks.append(
        _at_most("calibration.ece", challenger.performance.ece, policy.calibration.max_ece)
    )

    fair = policy.fairness
    for attribute in fair.attributes:
        summary = challenger.fairness.get(attribute)
        if summary is None:
            checks.append(
                CheckResult(
                    f"fairness.{attribute}",
                    False,
                    None,
                    None,
                    "attribute missing from evaluation; cannot certify",
                )
            )
            continue
        checks.append(
            _at_least(
                f"fairness.{attribute}.min_group_size", summary.min_group_size, fair.min_group_size
            )
        )
        checks.append(
            _at_most(
                f"fairness.{attribute}.demographic_parity_difference",
                summary.demographic_parity_difference,
                fair.max_demographic_parity_difference,
            )
        )
        checks.append(
            _at_most(
                f"fairness.{attribute}.false_positive_rate_difference",
                summary.false_positive_rate_difference,
                fair.max_false_positive_rate_difference,
            )
        )
        checks.append(
            _at_most(
                f"fairness.{attribute}.equalized_odds_difference",
                summary.equalized_odds_difference,
                fair.max_equalized_odds_difference,
            )
        )
        champion_summary = champion.fairness.get(attribute) if champion else None
        if champion_summary is not None:
            champion_gap = champion_summary.false_positive_rate_difference
            checks.append(
                _at_most(
                    f"fairness.{attribute}.fpr_difference_vs_champion",
                    summary.false_positive_rate_difference,
                    champion_gap + fair.max_regression_vs_champion,
                    f"champion={champion_gap:.4f}",
                )
            )

    for feature in policy.drift.features:
        if feature not in drift_psi:
            checks.append(
                CheckResult(
                    f"drift.{feature}.psi",
                    False,
                    None,
                    policy.drift.max_psi,
                    "feature missing from drift report",
                )
            )
            continue
        checks.append(_at_most(f"drift.{feature}.psi", drift_psi[feature], policy.drift.max_psi))

    return GateDecision(
        policy_version=policy.version,
        passed=all(c.passed for c in checks),
        checks=tuple(checks),
    )
