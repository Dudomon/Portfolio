"""Promotion policy loaded from a versioned YAML file.

The policy lives in ``conf/`` under version control, and its ``version`` is
stamped on every gate decision (MLflow tags and the Delta audit table), so any
promotion can be traced back to the exact thresholds that allowed it.
Unknown keys are rejected: a typo in a threshold name must fail loudly rather
than silently fall back to a default.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

import yaml

PrimaryMetric = Literal["roc_auc", "pr_auc"]


@dataclass(frozen=True)
class PerformancePolicy:
    primary_metric: PrimaryMetric
    min_value: float
    # Largest allowed drop versus the current champion (absolute, same units).
    max_degradation_vs_champion: float


@dataclass(frozen=True)
class CalibrationPolicy:
    max_ece: float


@dataclass(frozen=True)
class FairnessPolicy:
    attributes: tuple[str, ...]
    max_demographic_parity_difference: float
    # Primary harm metric: creditworthy producers flagged, compared across groups.
    max_false_positive_rate_difference: float
    # Backstop on the larger of the TPR/FPR gaps. Looser than the FPR limit
    # because TPR is estimated on each group's few defaults and is noisy.
    max_equalized_odds_difference: float
    # Fairness gaps measured on tiny groups are noise; too-small groups fail the
    # gate so a human looks at them instead of the gate passing by luck.
    min_group_size: int
    # A challenger may not widen the FPR gap by more than this versus the
    # champion, even if it stays under the absolute ceiling.
    max_regression_vs_champion: float


@dataclass(frozen=True)
class DriftPolicy:
    max_psi: float
    features: tuple[str, ...]


@dataclass(frozen=True)
class PromotionPolicy:
    version: str
    decision_threshold: float
    performance: PerformancePolicy
    calibration: CalibrationPolicy
    fairness: FairnessPolicy
    drift: DriftPolicy


_T = TypeVar("_T")


def _build(cls: type[_T], raw: Mapping[str, Any], section: str) -> _T:
    expected = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(raw) - expected
    missing = expected - set(raw)
    if unknown or missing:
        raise ValueError(
            f"Policy section '{section}': unknown keys {sorted(unknown)}, missing {sorted(missing)}"
        )
    values = {k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()}
    return cls(**values)


def parse_policy(raw: Mapping[str, Any]) -> PromotionPolicy:
    sections = {"performance", "calibration", "fairness", "drift"}
    top_level = {"version", "decision_threshold", *sections}
    if set(raw) != top_level:
        raise ValueError(f"Policy keys must be exactly {sorted(top_level)}, got {sorted(raw)}")
    policy = PromotionPolicy(
        version=str(raw["version"]),
        decision_threshold=float(raw["decision_threshold"]),
        performance=_build(PerformancePolicy, raw["performance"], "performance"),
        calibration=_build(CalibrationPolicy, raw["calibration"], "calibration"),
        fairness=_build(FairnessPolicy, raw["fairness"], "fairness"),
        drift=_build(DriftPolicy, raw["drift"], "drift"),
    )
    if not 0 < policy.decision_threshold < 1:
        raise ValueError("decision_threshold must be in (0, 1)")
    if policy.performance.primary_metric not in ("roc_auc", "pr_auc"):
        raise ValueError(f"Unsupported primary metric {policy.performance.primary_metric!r}")
    return policy


def load_policy(path: str | Path) -> PromotionPolicy:
    with Path(path).open(encoding="utf-8") as handle:
        return parse_policy(cast(Mapping[str, Any], yaml.safe_load(handle)))
