"""Local end-to-end run: the Databricks job graph collapsed into one process.

Uses a SQLite-backed MLflow tracking server and model registry under
``--output-dir`` so the alias-based champion/challenger flow can be exercised
without a workspace. Spark, Delta, and Feature Engineering are replaced by
their pandas equivalents in :mod:`credit_risk.pipeline`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from credit_risk.config import CHAMPION_ALIAS, LOAN_KEY, UnityCatalogNames
from credit_risk.data.synthetic import BiasConfig, GeneratorConfig, generate_loans
from credit_risk.governance.audit import alias_change_record
from credit_risk.governance.gate import GateDecision
from credit_risk.governance.policy import load_policy
from credit_risk.governance.registry import (
    get_alias_version,
    promote_to_champion,
    register_challenger,
)
from credit_risk.pipeline import (
    Mitigation,
    build_model_frame,
    compare_mitigations,
    fit_candidate,
    labeled,
    prepare_silver,
    run_gate,
    training_summary,
)
from credit_risk.responsible_ai.explain import explain
from credit_risk.responsible_ai.model_card import CardInputs, ModelCardContext, render_model_card
from credit_risk.tracking import (
    log_card_inputs,
    log_evaluation,
    log_explanations,
    log_gate_decision,
    log_model_card,
    nested_trial_logger,
)
from credit_risk.training.model import predict_default_proba, temporal_split
from credit_risk.training.pyfunc import PREDICTION_COLUMN, log_pyfunc_model, model_inputs

LOCAL_ACTOR = "local-runner"
DEFAULT_POLICY = Path(__file__).resolve().parents[2] / "conf" / "promotion_policy.yml"


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=Path(".local_run"))
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--n-loans", type=int, default=40_000)
    parser.add_argument("--n-trials", type=int, default=15)
    parser.add_argument("--mitigation", choices=("none", "reweighing"), default="reweighing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--unbiased", action="store_true", help="Generate data with the injected biases disabled."
    )
    parser.add_argument(
        "--no-auto-promote",
        action="store_true",
        help="Register a passing model as challenger only (mirrors the prod target).",
    )
    return parser.parse_args(argv)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def _setup_tracking(out: Path) -> MlflowClient:
    mlflow.set_tracking_uri(f"sqlite:///{(out / 'mlflow.db').as_posix()}")
    experiment = "credit-risk-local"
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=(out / "artifacts").as_uri())
    mlflow.set_experiment(experiment)
    return MlflowClient()


def _champion_scores(
    client: MlflowClient, names: UnityCatalogNames, holdout: pd.DataFrame
) -> tuple[str | None, np.ndarray | None]:
    version = get_alias_version(client, names.model_name, CHAMPION_ALIAS)
    if version is None:
        return None, None
    champion = mlflow.pyfunc.load_model(names.model_uri(CHAMPION_ALIAS))
    predictions = champion.predict(model_inputs(holdout))
    return version, np.asarray(predictions[PREDICTION_COLUMN], dtype=float)


def _register_and_promote(
    client: MlflowClient,
    names: UnityCatalogNames,
    *,
    model_uri: str,
    run_id: str,
    decision: GateDecision,
    audit_log: Path,
    auto_promote: bool,
) -> None:
    """Same sequence as notebook 06: register as challenger, optionally promote."""
    version = register_challenger(client, names.model_name, model_uri, run_id, decision)
    _append_jsonl(
        audit_log,
        alias_change_record(
            "registration",
            model_name=names.model_name,
            model_version=version,
            previous_version=None,
            actor=LOCAL_ACTOR,
        ),
    )
    print(f"Registered {names.model_name} v{version} as challenger")
    if not auto_promote:
        return
    previous = promote_to_champion(client, names.model_name, version, approved_by=LOCAL_ACTOR)
    _append_jsonl(
        audit_log,
        alias_change_record(
            "promotion",
            model_name=names.model_name,
            model_version=version,
            previous_version=previous,
            actor=LOCAL_ACTOR,
        ),
    )
    print(f"Promoted v{version} to champion (previous: {previous})")


def run_local(args: argparse.Namespace) -> None:
    out: Path = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    client = _setup_tracking(out)
    names = UnityCatalogNames(catalog="local", schema="credit_risk")
    policy = load_policy(args.policy)

    bias = BiasConfig.unbiased() if args.unbiased else BiasConfig()
    raw = generate_loans(GeneratorConfig(n_loans=args.n_loans, bias=bias, seed=args.seed))
    validation = prepare_silver(raw)
    validation.quarantined.to_csv(out / "quarantine.csv", index=False)
    split = temporal_split(labeled(build_model_frame(validation.valid)))
    mitigation: Mitigation = args.mitigation

    with mlflow.start_run(run_name=f"train-{mitigation}") as run:
        run_id = run.info.run_id
        mlflow.set_tags({"stage": "training", "policy_version": policy.version})
        mlflow.log_params(
            {
                "n_loans": args.n_loans,
                "mitigation": mitigation,
                "unbiased_data": args.unbiased,
                "holdout_cutoff": split.cutoff.isoformat(),
                "quarantined_rows": len(validation.quarantined),
            }
        )
        candidate = fit_candidate(
            split.train,
            mitigation=mitigation,
            n_trials=args.n_trials,
            seed=args.seed,
            on_trial=nested_trial_logger(),
        )
        mlflow.log_params({f"model.{k}": v for k, v in candidate.params.items()})
        model_info = log_pyfunc_model(candidate.pipeline, split.train)

        explanations = explain(candidate.pipeline, split.holdout, split.holdout[LOAN_KEY])
        log_explanations(explanations)
        inputs = CardInputs(
            training_data=training_summary(split, source="synthetic generator (local)"),
            hyperparameters=candidate.params,
            mitigations=compare_mitigations(
                split, candidate, policy.decision_threshold, seed=args.seed
            ),
            global_importance=explanations.global_importance,
        )
        log_card_inputs(inputs)

        champion_version, champion_scores = _champion_scores(client, names, split.holdout)
        challenger_scores = predict_default_proba(candidate.pipeline, split.holdout)
        outcome = run_gate(policy, split, challenger_scores, champion_scores)
        log_evaluation(outcome.challenger, "holdout_", "evaluation/challenger")
        if outcome.champion is not None:
            log_evaluation(outcome.champion, "champion_holdout_", "evaluation/champion")
        mlflow.log_dict(outcome.drift_psi, "governance/drift_psi.json")
        log_gate_decision(outcome.decision)
        card = render_model_card(
            ModelCardContext(
                model_name=names.model_name,
                run_id=run_id,
                generated_at=datetime.now(UTC),
                inputs=inputs,
                evaluation=outcome.challenger,
                gate=outcome.decision,
            )
        )
        log_model_card(card)
        (out / "model_card.md").write_text(card, encoding="utf-8", newline="\n")

    decision = outcome.decision
    audit_log = out / "promotion_audit.jsonl"
    _append_jsonl(
        audit_log,
        decision.to_record(
            model_name=names.model_name,
            run_id=run_id,
            champion_version=champion_version,
            actor=LOCAL_ACTOR,
        ),
    )
    print(
        f"Run {run_id}: gate {'PASSED' if decision.passed else 'FAILED'} "
        f"(policy {decision.policy_version})"
    )
    for check in decision.failed_checks:
        print(f"  FAIL {check.name}: observed={check.observed} threshold={check.threshold}")
    if decision.passed:
        _register_and_promote(
            client,
            names,
            model_uri=model_info.model_uri,
            run_id=run_id,
            decision=decision,
            audit_log=audit_log,
            auto_promote=not args.no_auto_promote,
        )
    print(f"Model card: {out / 'model_card.md'}")


def main(argv: Sequence[str] | None = None) -> int:
    run_local(_parse_args(argv))
    # A failed gate is a legitimate outcome, not a crash, so the exit code is 0
    # either way; the decision is in the output and the audit log.
    return 0


if __name__ == "__main__":
    sys.exit(main())
