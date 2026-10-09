"""Unity Catalog model registry operations.

Promotion is expressed only through aliases (``challenger``, ``champion``,
``previous_champion``). Model stages are deprecated and unsupported in Unity
Catalog; aliases are also atomic pointers, so rolling back is one call that
moves ``champion`` back to ``previous_champion``.
"""

from __future__ import annotations

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from credit_risk.config import CHALLENGER_ALIAS, CHAMPION_ALIAS, PREVIOUS_CHAMPION_ALIAS
from credit_risk.governance.gate import GateDecision


class PromotionRefusedError(RuntimeError):
    """Raised when a version without a passing gate decision is promoted."""


def get_alias_version(client: MlflowClient, model_name: str, alias: str) -> str | None:
    try:
        return str(client.get_model_version_by_alias(model_name, alias).version)
    except MlflowException as exc:
        # A missing alias or a not-yet-created model both mean "no version";
        # anything else (auth, network) must propagate.
        if exc.error_code in {"RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE", "NOT_FOUND"}:
            return None
        raise


def register_challenger(
    client: MlflowClient, model_name: str, model_uri: str, run_id: str, decision: GateDecision
) -> str:
    """Register a gate-approved model version and point ``challenger`` at it."""
    if not decision.passed:
        raise PromotionRefusedError("Only gate-approved models are registered as challenger")
    version = str(mlflow.register_model(model_uri, model_name).version)
    for key, value in {
        "gate.passed": "true",
        "gate.policy_version": decision.policy_version,
        "source_run_id": run_id,
    }.items():
        client.set_model_version_tag(model_name, version, key, value)
    client.set_registered_model_alias(model_name, CHALLENGER_ALIAS, version)
    return version


def promote_to_champion(
    client: MlflowClient, model_name: str, version: str, approved_by: str
) -> str | None:
    """Move ``champion`` to ``version``; returns the previous champion version.

    Re-checks the version's gate tag so a manual job cannot promote something
    the gate never approved.
    """
    tags = client.get_model_version(model_name, version).tags
    if tags.get("gate.passed") != "true":
        raise PromotionRefusedError(
            f"{model_name} v{version} has no passing gate decision; refusing to promote"
        )
    previous = get_alias_version(client, model_name, CHAMPION_ALIAS)
    if previous == version:
        return previous
    if previous is not None:
        client.set_registered_model_alias(model_name, PREVIOUS_CHAMPION_ALIAS, previous)
    client.set_registered_model_alias(model_name, CHAMPION_ALIAS, version)
    client.set_model_version_tag(model_name, version, "promoted_by", approved_by)
    return previous


def rollback_champion(client: MlflowClient, model_name: str, approved_by: str) -> str:
    """Point ``champion`` back to ``previous_champion``."""
    previous = get_alias_version(client, model_name, PREVIOUS_CHAMPION_ALIAS)
    if previous is None:
        raise PromotionRefusedError("No previous champion to roll back to")
    client.set_registered_model_alias(model_name, CHAMPION_ALIAS, previous)
    client.set_model_version_tag(model_name, previous, "rolled_back_by", approved_by)
    return previous
