"""Alias-based promotion against a real (SQLite-backed) MLflow registry."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("mlflow")
pytest.importorskip("lightgbm")

import mlflow
from mlflow import MlflowClient

from credit_risk.config import (
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    PREVIOUS_CHAMPION_ALIAS,
)
from credit_risk.data.synthetic import GeneratorConfig, generate_loans
from credit_risk.governance.gate import CheckResult, GateDecision
from credit_risk.governance.registry import (
    PromotionRefusedError,
    get_alias_version,
    promote_to_champion,
    register_challenger,
    rollback_champion,
)
from credit_risk.pipeline import (
    build_model_frame,
    fit_candidate,
    labeled,
    prepare_silver,
)
from credit_risk.training.pyfunc import PREDICTION_COLUMN, log_pyfunc_model

MODEL = "local.credit_risk.test_model"
PASSED = GateDecision("test-1", True, (CheckResult("x", True, 1.0, 0.5, ""),))
FAILED = GateDecision("test-1", False, (CheckResult("x", False, 0.1, 0.5, ""),))


@pytest.fixture(scope="module")
def train_frame() -> pd.DataFrame:
    raw = generate_loans(GeneratorConfig(n_loans=3_000, seed=4))
    return labeled(build_model_frame(prepare_silver(raw).valid))


@pytest.fixture
def client(tmp_path: Path) -> Iterator[MlflowClient]:
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    # Explicit artifact location: the default is ./mlruns under the working directory.
    mlflow.create_experiment("registry-test", artifact_location=(tmp_path / "artifacts").as_uri())
    mlflow.set_experiment("registry-test")
    yield MlflowClient()
    mlflow.set_tracking_uri(None)


def _log_model(frame: pd.DataFrame) -> tuple[str, str]:
    candidate = fit_candidate(frame, mitigation="none", n_trials=0)
    with mlflow.start_run() as run:
        info = log_pyfunc_model(candidate.pipeline, frame)
    return info.model_uri, run.info.run_id


def test_logged_model_serves_probabilities(client: MlflowClient, train_frame: pd.DataFrame) -> None:
    uri, _ = _log_model(train_frame)
    predictions = mlflow.pyfunc.load_model(uri).predict(train_frame.head(20))
    assert list(predictions.columns) == [PREDICTION_COLUMN]
    assert predictions[PREDICTION_COLUMN].between(0, 1).all()


def test_alias_lifecycle(client: MlflowClient, train_frame: pd.DataFrame) -> None:
    assert get_alias_version(client, MODEL, CHAMPION_ALIAS) is None

    uri, run_id = _log_model(train_frame)
    v1 = register_challenger(client, MODEL, uri, run_id, PASSED)
    assert get_alias_version(client, MODEL, CHALLENGER_ALIAS) == v1
    assert promote_to_champion(client, MODEL, v1, approved_by="alice") is None

    uri, run_id = _log_model(train_frame)
    v2 = register_challenger(client, MODEL, uri, run_id, PASSED)
    assert promote_to_champion(client, MODEL, v2, approved_by="bob") == v1
    assert get_alias_version(client, MODEL, CHAMPION_ALIAS) == v2
    assert get_alias_version(client, MODEL, PREVIOUS_CHAMPION_ALIAS) == v1
    assert client.get_model_version(MODEL, v2).tags["promoted_by"] == "bob"

    assert rollback_champion(client, MODEL, approved_by="carol") == v1
    assert get_alias_version(client, MODEL, CHAMPION_ALIAS) == v1


def test_failed_gate_cannot_be_registered(client: MlflowClient, train_frame: pd.DataFrame) -> None:
    uri, run_id = _log_model(train_frame)
    with pytest.raises(PromotionRefusedError):
        register_challenger(client, MODEL, uri, run_id, FAILED)


def test_version_without_gate_tag_cannot_be_promoted(
    client: MlflowClient, train_frame: pd.DataFrame
) -> None:
    uri, _ = _log_model(train_frame)
    version = str(mlflow.register_model(uri, MODEL).version)
    with pytest.raises(PromotionRefusedError, match="no passing gate decision"):
        promote_to_champion(client, MODEL, version, approved_by="mallory")
