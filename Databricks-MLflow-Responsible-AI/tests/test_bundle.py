"""Static checks on the Asset Bundle and notebooks (no workspace needed)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from credit_risk.config import UnityCatalogNames


def _load(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data


def _jobs(project_root: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    jobs = {}
    for path in sorted((project_root / "resources").glob("*.yml")):
        for name, job in _load(path).get("resources", {}).get("jobs", {}).items():
            jobs[name] = (path, job)
    return jobs


def test_bundle_has_three_targets_with_distinct_catalogs(project_root: Path) -> None:
    bundle = _load(project_root / "databricks.yml")
    targets = bundle["targets"]
    assert set(targets) == {"dev", "staging", "prod"}
    catalogs = {name: t["variables"]["catalog"] for name, t in targets.items()}
    assert len(set(catalogs.values())) == 3
    assert targets["prod"]["variables"]["auto_promote"] == "false"
    assert targets["prod"]["mode"] == "production"


def test_every_task_points_to_an_existing_databricks_notebook(project_root: Path) -> None:
    for path, job in _jobs(project_root).values():
        for task in job["tasks"]:
            if "notebook_task" not in task:
                continue
            notebook = (path.parent / task["notebook_task"]["notebook_path"]).resolve()
            assert notebook.is_file(), notebook
            assert notebook.read_text(encoding="utf-8").startswith("# Databricks notebook source")


def test_task_dependencies_resolve(project_root: Path) -> None:
    for name, (_, job) in _jobs(project_root).items():
        keys = {task["task_key"] for task in job["tasks"]}
        for task in job["tasks"]:
            for dependency in task.get("depends_on", []):
                assert dependency["task_key"] in keys, (name, dependency)


def test_registration_only_runs_after_a_passing_gate(project_root: Path) -> None:
    _, job = _jobs(project_root)["credit_risk_training"]
    tasks = {task["task_key"]: task for task in job["tasks"]}
    assert tasks["register"]["depends_on"] == [{"task_key": "gate_passed", "outcome": "true"}]
    condition = tasks["gate_passed"]["condition_task"]
    assert condition["left"] == "{{tasks.evaluate_gate.values.gate_passed}}"


@pytest.mark.parametrize("notebook", sorted(Path(__file__).parents[1].glob("notebooks/*.py")))
def test_notebook_widgets_are_declared_before_use(notebook: Path) -> None:
    text = notebook.read_text(encoding="utf-8")
    declared = {
        line.split('"')[1]
        for line in text.splitlines()
        if line.startswith(("dbutils.widgets.text(", "dbutils.widgets.dropdown("))
    }
    used = {part.split('"')[0] for part in text.split('dbutils.widgets.get("')[1:]}
    assert used <= declared, used - declared


def test_unity_catalog_names_reject_injection() -> None:
    with pytest.raises(ValueError, match="Invalid"):
        UnityCatalogNames("dev; DROP TABLE x", "s")
    names = UnityCatalogNames("dev_credit", "credit_risk")
    assert names.silver("loan_applications") == "dev_credit.credit_risk.silver_loan_applications"
    assert (
        names.model_uri("champion")
        == "models:/dev_credit.credit_risk.rural_credit_default@champion"
    )
