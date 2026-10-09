# Databricks MLOps with MLflow and Responsible AI

Rural credit risk scoring for agricultural producers in Brazil: a lakehouse
pipeline on Databricks, MLflow tracking and Unity Catalog model governance, and
a promotion gate that treats fairness, calibration, and drift as release
criteria alongside accuracy.

> **Portfolio project.** All data is synthetic and fictitious, generated with
> deliberately injected bias so the fairness tooling has something real to find.
> The bundle has not been deployed to a Databricks workspace; the package,
> tests, and local end-to-end run were executed locally. No metrics in this
> repository describe a real lending portfolio.

## Problem

Crop-financing loans fund a planting season. The model estimates the
probability that a loan defaults before maturity, and applications above a
policy threshold are flagged for decline or analyst review. Mistakes are not
symmetric: flagging a producer who would have repaid can cost them a season's
credit. Three attributes are treated as sensitive and evaluated, never used as
model inputs: **region**, **gender of the primary holder**, and **farm-size
band**. See [docs/RESPONSIBLE_AI.md](docs/RESPONSIBLE_AI.md) for why exclusion
alone does not make a model fair.

## Lakehouse flow

```mermaid
flowchart LR
    L[UC volume<br/>landing CSVs] -->|Auto Loader,<br/>enforced schema| B[(bronze)]
    B -->|dedupe, clean,<br/>DQ rules| S[(silver)]
    B -.->|rule violations| Q[(quarantine)]
    S -->|point-in-time snapshots| FT[(producer feature table)]
    S --> TR[train<br/>LightGBM + Optuna]
    FT -->|FeatureLookup| TR
    TR --> GT{promotion gate<br/>policy vX}
    GT -->|pass| RG[register<br/>@challenger]
    RG -->|dev/staging auto,<br/>prod human-approved| CH[@champion]
    CH --> BS[batch score]
    BS --> P[(inference table)]
    P --> LM[Lakehouse Monitoring<br/>sliced by sensitive attrs]
    BS --> D[(PSI drift)]
    GT --> A[(audit table)]
```

More detail, including the job graph and design decisions:
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## MLflow patterns used

| Pattern | Where |
|---|---|
| Parent run per training job, **nested runs per Optuna trial** | `tracking.nested_trial_logger`, `training/tuning.py` |
| Params, metrics, and per-group fairness / calibration tables as artifacts | `tracking.log_evaluation` |
| **pyfunc wrapper** returning P(default), with **signature** and **input example** | `training/pyfunc.py` |
| Model logged through **Feature Engineering** so lookups travel with the model | `notebooks/04_train.py` (`fe.log_model`) |
| **Unity Catalog registry with aliases** (`@challenger`, `@champion`, `@previous_champion`), no stages | `governance/registry.py` |
| Gate decision as run tags, a metric, and a JSON artifact; **model card** logged with the run | `tracking.log_gate_decision`, `responsible_ai/model_card.py` |
| SHAP global importance and per-application reason codes as artifacts | `responsible_ai/explain.py` |
| Batch scoring by alias with `fe.score_batch` | `lakehouse/scoring.py` |

## Responsible-AI approach

- **Controlled bias in the data.** The generator injects measurement bias
  (women's credit history under-recorded) and label bias (Nordeste
  renegotiations recorded as defaults); tests assert both exist as designed.
- **Fairness metrics** with fairlearn `MetricFrame`: selection (flag) rate, TPR,
  FPR, demographic parity difference, equalized odds difference, plus FPR
  difference as the primary harm metric.
- **Mitigation compared, not assumed.** Every run evaluates baseline,
  reweighing (deployable: attributes used only to weight training rows), and an
  equalized-odds ThresholdOptimizer (benchmark only: it would need gender at
  decision time).
- **Calibration** overall and per group (ECE, Brier, reliability tables).
- **Explainability** via SHAP, logged with the model.
- **Versioned promotion policy** in [`conf/promotion_policy.yml`](conf/promotion_policy.yml):
  minimum ROC AUC, maximum degradation vs champion, maximum ECE, fairness
  ceilings per attribute, minimum group size, no fairness regression vs
  champion, PSI drift limits. Decisions go to MLflow and a Delta audit table.
- **Human oversight.** Production promotion requires a named approver; rollback
  is one alias move. Details in [docs/RESPONSIBLE_AI.md](docs/RESPONSIBLE_AI.md);
  the generated card is described in [docs/MODEL_CARD_TEMPLATE.md](docs/MODEL_CARD_TEMPLATE.md).

## Run locally

Requires Python 3.11+. No Spark or Databricks workspace needed: the local runner
collapses the job graph into one process, using the same pandas stage functions
as the notebooks and a SQLite-backed MLflow tracking server and model registry.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

pytest                               # unit tests
ruff check src tests notebooks && mypy

credit-risk-local --output-dir .local_run --n-trials 10
credit-risk-local --output-dir .local_run --mitigation none   # baseline, expected to fail the gate
mlflow ui --backend-store-uri sqlite:///.local_run/mlflow.db
```

Each run prints the gate result, writes `.local_run/model_card.md`, appends to
`.local_run/promotion_audit.jsonl`, and, when the gate passes, registers a
version, sets `@challenger`, and moves `@champion` (pass `--no-auto-promote` to
mimic production). A second run compares the new candidate against the
registered champion.

## Deploy with Databricks Asset Bundles

Prerequisites: Databricks CLI, a workspace with Unity Catalog, and existing
catalogs `dev_credit` / `staging_credit` / `prod_credit` (the bundle creates the
schema, volume, registered model, and experiment inside them).

```bash
databricks bundle validate -t dev
databricks bundle deploy -t dev                  # builds the wheel, deploys jobs
databricks bundle run credit_risk_seed_demo_data -t dev
databricks bundle run credit_risk_training -t dev
databricks bundle run credit_risk_batch_scoring -t dev

# production promotion is manual and attributed
databricks bundle run credit_risk_promote -t prod \
  --params action=promote,version=3,approved_by=jane.doe@example.com
```

Production deploys run as a service principal
(`--var service_principal=<application-id>`). The staging and prod targets also
declare a model serving endpoint; see the bootstrap note in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#environments).

## Repository layout

```
databricks.yml              bundle: variables, dev/staging/prod targets, serving endpoint
resources/                  jobs (training, scoring, promote, seed) and UC objects
conf/promotion_policy.yml   versioned promotion policy
notebooks/                  thin Databricks entry points with widgets
src/credit_risk/            the package (Spark-free except lakehouse/)
tests/                      pytest suite
docs/                       architecture, responsible AI, model card template
.ci/workflow.yml            GitHub Actions workflow (to be moved to .github/workflows)
```

## What is and is not verified

- Verified locally: unit tests (data quality, features, generator bias, fairness
  metrics, gate decisions, PSI, model card, training, SHAP, MLflow alias
  lifecycle on a SQLite registry), ruff, mypy, and repeated end-to-end
  `credit-risk-local` runs.
- Not verified: anything that needs a workspace, namely the Spark modules in
  `src/credit_risk/lakehouse/`, the notebooks, Feature Engineering calls,
  Lakehouse Monitoring, the serving endpoint, and `databricks bundle validate`.
