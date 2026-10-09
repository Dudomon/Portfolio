# Model card template

The model card is generated, not hand-written. The template is
`src/credit_risk/responsible_ai/templates/model_card.md.j2`, rendered by
`render_model_card` and logged as `model_card.md` in the training run. The
registered model version links back to that run through its `source_run_id` tag.

## Why generate it

- **The numbers match the decision.** The card is rendered from the same
  `EvaluationReport` and `GateDecision` objects that decided the promotion. A
  hand-written card drifts from reality after the first retrain.
- **Missing data fails loudly.** Jinja2 runs with `StrictUndefined`; a missing
  field raises instead of rendering an empty cell (covered by
  `tests/test_model_card.py`).
- **It is reviewable.** Markdown diffs cleanly, so two versions' cards can be
  compared in a pull request or in the MLflow UI.

## Sections and their sources

| Section | Source | Static or generated |
|---|---|---|
| Header (run id, timestamp, synthetic-data warning) | `ModelCardContext` | generated |
| Intended use, decision threshold | policy `decision_threshold` | generated threshold, static text |
| Out-of-scope uses | template | static |
| Training data (source table, cutoff, sizes, default rates) | `TrainingDataSummary` from the temporal split | generated |
| Features used / excluded attributes | `MODEL_FEATURES`, `SENSITIVE_ATTRIBUTES` | generated |
| Hyperparameters | Optuna best params | generated |
| Performance (ROC AUC, PR AUC, Brier, ECE, flag rate) | `EvaluationReport.performance` on the holdout | generated |
| Fairness per attribute (DP, FPR and EO differences; per-group flag rate, TPR, FPR, mean predicted vs observed) | `FairnessSummary` + group calibration | generated |
| Mitigation comparison (baseline, reweighing, ThresholdOptimizer benchmark) | `MitigationComparison` rows | generated |
| Explainability (mean absolute SHAP) | `Explanations.global_importance` | generated |
| Promotion gate (every check, observed vs threshold) | `GateDecision` | generated |
| Known limitations | template | static, reviewed with each policy change |
| Human oversight | template | static |

## How the pieces reach the renderer on Databricks

The training task and the gate task are separate job tasks, so the card's
training-time inputs are persisted on the run and read back:

| Artifact | Written by | Read by |
|---|---|---|
| `governance/training_summary.json` | `log_card_inputs` (train) | `load_card_inputs` (gate) |
| `governance/hyperparameters.json` | `log_card_inputs` (train) | `load_card_inputs` (gate) |
| `governance/mitigation_comparison.json` | `log_card_inputs` (train) | `load_card_inputs` (gate) |
| `explanations/global_importance.csv` | `log_card_inputs` (train) | `load_card_inputs` (gate) |
| `governance/gate_decision.json` | `log_gate_decision` (gate) | `load_gate_decision` (register) |

The gate task then renders the card with the freshly computed holdout
evaluation and the gate decision, and logs `model_card.md` to the same run.

## Changing the template

- Static sections (limitations, oversight, out-of-scope uses) are part of the
  governance record. Change them in the same pull request as the policy change
  or code change that motivates them.
- New generated fields go into `ModelCardContext` (or `CardInputs` if they are
  computed at training time) so `StrictUndefined` keeps protecting them.
- Run `pytest tests/test_model_card.py` and `credit-risk-local` to see the
  rendered output.
