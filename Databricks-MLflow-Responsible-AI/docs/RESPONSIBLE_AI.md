# Responsible AI approach

This document explains which fairness definitions the project uses, why, what
the trade-offs are, and where the approach stops working. The data is
synthetic, with bias injected on purpose (see `src/credit_risk/data/synthetic.py`),
so the tooling has a known ground truth to detect. Nothing here is legal
advice. Brazilian context (LGPD, Central Bank rules on rural credit) is
referenced to explain design choices, not to claim compliance.

## The decision and who it affects

The model estimates the probability that a crop-financing loan defaults.
Applications at or above the policy threshold (`decision_threshold` in
`conf/promotion_policy.yml`) are **flagged**: declined or sent to manual
review. Two errors matter:

| Error | Who bears it | Metric |
|---|---|---|
| A producer who would have repaid is flagged | The producer: loses or delays credit for a planting season | False positive rate (FPR) |
| A producer who will default is not flagged | The lender (and, through renegotiation, often the producer too) | 1 - true positive rate (TPR) |

For a credit decision affecting livelihoods, the first error is the more direct
harm, and it is the one the gate weighs most heavily.

## Sensitive attributes

| Attribute | Why it is sensitive here |
|---|---|
| `region` (macro-region) | Climate exposure and historical credit practices differ strongly between regions; the Nordeste has a history of drought-driven renegotiation. |
| `gender` of the primary holder | Women farmers are under-represented in formal rural credit, and their credit history is often recorded under a spouse. |
| `farm_size_band` | Smallholders (family farming) have different access to insurance and subsidized credit lines. |

### Why they are excluded from training but kept for evaluation

1. **Disparate treatment.** Using gender or region as a direct input would mean
   two otherwise identical applicants receive different scores because of who
   they are. That is not acceptable for this use, and gender is sensitive
   personal data under LGPD.
2. **Exclusion is not fairness.** Removing an attribute does not remove its
   signal. `rainfall_anomaly` is correlated with region, and
   `credit_history_months` is systematically under-recorded for women. A model
   blind to gender can still treat women worse, which is exactly what the
   generator's measurement bias produces.
3. **So they are needed to measure.** The attributes are carried next to the
   predictions (holdout evaluation, inference table) but never in the model's
   input schema. Tests enforce that `MODEL_FEATURES` and `SENSITIVE_ATTRIBUTES`
   are disjoint, and that shuffling the sensitive columns does not change a
   single prediction.

Collecting these attributes for monitoring needs its own legal basis and access
controls. In a real deployment they would live in a restricted column or table
with Unity Catalog grants limited to the model risk function.

## Fairness definitions chosen

All metrics are computed with fairlearn's `MetricFrame` per attribute at the
policy's decision threshold (`credit_risk/responsible_ai/fairness.py`).

| Metric | Definition | Role in the gate |
|---|---|---|
| FPR difference | max - min across groups of P(flagged given repaid) | **Primary.** Strict threshold. |
| Demographic parity difference | max - min of the flag rate | Secondary. Strict threshold. |
| Equalized odds difference | max of the TPR gap and FPR gap | Backstop. Loose threshold. |
| Group calibration | mean predicted vs observed rate, ECE, Brier per group | Reported in the model card. |

**Why FPR difference is primary.** It measures the harm that falls on the
producer, and it is estimated on the repaying majority of each group (thousands
of rows), so it is statistically stable.

**Why equalized odds only gets a loose backstop.** Its TPR half is estimated on
each group's defaulters, which in the holdout can be fewer than a hundred
(for example, Norte). A strict EO threshold would make the gate pass or fail on
sampling noise. The loose threshold still catches gross failures.

**Why demographic parity is not the primary metric.** Base rates genuinely
differ: smallholders without crop insurance do default more often, and a model
that flagged them at exactly the same rate as large insured farms would be
mis-pricing risk. Parity is still bounded, because a large flag-rate gap is a
signal worth a human look even when it is partly justified.

**Why not calibration within groups as a hard gate.** Calibration within groups
and equal error rates cannot generally both hold when base rates differ
(Kleinberg et al., 2016; Chouldechova, 2017). The policy prioritizes error-rate
parity on the harmful error and reports group calibration so reviewers see the
cost.

## Mitigation and its trade-offs

Every training run compares three variants on the same holdout, using the
candidate's hyperparameters (`compare_mitigations` in `pipeline.py`):

1. **Baseline**: no mitigation.
2. **Reweighing** (Kamiran and Calders, 2012): training rows are weighted so the
   label is independent of the intersection of sensitive groups. The deployed
   model still never sees the attributes. **This is the deployable option and
   the default.**
3. **ThresholdOptimizer** (Hardt et al., 2016) under equalized odds, on the
   attribute with the largest baseline gap. It needs the sensitive attribute
   at decision time, so it is **a benchmark only**: it shows what an
   equalized-odds decision rule would cost, not something we would ship.

What a reviewer should expect to see in the model card on the default synthetic
data (run `credit-risk-local` to reproduce; exact values depend on the seed):

- The baseline fails the gate on region and farm-size FPR and parity gaps.
- Reweighing narrows those gaps enough to pass, at a small cost in ROC AUC.
- Reweighing shifts predicted risk between groups, so **group calibration gets
  worse** (for example, predicted risk for the Nordeste falls below its observed
  default rate). This is the trade-off made explicit, and it matters if the score
  is ever used for pricing, which the card lists as out of scope.
- The gender gap shrinks only slightly. The cause is measurement bias in a
  feature, which reweighing labels cannot fix. The real fix is data: link
  spouse-registered credit history, or replace the feature.
- The Nordeste label bias (renegotiations recorded as defaults) is learned by
  every variant. Reweighing balances its effect; correcting the labels at the
  source is the real fix.

## Explainability

- **Global:** mean absolute SHAP value per feature (one-hot columns folded back
  to `crop_type`), logged as `explanations/global_importance.csv` and shown in
  the model card.
- **Local:** for the highest-risk holdout applications, the top factors with
  value, log-odds contribution, and direction, logged as
  `explanations/local_reason_codes.json`. These are the raw material for
  adverse-action style explanations to a declined producer and for analysts
  reviewing a flagged case.

SHAP explains the model, not the world: a large contribution from
`rainfall_anomaly` means the model relies on it, not that rainfall causes
default for that producer.

## Human oversight

- **Decision level:** the score supports an analyst; flagged applications are
  reviewed, overrides are allowed and must record a reason. Fully automated
  declines are out of scope (model card).
- **Model level:** the gate is necessary, not sufficient. In production,
  `@champion` only moves through the `credit-risk-promote` job, which requires a
  named `approved_by`, re-checks the version's gate tag, and writes to the
  audit table. Dev and staging auto-promote to keep iteration fast.
- **Policy level:** thresholds live in `conf/promotion_policy.yml`. Changing one
  is a reviewed pull request that bumps `version`; every gate decision records
  the policy version it was made under.
- **Production level:** the inference table is monitored with Lakehouse
  Monitoring sliced by the three attributes, and PSI drift is written per
  scoring batch. Labels are back-filled as loans mature, after which the
  monitor computes performance and fairness by slice. Sustained divergence
  should trigger a model review and, if needed, a rollback (one alias move).

## Limits

- **Synthetic data.** The biases are simple and known. Real bias is messier,
  interacts across attributes, and is rarely labeled.
- **Group fairness only.** Nothing here guarantees individual fairness or
  checks counterfactual fairness.
- **Intersectional gaps** (for example, women smallholders in the Nordeste) are
  used for reweighing but not gated, because intersections become too small to
  estimate reliably. They should be reviewed manually when sample sizes allow.
- **Label delay.** Outcomes arrive months after scoring, so production fairness
  monitoring always lags.
- **Thresholds are judgment calls.** The numbers in the policy file are
  defensible defaults for a demonstration, not values derived from regulation.
