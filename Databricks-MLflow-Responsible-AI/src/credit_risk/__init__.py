"""Rural credit risk scoring with MLflow governance and responsible-AI checks.

Modules are split so that everything except :mod:`credit_risk.lakehouse` runs on
plain pandas/numpy. The lakehouse package is the only place that touches Spark,
Delta, or the Databricks Feature Engineering client, which keeps the business
logic unit-testable on a laptop and in CI.
"""

__version__ = "0.1.0"
