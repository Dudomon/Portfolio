# Databricks notebook source
# MAGIC %md
# MAGIC # 00 - Seed synthetic landing files
# MAGIC
# MAGIC Writes **fictitious** loan applications as CSV files into the Unity Catalog
# MAGIC volume that Auto Loader watches. Historical applications carry outcomes
# MAGIC (training); new applications do not (scoring). In a real deployment this
# MAGIC notebook would not exist: an upstream system lands the files.

# COMMAND ----------

dbutils.widgets.text("catalog", "dev_credit")
dbutils.widgets.text("schema", "credit_risk")
dbutils.widgets.text("n_historical", "40000")
dbutils.widgets.text("n_new", "5000")
dbutils.widgets.text("seed", "42")

# COMMAND ----------

from datetime import UTC, date, datetime
from pathlib import Path

from credit_risk.config import UnityCatalogNames
from credit_risk.data.synthetic import GeneratorConfig, generate_loans

names = UnityCatalogNames(dbutils.widgets.get("catalog"), dbutils.widgets.get("schema"))
seed = int(dbutils.widgets.get("seed"))
# Synthetic rows must never be mixed into a production catalog.
if names.catalog.startswith("prod"):
    raise RuntimeError("Refusing to seed synthetic data into a production catalog")
# The schema and the `raw` volume are created by the bundle (resources/unity_catalog.yml).

# COMMAND ----------

batch_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
historical = generate_loans(
    GeneratorConfig(n_loans=int(dbutils.widgets.get("n_historical")), seed=seed)
)
# New applications start after the historical window and use another seed; the
# ID prefixes keep them from colliding with historical loans and producers.
new = generate_loans(
    GeneratorConfig(
        n_loans=int(dbutils.widgets.get("n_new")),
        start_date=date(2025, 1, 1),
        span_days=90,
        seed=seed + 1,
    ),
    include_label=False,
)
new["loan_id"] = "N" + new["loan_id"].str[1:]
new["producer_id"] = new["producer_id"].str.replace("P", "PN", n=1)

for dataset, frame in (("loan_applications", historical), ("new_applications", new)):
    target = Path(names.landing_path(dataset))
    target.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target / f"batch_{batch_id}.csv", index=False)
    print(f"Wrote {len(frame):,} rows to {target}")
