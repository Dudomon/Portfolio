"""Lakehouse Monitoring (inference profile) on the predictions table.

The monitor is created by the scoring job rather than declared in the bundle:
a monitor can only be attached to an existing table, and the predictions table
does not exist until the first scoring run. Creation is idempotent.

Slicing by the sensitive attributes makes the monitor compute performance and
drift per region, gender, and farm-size band once labels are back-filled,
which is the production counterpart of the fairness checks in the gate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from credit_risk.config import LABEL, SENSITIVE_ATTRIBUTES

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient

PREDICTED_LABEL = "predicted_default"
PREDICTED_PROBA = "prediction"
MODEL_ID_COLUMN = "model_version"
TIMESTAMP_COLUMN = "scored_at"


def ensure_inference_monitor(
    workspace: WorkspaceClient,
    *,
    table_name: str,
    output_schema: str,
    assets_dir: str,
) -> None:
    """Create the monitor if missing, then trigger a refresh."""
    from databricks.sdk.errors import NotFound
    from databricks.sdk.service.catalog import (
        MonitorCronSchedule,
        MonitorInferenceLog,
        MonitorInferenceLogProblemType,
    )

    try:
        workspace.quality_monitors.get(table_name=table_name)
    except NotFound:
        workspace.quality_monitors.create(
            table_name=table_name,
            assets_dir=assets_dir,
            output_schema_name=output_schema,
            inference_log=MonitorInferenceLog(
                problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
                prediction_col=PREDICTED_LABEL,
                prediction_proba_col=PREDICTED_PROBA,
                label_col=LABEL,
                model_id_col=MODEL_ID_COLUMN,
                timestamp_col=TIMESTAMP_COLUMN,
                # Weekly windows smooth the seasonality of crop-financing demand.
                granularities=["1 day", "1 week"],
            ),
            slicing_exprs=list(SENSITIVE_ATTRIBUTES),
            schedule=MonitorCronSchedule(
                quartz_cron_expression="0 0 6 * * ?", timezone_id="America/Sao_Paulo"
            ),
        )
    workspace.quality_monitors.run_refresh(table_name=table_name)
