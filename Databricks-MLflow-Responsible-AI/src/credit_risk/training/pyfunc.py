"""MLflow pyfunc wrapper that returns P(default) as a single column.

A plain sklearn flavor would serve ``predict`` (hard 0/1 labels at an implicit
0.5 cut). The decision threshold belongs to the versioned promotion policy, so
the served artifact returns calibrated probabilities and the threshold is
applied downstream where it is governed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.pyfunc.model import PythonModel
from sklearn.pipeline import Pipeline

import credit_risk
from credit_risk.config import MODEL_FEATURES
from credit_risk.training.model import feature_matrix

PREDICTION_COLUMN = "default_probability"


class DefaultProbabilityModel(PythonModel):
    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline

    def predict(self, context: Any, model_input: pd.DataFrame, params: Any = None) -> pd.DataFrame:
        proba = self.pipeline.predict_proba(feature_matrix(model_input))[:, 1]
        return pd.DataFrame({PREDICTION_COLUMN: np.asarray(proba, dtype=float)})


def package_code_path() -> str:
    """Ship the package source with the model so serving can unpickle the wrapper."""
    return str(Path(credit_risk.__file__).parent)


def model_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Model inputs with their contract dtypes (integers stay integers).

    The signature mirrors what callers actually send (BIGINT columns from
    Spark), because MLflow refuses to coerce int64 to double. Integer features
    cannot be null here: the silver rules quarantine such rows.
    """
    return df.loc[:, list(MODEL_FEATURES)]


def model_signature(pipeline: Pipeline, example: pd.DataFrame) -> Any:
    inputs = model_inputs(example)
    outputs = DefaultProbabilityModel(pipeline).predict(None, inputs)
    return infer_signature(inputs, outputs)


def log_pyfunc_model(pipeline: Pipeline, example: pd.DataFrame, name: str = "model") -> Any:
    """Log locally (outside Feature Engineering) with signature and input example."""
    input_example = model_inputs(example.head(5))
    return mlflow.pyfunc.log_model(
        name=name,
        python_model=DefaultProbabilityModel(pipeline),
        signature=model_signature(pipeline, input_example),
        input_example=input_example,
        code_paths=[package_code_path()],
        metadata={"features": list(MODEL_FEATURES)},
    )
