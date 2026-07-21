"""TabFM preparation, schema alignment, prediction, and evaluation."""

import hashlib
import logging
from contextlib import suppress
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal, Protocol, cast

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

TaskType = Literal["classification", "regression"]
MetricValue = float | int
logger = logging.getLogger(__name__)


class InferenceError(ValueError):
    """Raised when data cannot be prepared or predicted safely."""


class Estimator(Protocol):
    def fit(self, features: pd.DataFrame, target: pd.Series) -> Any: ...

    def predict(self, features: pd.DataFrame) -> np.ndarray: ...


@dataclass(frozen=True)
class TaskSuggestion:
    task: TaskType
    rationale: str


@dataclass(frozen=True)
class SchemaAlignment:
    frame: pd.DataFrame
    missing_columns: tuple[str, ...]
    extra_columns: tuple[str, ...]


@dataclass(frozen=True)
class PredictionResult:
    predictions: pd.Series
    probabilities: pd.DataFrame | None
    metrics: dict[str, MetricValue]
    warnings: tuple[str, ...]
    latency_ms: float
    device: str


def suggest_task(target: pd.Series) -> TaskSuggestion:
    """Suggest a task while leaving final choice to the user."""
    values = target.dropna()
    if values.empty:
        raise InferenceError("Target has no labeled values.")
    if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
        return TaskSuggestion("classification", "Target contains categorical values.")
    numeric = pd.to_numeric(values, errors="coerce")
    unique_count = numeric.nunique()
    integer_like = bool(np.isclose(numeric.to_numpy() % 1, 0).all())
    if integer_like and 2 <= unique_count <= 10:
        return TaskSuggestion(
            "classification",
            f"Target has {unique_count} integer-like values, within TabFM's 10-class limit.",
        )
    return TaskSuggestion("regression", "Target is numeric and appears continuous.")


def align_features(frame: pd.DataFrame, expected_columns: list[str]) -> SchemaAlignment:
    """Align prediction data to prepared context schema with explicit diagnostics."""
    if frame.columns.duplicated().any():
        raise InferenceError("Prediction data contains duplicate column names.")
    missing = tuple(column for column in expected_columns if column not in frame.columns)
    extra = tuple(column for column in frame.columns if column not in expected_columns)
    aligned = frame.drop(columns=list(extra), errors="ignore").copy()
    for column in missing:
        aligned[column] = pd.NA
    return SchemaAlignment(aligned.loc[:, expected_columns], missing, extra)


def context_fingerprint(features: pd.DataFrame, target: pd.Series, task: TaskType) -> str:
    """Fingerprint context values, schema, and task for stale-state invalidation."""
    digest = hashlib.sha256(task.encode("utf-8"))
    digest.update("\x1f".join(map(str, features.columns)).encode("utf-8"))
    digest.update("\x1f".join(map(str, features.dtypes)).encode("utf-8"))
    digest.update(np.asarray(pd.util.hash_pandas_object(features, index=True)).tobytes())
    digest.update(np.asarray(pd.util.hash_pandas_object(target, index=True)).tobytes())
    return digest.hexdigest()


class PreparedPredictor:
    """Official TabFM estimator prepared once for repeated predictions."""

    def __init__(self, task: TaskType, estimator: Estimator, *, device: str) -> None:
        self.task = task
        self.estimator = estimator
        self.device = device
        self.feature_columns: list[str] = []
        self.is_prepared = False

    def prepare(self, features: pd.DataFrame, target: pd.Series) -> "PreparedPredictor":
        if len(features) < 2:
            raise InferenceError("At least two labeled context rows are required.")
        if features.columns.duplicated().any():
            raise InferenceError("Context data contains duplicate column names.")
        if features.shape[1] == 0:
            raise InferenceError("At least one feature column is required.")
        if features.shape[1] > 500:
            raise InferenceError("TabFM supports at most 500 input features in this workbench.")
        if target.isna().any():
            raise InferenceError("Prepared context target cannot contain missing values.")

        prepared_target = target
        if self.task == "classification":
            classes = target.nunique(dropna=True)
            if not 2 <= classes <= 10:
                raise InferenceError("Classification requires between 2 and 10 target classes.")
        else:
            prepared_target = pd.to_numeric(target, errors="coerce")
            if prepared_target.isna().any():
                raise InferenceError("Regression target must contain only numeric labeled values.")

        self.estimator.fit(features, prepared_target)
        self.feature_columns = [str(column) for column in features.columns]
        self.is_prepared = True
        return self

    def predict(
        self,
        features: pd.DataFrame,
        *,
        expected: pd.Series | None = None,
    ) -> PredictionResult:
        if not self.is_prepared:
            raise InferenceError("Prepare the model context before prediction.")
        if features.empty:
            raise InferenceError("At least one prediction row is required.")
        alignment = align_features(features, self.feature_columns)
        warnings = _alignment_warnings(alignment)
        started = perf_counter()
        values = self.estimator.predict(alignment.frame)
        predictions = pd.Series(values, index=alignment.frame.index, name="prediction")
        probabilities: pd.DataFrame | None = None
        if self.task == "classification":
            classifier = cast(Any, self.estimator)
            probabilities = pd.DataFrame(
                classifier.predict_proba(alignment.frame),
                index=alignment.frame.index,
                columns=list(classifier.classes_),
            )
        latency_ms = (perf_counter() - started) * 1000
        metrics = self._evaluate(predictions, probabilities, expected)
        logger.info(
            "tabfm_prediction_complete task=%s test_rows=%d features=%d latency_ms=%.1f device=%s",
            self.task,
            len(alignment.frame),
            len(self.feature_columns),
            latency_ms,
            self.device,
        )
        return PredictionResult(
            predictions=predictions,
            probabilities=probabilities,
            metrics=metrics,
            warnings=warnings,
            latency_ms=latency_ms,
            device=self.device,
        )

    def _evaluate(
        self,
        predictions: pd.Series,
        probabilities: pd.DataFrame | None,
        expected: pd.Series | None,
    ) -> dict[str, MetricValue]:
        if expected is None:
            return {}
        labeled = expected.reindex(predictions.index).notna()
        if not labeled.any():
            return {}
        actual = expected.reindex(predictions.index).loc[labeled]
        predicted = predictions.loc[labeled]
        metrics: dict[str, MetricValue] = {"evaluated_rows": int(labeled.sum())}
        if self.task == "classification":
            metrics["accuracy"] = float(accuracy_score(actual, predicted))
            if probabilities is not None:
                with suppress(ValueError):
                    metrics["log_loss"] = float(
                        log_loss(actual, probabilities.loc[labeled], labels=probabilities.columns)
                    )
            return metrics
        numeric_actual = pd.to_numeric(actual, errors="coerce")
        valid = numeric_actual.notna()
        numeric_predicted = pd.to_numeric(predicted.loc[valid], errors="coerce")
        numeric_actual = numeric_actual.loc[valid]
        metrics["evaluated_rows"] = int(valid.sum())
        if not valid.any():
            return metrics
        metrics["mae"] = float(mean_absolute_error(numeric_actual, numeric_predicted))
        metrics["rmse"] = float(np.sqrt(mean_squared_error(numeric_actual, numeric_predicted)))
        if valid.sum() >= 2:
            metrics["r2"] = float(r2_score(numeric_actual, numeric_predicted))
        return metrics


def _alignment_warnings(alignment: SchemaAlignment) -> tuple[str, ...]:
    warnings: list[str] = []
    if alignment.missing_columns:
        warnings.append(
            "Missing columns filled with nulls: " + ", ".join(alignment.missing_columns)
        )
    if alignment.extra_columns:
        warnings.append("Extra columns ignored: " + ", ".join(alignment.extra_columns))
    return tuple(warnings)


def resolve_device(requested: str = "auto") -> str:
    """Choose CUDA when available, otherwise provide CPU fallback."""
    if requested in {"cuda", "cpu"}:
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_tabfm_predictor(task: TaskType, *, device: str = "auto") -> PreparedPredictor:
    """Load official PyTorch checkpoint and configured sklearn wrapper."""
    try:
        from tabfm import TabFMClassifier, TabFMRegressor, tabfm_v1_0_0_pytorch
    except ImportError as exc:
        raise RuntimeError(
            "TabFM runtime is not installed. Run `uv sync --extra cu130` or `uv sync --extra cpu`."
        ) from exc

    resolved_device = resolve_device(device)
    logger.info("tabfm_model_load task=%s device=%s", task, resolved_device)
    model = tabfm_v1_0_0_pytorch.load(model_type=task, device=resolved_device)
    options: dict[str, Any] = {
        "model": model,
        "n_estimators": 8,
        "batch_size": 1,
        "max_num_features": 500,
        "max_num_rows": 5000,
        "random_state": 42,
        "cache_context": True,
        "maybe_quantize_kv_cache": True,
        "keep_cache_on_device": False,
    }
    estimator = (
        TabFMClassifier(**options) if task == "classification" else TabFMRegressor(**options)
    )
    return PreparedPredictor(task, estimator, device=resolved_device)
