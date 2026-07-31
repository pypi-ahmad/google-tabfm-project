"""TabFM preparation, schema alignment, prediction, and evaluation."""

import hashlib
import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal, Protocol, cast

import numpy as np
import pandas as pd

from .analytics import EvaluationDiagnostics, evaluate_predictions
from .config import assert_model_use_allowed

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
    diagnostics: EvaluationDiagnostics | None = None


def _normalize_feature_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = [str(column) for column in frame.columns]
    if len(set(normalized)) != len(normalized):
        raise InferenceError("Feature column names collide after normalization to strings.")
    result = frame.copy()
    result.columns = normalized
    return result


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


def align_features(
    frame: pd.DataFrame,
    expected_columns: list[str],
    expected_dtypes: dict[str, Any] | None = None,
) -> SchemaAlignment:
    """Align prediction data to prepared context schema with explicit diagnostics."""
    if frame.columns.duplicated().any():
        raise InferenceError("Prediction data contains duplicate column names.")
    missing = tuple(column for column in expected_columns if column not in frame.columns)
    extra = tuple(column for column in frame.columns if column not in expected_columns)
    aligned = frame.drop(columns=list(extra), errors="ignore").copy()
    for column in missing:
        dtype = expected_dtypes.get(column) if expected_dtypes else None
        aligned[column] = _null_column(aligned.index, dtype) if dtype is not None else pd.NA
    return SchemaAlignment(aligned.loc[:, expected_columns], missing, extra)


def _null_column(index: pd.Index, dtype: Any) -> Any:
    """Build an all-missing column typed as `dtype`, or fall back to plain `pd.NA`.

    Some numpy dtypes (bool, plain int) have no null representation: constructing
    a Series from them without explicit values can silently produce non-null
    "garbage" (e.g. dtype="bool" yields real True/False, not missing) instead of
    raising. Verify the result is actually all-null before trusting it.
    """
    try:
        candidate = pd.Series(index=index, dtype=dtype)
    except (TypeError, ValueError):
        return pd.NA
    return candidate if candidate.isna().all() else pd.NA


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
        self.feature_dtypes: dict[str, Any] = {}
        self.is_prepared = False

    def prepare(self, features: pd.DataFrame, target: pd.Series) -> "PreparedPredictor":
        features = _normalize_feature_columns(features)
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
        self.feature_columns = list(features.columns)
        self.feature_dtypes = dict(zip(features.columns, features.dtypes, strict=True))
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
        alignment = align_features(
            _normalize_feature_columns(features), self.feature_columns, self.feature_dtypes
        )
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
        diagnostics = (
            evaluate_predictions(self.task, expected, predictions, probabilities)
            if expected is not None and expected.reindex(predictions.index).notna().any()
            else None
        )
        metrics = dict(diagnostics.metrics) if diagnostics is not None else {}
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
            diagnostics=diagnostics,
        )


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


def load_tabfm_predictor(
    task: TaskType, *, device: str = "auto", accept_non_commercial_license: bool
) -> PreparedPredictor:
    """Load official PyTorch checkpoint and configured sklearn wrapper."""
    assert_model_use_allowed(accept_non_commercial_license)
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
