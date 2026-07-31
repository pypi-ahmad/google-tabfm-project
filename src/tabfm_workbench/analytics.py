"""Pure exploratory-data-analysis and prediction-evaluation domain logic."""

from dataclasses import dataclass
from typing import Literal
from warnings import catch_warnings, simplefilter

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    explained_variance_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
    roc_curve,
)

TaskType = Literal["classification", "regression"]
MetricValue = float | int
COLUMN_QUALITY_COLUMNS = [
    "column",
    "dtype",
    "non_null_count",
    "missing_count",
    "missing_percent",
    "unique_count",
    "is_constant",
    "is_suspected_id",
    "is_high_cardinality",
]


@dataclass(frozen=True)
class DatasetOverview:
    rows: int
    columns: int
    memory_bytes: int
    duplicate_rows: int
    missing_cell_percent: float


@dataclass(frozen=True)
class EdaSnapshot:
    overview: DatasetOverview
    column_quality: pd.DataFrame
    numeric_summary: pd.DataFrame
    categorical_summary: pd.DataFrame
    sample: pd.DataFrame
    scatter_sample: pd.DataFrame
    correlations: pd.DataFrame


@dataclass(frozen=True)
class EvaluationDiagnostics:
    metrics: dict[str, MetricValue]
    warnings: tuple[str, ...]
    confusion_matrix: pd.DataFrame | None = None
    per_class: pd.DataFrame | None = None
    roc_curve: pd.DataFrame | None = None
    probability_diagnostics: pd.DataFrame | None = None
    actual_vs_predicted: pd.DataFrame | None = None
    residuals: pd.DataFrame | None = None
    error_quantiles: pd.DataFrame | None = None


def build_eda_snapshot(frame: pd.DataFrame) -> EdaSnapshot:
    """Build deterministic, display-ready EDA data without mutating the input."""
    cell_count = frame.shape[0] * frame.shape[1]
    missing_count = int(frame.isna().sum().sum())
    overview = DatasetOverview(
        rows=len(frame),
        columns=frame.shape[1],
        memory_bytes=int(frame.memory_usage(index=True, deep=True).sum()),
        duplicate_rows=int(frame.duplicated().sum()),
        missing_cell_percent=_percent(missing_count, cell_count),
    )

    quality_rows: list[dict[str, object]] = []
    for column in frame.columns:
        series = frame[column]
        non_null = int(series.notna().sum())
        unique = int(series.nunique(dropna=True))
        ratio = unique / non_null if non_null else 0.0
        name = str(column)
        quality_rows.append(
            {
                "column": name,
                "dtype": str(series.dtype),
                "non_null_count": non_null,
                "missing_count": len(series) - non_null,
                "missing_percent": _percent(len(series) - non_null, len(series)),
                "unique_count": unique,
                "is_constant": unique <= 1,
                "is_suspected_id": unique > 1
                and ratio >= 0.98
                and (non_null >= 20 or name.lower() == "id" or name.lower().endswith("_id")),
                "is_high_cardinality": unique > 20 and ratio > 0.5,
            }
        )
    quality = pd.DataFrame(quality_rows, columns=COLUMN_QUALITY_COLUMNS)

    numeric_columns = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if not pd.api.types.is_bool_dtype(frame[column])
    ]
    numeric_rows: list[dict[str, object]] = []
    for column in numeric_columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        numeric_rows.append(
            {
                "column": str(column),
                "count": int(series.count()),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
            }
        )
    numeric_summary = pd.DataFrame(
        numeric_rows, columns=["column", "count", "mean", "median", "std", "min", "max"]
    )

    categorical_columns = [column for column in frame.columns if column not in numeric_columns]
    categorical_rows: list[dict[str, object]] = []
    for column in categorical_columns:
        series = frame[column].dropna()
        counts = series.value_counts()
        categorical_rows.append(
            {
                "column": str(column),
                "count": int(series.count()),
                "unique_count": int(series.nunique()),
                "top": counts.index[0] if not counts.empty else None,
                "top_frequency": int(counts.iloc[0]) if not counts.empty else 0,
            }
        )
    categorical_summary = pd.DataFrame(
        categorical_rows,
        columns=["column", "count", "unique_count", "top", "top_frequency"],
    )

    correlation_columns = numeric_columns[:20]
    correlations = (
        frame.loc[:, correlation_columns].corr() if correlation_columns else pd.DataFrame()
    )
    return EdaSnapshot(
        overview=overview,
        column_quality=quality,
        numeric_summary=numeric_summary,
        categorical_summary=categorical_summary,
        sample=_sample(frame, 10_000),
        scatter_sample=_sample(frame, 5_000),
        correlations=correlations,
    )


def evaluate_predictions(
    task: TaskType,
    actual: pd.Series,
    predicted: pd.Series,
    probabilities: pd.DataFrame | None = None,
) -> EvaluationDiagnostics:
    """Compute comprehensive diagnostics, reporting undefined metrics as warnings."""
    aligned = pd.concat(
        [actual.rename("actual"), predicted.rename("predicted")], axis=1, join="inner"
    ).dropna()
    warnings: list[str] = []
    if aligned.empty:
        return EvaluationDiagnostics({}, ("No comparable non-null rows are available.",))
    if task == "classification":
        return _classification_diagnostics(aligned, probabilities, warnings)
    return _regression_diagnostics(aligned, warnings)


def _classification_diagnostics(
    values: pd.DataFrame, probabilities: pd.DataFrame | None, warnings: list[str]
) -> EvaluationDiagnostics:
    actual = values["actual"]
    predicted = values["predicted"]
    confusion_labels = list(pd.unique(pd.concat([actual, predicted], ignore_index=True)))
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=confusion_labels, zero_division=0
    )
    with catch_warnings():
        simplefilter("ignore", UserWarning)
        balanced_accuracy = balanced_accuracy_score(actual, predicted)
    metrics: dict[str, MetricValue] = {
        "evaluated_rows": len(values),
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
    }
    if actual.nunique() >= 2:
        metrics["mcc"] = float(matthews_corrcoef(actual, predicted))
    else:
        warnings.append("MCC is undefined when fewer than two classes are observed.")
    majority_accuracy = float(actual.value_counts(normalize=True).max())
    metrics["majority_baseline_accuracy"] = majority_accuracy
    metrics["majority_baseline_lift"] = float(metrics["accuracy"]) - majority_accuracy
    if len(confusion_labels) == 1:
        matrix = pd.crosstab(actual, predicted).reindex(
            index=confusion_labels, columns=confusion_labels, fill_value=0
        )
    else:
        matrix = pd.DataFrame(
            confusion_matrix(actual, predicted, labels=confusion_labels),
            index=confusion_labels,
            columns=confusion_labels,
        )
    matrix.index.name = "actual"
    matrix.columns.name = "predicted"
    per_class = pd.DataFrame(
        {
            "class": confusion_labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support.astype(int),
        }
    )
    probability_diagnostics: pd.DataFrame | None = None
    roc_data: pd.DataFrame | None = None
    if probabilities is not None:
        probabilities = probabilities.reindex(values.index)
        probability_diagnostics = probabilities.copy()
        probability_diagnostics.columns = [
            f"probability::{label}" for label in probability_diagnostics.columns
        ]
        probability_diagnostics.insert(0, "observed_target", actual)
        probability_diagnostics.insert(1, "predicted_target", predicted)
        canonical_labels = sorted(
            probabilities.columns,
            key=lambda label: (
                type(label).__module__,
                type(label).__qualname__,
                repr(label),
            ),
        )
        metric_probabilities = probabilities.loc[:, canonical_labels]
        label_to_index = {label: index for index, label in enumerate(canonical_labels)}
        encoded_actual = actual.map(label_to_index)
        encoded_labels = list(range(len(canonical_labels)))
        try:
            metrics["log_loss"] = float(
                log_loss(
                    encoded_actual,
                    metric_probabilities,
                    labels=encoded_labels,
                )
            )
        except ValueError as exc:
            warnings.append(f"Log loss is undefined: {exc}")
        try:
            if len(canonical_labels) == 2 and actual.nunique() == 2:
                positive = canonical_labels[1]
                binary_actual = (actual == positive).astype(int)
                scores = metric_probabilities[positive]
                metrics["roc_auc"] = float(roc_auc_score(binary_actual, scores))
                false_positive, true_positive, thresholds = roc_curve(binary_actual, scores)
                roc_data = pd.DataFrame(
                    {"false_positive_rate": false_positive, "true_positive_rate": true_positive,
                     "threshold": thresholds}
                )
            elif actual.nunique() > 2:
                metrics["roc_auc"] = float(
                    roc_auc_score(
                        encoded_actual,
                        metric_probabilities,
                        labels=encoded_labels,
                        multi_class="ovr",
                        average="macro",
                    )
                )
            else:
                warnings.append(
                    "ROC-AUC is undefined when the actual target has fewer than two classes."
                )
        except ValueError as exc:
            warnings.append(f"ROC-AUC is undefined: {exc}")
    else:
        warnings.append(
            "Probability metrics are unavailable because probabilities were not provided."
        )
    return EvaluationDiagnostics(
        metrics, tuple(warnings), matrix, per_class, roc_data, probability_diagnostics
    )


def _regression_diagnostics(
    values: pd.DataFrame, warnings: list[str]
) -> EvaluationDiagnostics:
    actual = pd.to_numeric(values["actual"], errors="coerce")
    predicted = pd.to_numeric(values["predicted"], errors="coerce")
    valid = actual.notna() & predicted.notna() & np.isfinite(actual) & np.isfinite(predicted)
    actual = actual.loc[valid]
    predicted = predicted.loc[valid]
    if actual.empty:
        return EvaluationDiagnostics({}, ("No finite numeric rows are available.",))
    errors = actual - predicted
    absolute_errors = errors.abs()
    metrics: dict[str, MetricValue] = {
        "evaluated_rows": len(actual),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "median_absolute_error": float(median_absolute_error(actual, predicted)),
    }
    baseline = pd.Series(float(actual.mean()), index=actual.index)
    metrics["mean_baseline_mae"] = float(mean_absolute_error(actual, baseline))
    metrics["mean_baseline_rmse"] = float(np.sqrt(mean_squared_error(actual, baseline)))
    metrics["mean_baseline_mae_lift"] = float(metrics["mean_baseline_mae"]) - float(metrics["mae"])
    if len(actual) < 2:
        warnings.append("R2 is undefined for fewer than two evaluated rows.")
        warnings.append("Explained variance is undefined for fewer than two evaluated rows.")
    elif actual.nunique() < 2:
        warnings.append("R2 is undefined for a constant actual target.")
        warnings.append("Explained variance is undefined for a constant actual target.")
    else:
        metrics["r2"] = float(r2_score(actual, predicted))
        metrics["explained_variance"] = float(explained_variance_score(actual, predicted))
    comparison = pd.DataFrame({"actual": actual, "predicted": predicted})
    residuals = comparison.assign(residual=errors, absolute_error=absolute_errors)
    quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]
    error_quantiles = pd.DataFrame(
        {
            "quantile": quantiles,
            "absolute_error": [float(absolute_errors.quantile(q)) for q in quantiles],
        }
    )
    return EvaluationDiagnostics(
        metrics=metrics,
        warnings=tuple(warnings),
        actual_vs_predicted=comparison,
        residuals=residuals,
        error_quantiles=error_quantiles,
    )


def _percent(numerator: int, denominator: int) -> float:
    return round(numerator / denominator * 100, 2) if denominator else 0.0


def _sample(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    return frame.copy() if len(frame) <= limit else frame.sample(n=limit, random_state=42)
