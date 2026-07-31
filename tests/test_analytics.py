import numpy as np
import pandas as pd
import pytest

from tabfm_workbench.analytics import build_eda_snapshot, evaluate_predictions


def test_eda_snapshot_reports_quality_and_descriptive_tables() -> None:
    frame = pd.DataFrame(
        {
            "customer_id": range(1, 101),
            "amount": [float(value) if value != 50 else np.nan for value in range(100)],
            "segment": ["a", "b"] * 50,
            "constant": [1] * 100,
        }
    )
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    snapshot = build_eda_snapshot(frame)

    assert snapshot.overview.rows == 101
    assert snapshot.overview.columns == 4
    assert snapshot.overview.duplicate_rows == 1
    assert snapshot.overview.missing_cell_percent == 0.25
    quality = snapshot.column_quality.set_index("column")
    assert quality.loc["amount", "missing_count"] == 1
    assert quality.loc["amount", "missing_percent"] == 0.99
    assert bool(quality.loc["constant", "is_constant"])
    assert bool(quality.loc["customer_id", "is_suspected_id"])
    assert {"column", "mean", "median", "std", "min", "max"}.issubset(
        snapshot.numeric_summary.columns
    )
    assert snapshot.categorical_summary.loc[0, "column"] == "segment"


def test_eda_sampling_and_correlations_are_deterministic_and_capped() -> None:
    frame = pd.DataFrame(
        np.arange(12_050 * 22, dtype=float).reshape(12_050, 22),
        columns=[f"n{index}" for index in range(22)],
    )

    first = build_eda_snapshot(frame)
    second = build_eda_snapshot(frame)

    assert len(first.sample) == 10_000
    assert len(first.scatter_sample) == 5_000
    assert first.sample.index.tolist() == second.sample.index.tolist()
    assert first.scatter_sample.index.tolist() == second.scatter_sample.index.tolist()
    assert first.correlations.shape == (20, 20)
    assert first.correlations.columns.tolist() == [f"n{index}" for index in range(20)]


def test_zero_column_eda_has_stable_quality_schema() -> None:
    snapshot = build_eda_snapshot(pd.DataFrame(index=range(3)))

    assert snapshot.overview.rows == 3
    assert snapshot.overview.columns == 0
    assert snapshot.column_quality.empty
    assert snapshot.column_quality.columns.tolist() == [
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


def test_empty_eda_with_columns_retains_summary_schemas() -> None:
    snapshot = build_eda_snapshot(
        pd.DataFrame(
            {
                "number": pd.Series(dtype="float64"),
                "category": pd.Series(dtype="object"),
            }
        )
    )

    assert snapshot.overview.rows == 0
    assert snapshot.numeric_summary.columns.tolist() == [
        "column", "count", "mean", "median", "std", "min", "max"
    ]
    assert snapshot.categorical_summary.columns.tolist() == [
        "column", "count", "unique_count", "top", "top_frequency"
    ]


def test_classification_diagnostics_include_metrics_tables_and_probability_data() -> None:
    actual = pd.Series(["no", "no", "yes", "yes"])
    predicted = pd.Series(["no", "yes", "yes", "yes"])
    probabilities = pd.DataFrame(
        {"no": [0.9, 0.4, 0.2, 0.1], "yes": [0.1, 0.6, 0.8, 0.9]}
    )

    diagnostics = evaluate_predictions("classification", actual, predicted, probabilities)

    assert diagnostics.metrics["accuracy"] == 0.75
    assert diagnostics.metrics["balanced_accuracy"] == 0.75
    assert diagnostics.metrics["macro_f1"] == 0.7333333333333334
    assert diagnostics.metrics["majority_baseline_accuracy"] == 0.5
    assert diagnostics.metrics["majority_baseline_lift"] == 0.25
    assert diagnostics.confusion_matrix is not None
    assert diagnostics.confusion_matrix.loc["no", "yes"] == 1
    assert diagnostics.per_class is not None
    assert set(diagnostics.per_class["class"]) == {"no", "yes"}
    assert diagnostics.probability_diagnostics is not None
    assert diagnostics.roc_curve is not None
    assert not diagnostics.warnings


def test_probability_diagnostics_preserve_colliding_class_names_and_align_rows() -> None:
    diagnostics = evaluate_predictions(
        "classification",
        pd.Series(["actual", None, "predicted"], index=[10, 11, 12]),
        pd.Series(["actual", "actual", "predicted"], index=[10, 11, 12]),
        pd.DataFrame(
            {
                "actual": [0.9, 0.7, 0.1, 0.5],
                "predicted": [0.1, 0.3, 0.9, 0.5],
            },
            index=[10, 11, 12, 99],
        ),
    )

    assert diagnostics.probability_diagnostics is not None
    assert diagnostics.probability_diagnostics.index.tolist() == [10, 12]
    assert diagnostics.probability_diagnostics.columns.tolist() == [
        "observed_target",
        "predicted_target",
        "probability::actual",
        "probability::predicted",
    ]
    assert diagnostics.probability_diagnostics.loc[12, "probability::predicted"] == 0.9


def test_probability_diagnostic_columns_are_structurally_namespaced() -> None:
    classes = ["observed_target", "predicted_target", "probability::vip"]
    diagnostics = evaluate_predictions(
        "classification",
        pd.Series(classes),
        pd.Series(classes),
        pd.DataFrame(
            np.eye(3),
            columns=classes,
        ),
    )

    assert diagnostics.probability_diagnostics is not None
    assert diagnostics.probability_diagnostics.columns.tolist() == [
        "observed_target",
        "predicted_target",
        "probability::observed_target",
        "probability::predicted_target",
        "probability::probability::vip",
    ]
    assert diagnostics.metrics["log_loss"] == pytest.approx(0.0, abs=1e-12)
    assert diagnostics.metrics["roc_auc"] == 1.0


def test_probability_metrics_do_not_assume_incoming_column_order() -> None:
    diagnostics = evaluate_predictions(
        "classification",
        pd.Series(["z", "a", "z", "a"]),
        pd.Series(["z", "a", "z", "a"]),
        pd.DataFrame(
            {
                "z": [0.9, 0.1, 0.8, 0.2],
                "a": [0.1, 0.9, 0.2, 0.8],
            }
        ),
    )

    assert diagnostics.metrics["log_loss"] < 0.3
    assert diagnostics.metrics["roc_auc"] == 1.0


def test_classification_undefined_metrics_become_warnings() -> None:
    diagnostics = evaluate_predictions(
        "classification",
        pd.Series(["only", "only"]),
        pd.Series(["only", "only"]),
        pd.DataFrame({"only": [1.0, 1.0]}),
    )

    assert "roc_auc" not in diagnostics.metrics
    assert diagnostics.warnings


def test_classification_macro_metrics_use_actual_classes_not_predicted_only_labels() -> None:
    diagnostics = evaluate_predictions(
        "classification",
        pd.Series(["a", "a", "b", "b"]),
        pd.Series(["a", "other", "other", "other"]),
    )

    assert diagnostics.metrics["balanced_accuracy"] == 0.25
    assert diagnostics.metrics["macro_precision"] == 1 / 3
    assert diagnostics.metrics["macro_recall"] == 1 / 6
    assert diagnostics.metrics["macro_f1"] == 2 / 9
    assert diagnostics.confusion_matrix is not None
    assert "other" in diagnostics.confusion_matrix.columns
    assert diagnostics.per_class is not None
    assert diagnostics.per_class["class"].tolist() == ["a", "b", "other"]


def test_mcc_is_omitted_when_actual_has_only_one_class() -> None:
    diagnostics = evaluate_predictions(
        "classification",
        pd.Series(["only", "only"]),
        pd.Series(["only", "other"]),
    )

    assert "mcc" not in diagnostics.metrics
    assert any("MCC" in warning for warning in diagnostics.warnings)


def test_regression_diagnostics_include_residuals_quantiles_and_baseline() -> None:
    diagnostics = evaluate_predictions(
        "regression",
        pd.Series([1.0, 2.0, 3.0, 4.0]),
        pd.Series([1.0, 2.0, 4.0, 4.0]),
    )

    assert diagnostics.metrics["mae"] == 0.25
    assert diagnostics.metrics["rmse"] == 0.5
    assert diagnostics.metrics["median_absolute_error"] == 0.0
    assert diagnostics.metrics["mean_baseline_mae"] == 1.0
    assert diagnostics.actual_vs_predicted is not None
    assert diagnostics.residuals is not None
    assert diagnostics.residuals["residual"].tolist() == [0.0, 0.0, -1.0, 0.0]
    assert diagnostics.error_quantiles is not None
    assert diagnostics.error_quantiles["quantile"].tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_single_row_regression_omits_undefined_variance_metrics() -> None:
    diagnostics = evaluate_predictions(
        "regression",
        pd.Series([2.0]),
        pd.Series([2.5]),
    )

    assert "r2" not in diagnostics.metrics
    assert "explained_variance" not in diagnostics.metrics
    assert any("Explained variance" in warning for warning in diagnostics.warnings)


def test_constant_actual_regression_omits_undefined_variance_metrics() -> None:
    diagnostics = evaluate_predictions(
        "regression",
        pd.Series([2.0, 2.0, 2.0]),
        pd.Series([1.0, 2.0, 3.0]),
    )

    assert "r2" not in diagnostics.metrics
    assert "explained_variance" not in diagnostics.metrics
    assert any("constant actual target" in warning for warning in diagnostics.warnings)
