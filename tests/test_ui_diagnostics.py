from types import SimpleNamespace

import pandas as pd

from tabfm_workbench.analytics import EvaluationDiagnostics
from tabfm_workbench.ui import (
    build_actual_vs_predicted_chart,
    build_residual_chart,
    build_roc_curve_chart,
    collect_result_warnings,
)


def test_evaluation_charts_use_explicit_diagnostic_fields() -> None:
    roc = build_roc_curve_chart(
        pd.DataFrame(
            {"false_positive_rate": [0.0], "true_positive_rate": [1.0], "threshold": [0.5]}
        )
    ).to_dict()
    actual = build_actual_vs_predicted_chart(
        pd.DataFrame({"actual": [1.0], "predicted": [1.1]})
    ).to_dict()
    residual = build_residual_chart(
        pd.DataFrame({"predicted": [1.1], "residual": [-0.1], "absolute_error": [0.1]})
    ).to_dict()

    assert roc["encoding"]["x"]["field"] == "false_positive_rate"
    assert roc["encoding"]["y"]["field"] == "true_positive_rate"
    assert actual["encoding"]["x"]["field"] == "actual"
    assert actual["encoding"]["y"]["field"] == "predicted"
    assert residual["encoding"]["x"]["field"] == "predicted"
    assert residual["encoding"]["y"]["field"] == "residual"


def test_result_warnings_include_undefined_metrics_without_duplicates() -> None:
    result = SimpleNamespace(
        warnings=("alignment warning", "R2 is undefined"),
        diagnostics=EvaluationDiagnostics(
            {}, ("R2 is undefined", "Explained variance is undefined")
        ),
    )

    assert collect_result_warnings(result) == (
        "alignment warning",
        "R2 is undefined",
        "Explained variance is undefined",
    )
