import numpy as np
import pandas as pd
import pytest

from tabfm_workbench.predictor import (
    InferenceError,
    PreparedPredictor,
    align_features,
    suggest_task,
)


class FakeEstimator:
    classes_ = np.array(["no", "yes"])

    def __init__(self) -> None:
        self.fit_args: tuple[pd.DataFrame, pd.Series] | None = None
        self.fit_calls = 0

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "FakeEstimator":
        self.fit_args = (features, target)
        self.fit_calls += 1
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.array(["yes"] * len(features))

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return np.array([[0.2, 0.8]] * len(features))


class NumericFakeEstimator(FakeEstimator):
    classes_ = np.array([0, 1])

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.array([1] * len(features))


def test_classification_returns_labels_and_named_probabilities() -> None:
    estimator = FakeEstimator()
    session = PreparedPredictor("classification", estimator, device="cpu")
    context_x = pd.DataFrame({"x": [1, 2]})
    context_y = pd.Series(["no", "yes"])
    session.prepare(context_x, context_y)
    result = session.predict(pd.DataFrame({"x": [3]}))
    assert result.predictions.tolist() == ["yes"]
    assert result.probabilities is not None
    assert result.probabilities.to_dict(orient="records") == [{"no": 0.2, "yes": 0.8}]


def test_regression_rejects_non_numeric_target() -> None:
    session = PreparedPredictor("regression", FakeEstimator(), device="cpu")
    with pytest.raises(InferenceError, match="numeric"):
        session.prepare(pd.DataFrame({"x": [1, 2]}), pd.Series(["not-a-number", "2"]))


def test_session_rejects_feature_schema_mismatch() -> None:
    aligned = align_features(pd.DataFrame({"different": [2]}), ["x"])
    assert aligned.frame.columns.tolist() == ["x"]
    assert aligned.frame["x"].isna().all()
    assert aligned.missing_columns == ("x",)
    assert aligned.extra_columns == ("different",)


def test_task_suggestion_handles_integer_labels_and_continuous_targets() -> None:
    classification = suggest_task(pd.Series([0, 1, 0, 1]))
    regression = suggest_task(pd.Series([1.2, 2.8, 4.1, 5.9]))
    assert classification.task == "classification"
    assert regression.task == "regression"


def test_prepared_predictor_fits_context_only_once() -> None:
    estimator = FakeEstimator()
    session = PreparedPredictor("classification", estimator, device="cpu")
    session.prepare(pd.DataFrame({"x": [1, 2]}), pd.Series(["no", "yes"]))
    session.predict(pd.DataFrame({"x": [3]}))
    session.predict(pd.DataFrame({"x": [4]}))
    assert estimator.fit_calls == 1


def test_classification_metrics_use_only_labeled_rows() -> None:
    session = PreparedPredictor("classification", FakeEstimator(), device="cpu")
    session.prepare(pd.DataFrame({"x": [1, 2]}), pd.Series(["no", "yes"]))
    result = session.predict(
        pd.DataFrame({"x": [3, 4]}),
        expected=pd.Series(["yes", None]),
    )
    assert result.metrics["evaluated_rows"] == 1
    assert result.metrics["accuracy"] == 1.0


def test_classification_metrics_support_numeric_class_labels() -> None:
    estimator = NumericFakeEstimator()
    session = PreparedPredictor("classification", estimator, device="cpu")
    session.prepare(pd.DataFrame({"x": [1, 2]}), pd.Series([0, 1]))
    result = session.predict(pd.DataFrame({"x": [3]}), expected=pd.Series([1]))
    assert "log_loss" in result.metrics
