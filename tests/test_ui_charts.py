import pandas as pd

from tabfm_workbench.ui import (
    build_categorical_numeric_box_chart,
    build_numeric_histogram_chart,
)


def test_altair_builders_preserve_adversarial_column_names_as_fields() -> None:
    numeric = "amount: net.[usd]\\value"
    category = "group: [name]"
    frame = pd.DataFrame({numeric: [1.0, 2.0], category: ["a", "b"]})
    histogram = build_numeric_histogram_chart(frame, numeric).to_dict()
    box = build_categorical_numeric_box_chart(frame, category, numeric).to_dict()
    assert histogram["encoding"]["x"]["field"] == numeric
    assert box["encoding"]["x"]["field"] == category
    assert box["encoding"]["y"]["field"] == numeric
