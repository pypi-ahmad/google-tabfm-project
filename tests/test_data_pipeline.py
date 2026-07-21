from io import BytesIO

import pandas as pd
import pytest

from tabfm_workbench.loader import DataFormatError, load_many, load_table, partition_rows


def test_load_csv_preserves_columns() -> None:
    table = load_table(BytesIO(b"age,city,target\n31,Pune,yes\n"), "sample.csv")
    assert table.to_dict(orient="records") == [{"age": 31, "city": "Pune", "target": "yes"}]


def test_load_rejects_unsupported_extension() -> None:
    with pytest.raises(DataFormatError, match="Unsupported file type"):
        load_table(BytesIO(b"x"), "sample.json")


def test_partition_uses_blank_targets_as_test_rows() -> None:
    table = pd.DataFrame({"x": [1, 2, 3], "target": ["yes", None, "no"]})
    split = partition_rows(table, "target")
    assert split.context.index.tolist() == [0, 2]
    assert split.test.index.tolist() == [1]
    assert split.test_features.columns.tolist() == ["x"]


def test_partition_requires_context_and_test_rows() -> None:
    with pytest.raises(ValueError, match="blank target"):
        partition_rows(pd.DataFrame({"x": [1], "target": ["yes"]}), "target")


def test_load_many_keeps_valid_files_when_one_file_is_invalid() -> None:
    result = load_many(
        [
            ("valid.csv", b"x,target\n1,yes\n"),
            ("invalid.json", b"{}"),
        ]
    )
    assert [artifact.name for artifact in result.artifacts] == ["valid.csv"]
    assert [failure.name for failure in result.failures] == ["invalid.json"]
    assert "Unsupported file type" in result.failures[0].message


def test_load_table_rejects_empty_dataset() -> None:
    with pytest.raises(DataFormatError, match="no rows"):
        load_table(BytesIO(b"x,target\n"), "empty.csv")
