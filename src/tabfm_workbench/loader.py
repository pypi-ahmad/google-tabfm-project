"""Tabular file ingestion and context/test partitioning."""

from collections.abc import Iterable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import pandas as pd


class DataFormatError(ValueError):
    """Raised when a table cannot be loaded safely."""


@dataclass(frozen=True)
class DatasetArtifact:
    """Successfully parsed table and its provenance."""

    name: str
    dataframe: pd.DataFrame
    source: str = "upload"
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class LoadFailure:
    """One failed input from a multi-file load."""

    name: str
    message: str


@dataclass(frozen=True)
class MultiLoadResult:
    artifacts: tuple[DatasetArtifact, ...]
    failures: tuple[LoadFailure, ...]


@dataclass(frozen=True)
class ContextAndTest:
    """Labeled in-context examples and unlabeled rows to predict."""

    context: pd.DataFrame
    test: pd.DataFrame
    context_features: pd.DataFrame
    context_target: pd.Series
    test_features: pd.DataFrame


def load_table(source: BinaryIO | BytesIO, filename: str) -> pd.DataFrame:
    """Load one supported table from a binary stream."""
    extension = Path(filename).suffix.lower()
    try:
        if extension == ".csv":
            table = pd.read_csv(source)
        elif extension == ".parquet":
            table = pd.read_parquet(source)
        elif extension == ".xlsx":
            table = pd.read_excel(source, engine="openpyxl")
        else:
            raise DataFormatError("Unsupported file type. Use CSV, Parquet, or XLSX.")
    except DataFormatError:
        raise
    except Exception as exc:
        raise DataFormatError(f"Could not parse {extension or 'file'}: {exc}") from exc
    if table.empty:
        raise DataFormatError("Dataset contains no rows.")
    if table.columns.empty:
        raise DataFormatError("Dataset contains no columns.")
    if table.columns.duplicated().any():
        raise DataFormatError("Dataset contains duplicate column names.")
    return table


def load_many(files: Iterable[tuple[str, bytes]], *, source: str = "upload") -> MultiLoadResult:
    """Parse files independently so one invalid file does not discard valid inputs."""
    artifacts: list[DatasetArtifact] = []
    failures: list[LoadFailure] = []
    for name, content in files:
        try:
            dataframe = load_table(BytesIO(content), name)
            artifacts.append(DatasetArtifact(name=name, dataframe=dataframe, source=source))
        except DataFormatError as exc:
            failures.append(LoadFailure(name=name, message=str(exc)))
    return MultiLoadResult(tuple(artifacts), tuple(failures))


def partition_rows(table: pd.DataFrame, target: str) -> ContextAndTest:
    """Treat labeled rows as context and blank-target rows as prediction cases."""
    if target not in table.columns:
        raise ValueError(f"Target column not found: {target}")
    blank = table[target].isna()
    context = table.loc[~blank].copy()
    test = table.loc[blank].copy()
    if context.empty:
        raise ValueError("At least one labeled context row is required.")
    if test.empty:
        raise ValueError("At least one row with a blank target is required.")
    return ContextAndTest(
        context=context,
        test=test,
        context_features=context.drop(columns=[target]),
        context_target=context[target],
        test_features=test.drop(columns=[target]),
    )
