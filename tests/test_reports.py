from __future__ import annotations

import io
import json
import shutil
import sqlite3
import uuid
import zipfile
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from pypdf import PdfReader

from tabfm_workbench.analytics import build_eda_snapshot, evaluate_predictions
from tabfm_workbench.reports import (
    HistoryRepository,
    ReportInput,
    RunRecord,
    generate_report_bundle,
)


@pytest.fixture
def history_root() -> Path:
    base = (Path.cwd() / ".test-temp" / "history").resolve()
    root = (base / str(uuid.uuid4())).resolve()
    if not root.is_relative_to(base):
        raise RuntimeError("Test history path escaped its workspace-local base.")
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        if root.is_relative_to(base) and root != base:
            shutil.rmtree(root, ignore_errors=True)


def _report_input(rows: int = 12) -> ReportInput:
    frame = pd.DataFrame(
        {
            "feature": range(rows),
            "group": ["a", "b"] * (rows // 2) + (["a"] if rows % 2 else []),
            "target": [float(value) for value in range(rows)],
        }
    )
    predictions = frame.assign(prediction=frame["target"] + 0.5)
    return ReportInput(
        dataset_name="sample.csv",
        dataset_source="upload",
        task="regression",
        target="target",
        prediction_mode="evaluation",
        predictions=predictions,
        eda=build_eda_snapshot(frame),
        diagnostics=evaluate_predictions("regression", frame["target"], predictions["prediction"]),
        created_at=datetime(2026, 1, 2, 3, 4, tzinfo=UTC),
        latency_ms=1250.0,
        device="cpu",
        warnings=("Research use only.",),
    )


def test_report_bundle_is_self_contained_complete_and_parseable() -> None:
    report_input = _report_input()
    bundle = generate_report_bundle(report_input)

    with zipfile.ZipFile(io.BytesIO(bundle.data)) as archive:
        assert set(archive.namelist()) == {
            "report.html",
            "report.pdf",
            "predictions.csv",
            "metrics.json",
        }
        html = archive.read("report.html").decode()
        assert "sample.csv" in html
        assert "Dataset overview" in html
        assert "https://" not in html and "http://" not in html
        assert "<script src=" not in html and "<link rel=" not in html
        assert "data:image/png;base64," in html
        assert len(pd.read_csv(io.BytesIO(archive.read("predictions.csv")))) == 12
        metrics = json.loads(archive.read("metrics.json"))
        assert metrics["metrics"]["mae"] == 0.5
        assert "Research use only." in metrics["warnings"]
        assert {"actual_vs_predicted", "residuals", "error_quantiles"}.issubset(
            metrics["diagnostics"]
        )
        assert "Residual diagnostics preview" in html
        pdf = PdfReader(io.BytesIO(archive.read("report.pdf")))
        assert len(pdf.pages) >= 2
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            assert "TabFM Workbench Report" in text
            assert "TabFM Workbench - local research use" in text
            assert f"Page {page_number}" in text
        assert "Actual vs predicted preview" in "\n".join(
            page.extract_text() or "" for page in pdf.pages
        )

    assert generate_report_bundle(report_input).data == bundle.data


def test_classification_report_serializes_bounded_diagnostics() -> None:
    rows = 120
    frame = pd.DataFrame({"feature": range(rows), "target": ["A", "B"] * 60})
    predicted = pd.Series(["A", "B", "A"] * 40)
    probabilities = pd.DataFrame(
        {"A": [float("nan"), *([0.8, 0.2] * 60)[: rows - 1]],
         "B": [0.2, 0.8] * 60}
    )
    diagnostics = evaluate_predictions(
        "classification", frame["target"], predicted, probabilities
    )
    report = replace(
        _report_input(rows),
        task="classification",
        diagnostics=diagnostics,
        predictions=frame.assign(prediction=predicted),
    )

    with zipfile.ZipFile(io.BytesIO(generate_report_bundle(report).data)) as archive:
        payload = json.loads(archive.read("metrics.json"))
        diagnostics_json = payload["diagnostics"]
        assert {"confusion_matrix", "per_class", "probability_diagnostics"}.issubset(
            diagnostics_json
        )
        probability_preview = diagnostics_json["probability_diagnostics"]
        assert probability_preview["total_rows"] == 120
        assert len(probability_preview["rows"]) == 100
        assert probability_preview["truncated"] is True
        assert probability_preview["rows"][0][3] is None
        html_report = archive.read("report.html").decode()
        assert "Confusion matrix" in html_report
        assert "Probability diagnostics preview" in html_report
        pdf = PdfReader(io.BytesIO(archive.read("report.pdf")))
        pdf_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        assert "Per-class metrics" in pdf_text
        assert "Probability diagnostics preview" in pdf_text


def test_high_cardinality_diagnostics_bound_rows_and_columns_deterministically() -> None:
    labels = [f"class-{index}" for index in range(120)]
    frame = pd.DataFrame({"feature": range(120), "target": labels})
    predicted = pd.Series(labels[1:] + ["predicted-only"])
    probability_columns = {label: [1 / 120] * 120 for label in labels}
    diagnostics = evaluate_predictions(
        "classification",
        frame["target"],
        predicted,
        pd.DataFrame(probability_columns),
    )
    report = replace(
        _report_input(120),
        task="classification",
        diagnostics=diagnostics,
        predictions=frame.assign(prediction=predicted),
    )

    first = generate_report_bundle(report).data
    second = generate_report_bundle(report).data

    assert first == second
    with zipfile.ZipFile(io.BytesIO(first)) as archive:
        payload = json.loads(archive.read("metrics.json"))["diagnostics"]
        confusion = payload["confusion_matrix"]
        assert len(confusion["rows"]) == 50
        assert len(confusion["columns"]) == 51  # index label plus 50 class columns
        assert confusion["total_rows"] == 121
        assert confusion["total_columns"] == 121
        assert confusion["rows_truncated"] is True
        assert confusion["columns_truncated"] is True
        per_class = payload["per_class"]
        assert len(per_class["rows"]) == 100
        assert per_class["rows_truncated"] is True
        assert PdfReader(io.BytesIO(archive.read("report.pdf"))).pages


def _record(index: int, *, status: str = "available") -> RunRecord:
    identifier = f"00000000-0000-0000-0000-{index:012d}"
    return RunRecord(
        id=identifier,
        created_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=index),
        dataset_name=f"data-{index}.csv",
        dataset_source="upload",
        task="regression",
        target=None,
        prediction_mode="predict",
        row_count=index,
        metrics={"mae": float(index)},
        warnings=(),
        latency_ms=100.0,
        device="cpu",
        bundle_relative_path=(f"bundles/{identifier}.zip" if status == "available" else None),
        status=status,
        error=("generation failed" if status == "failed" else None),
    )


def _zip_bytes(value: str = "bundle") -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("value.txt", value)
    return output.getvalue()


def test_history_persists_reopens_and_paginates_newest_first(history_root: Path) -> None:
    repository = HistoryRepository(history_root)
    for index in range(12):
        repository.create(_record(index), bundle_data=_zip_bytes(f"bundle-{index}"))

    reopened = HistoryRepository(history_root)
    assert [item.row_count for item in reopened.list(page=1)] == list(range(11, 1, -1))
    assert [item.row_count for item in reopened.list(page=2)] == [1, 0]
    expected = replace(
        _record(4), bundle_relative_path=f"bundles/{_record(4).id}.zip"
    )
    assert reopened.get(_record(4).id) == expected
    round_trip = reopened.get(_record(4).id)
    assert round_trip is not None
    assert round_trip.latency_ms == 100.0
    assert round_trip.metrics == {"mae": 4.0}
    assert round_trip.created_at == _record(4).created_at


def test_history_handles_failed_and_missing_bundles(history_root: Path) -> None:
    repository = HistoryRepository(history_root)
    repository.create(_record(1, status="failed"))
    available = repository.create(_record(2), bundle_data=_zip_bytes())
    assert available.bundle_path is not None
    available.bundle_path.unlink()

    statuses = {item.id: item.status for item in repository.list()}
    assert statuses[_record(1).id] == "failed"
    assert statuses[_record(2).id] == "unavailable"
    missing = repository.get(_record(2).id)
    assert missing is not None
    assert missing.status == "unavailable"

    corrupt = repository.create(_record(3), bundle_data=_zip_bytes())
    assert corrupt.bundle_path is not None
    corrupt.bundle_path.write_bytes(b"not a zip")
    statuses = {item.id: item.status for item in repository.list()}
    assert statuses[_record(3).id] == "unavailable"


def test_history_rejects_failed_record_with_bundle_path(history_root: Path) -> None:
    repository = HistoryRepository(history_root)
    invalid = replace(_record(1, status="failed"), bundle_relative_path="bundles/failed.zip")

    with pytest.raises(ValueError, match="cannot reference a bundle"):
        repository.create(invalid)


def test_duplicate_create_preserves_original_bundle(history_root: Path) -> None:
    repository = HistoryRepository(history_root)
    original = repository.create(_record(1), bundle_data=_zip_bytes("original"))

    with pytest.raises(sqlite3.IntegrityError):
        repository.create(_record(1), bundle_data=_zip_bytes("replacement"))

    assert original.bundle_path is not None
    with zipfile.ZipFile(original.bundle_path) as archive:
        assert archive.read("value.txt") == b"original"


def test_create_never_deletes_preexisting_orphan_bundle(history_root: Path) -> None:
    repository = HistoryRepository(history_root)
    orphan = repository.bundle_root / f"{_record(1).id}.zip"
    orphan.write_bytes(_zip_bytes("orphan"))

    with pytest.raises(FileExistsError):
        repository.create(_record(1), bundle_data=_zip_bytes("new"))

    with zipfile.ZipFile(orphan) as archive:
        assert archive.read("value.txt") == b"orphan"
    assert repository.list() == []


def test_history_clear_deletes_only_indexed_internal_bundles(history_root: Path) -> None:
    repository = HistoryRepository(history_root)
    record = repository.create(_record(1), bundle_data=_zip_bytes())
    unrelated = history_root / "bundles" / "unrelated.zip"
    unrelated.write_bytes(b"keep")

    repository.clear()

    assert repository.list() == []
    assert record.bundle_path is not None and not record.bundle_path.exists()
    assert unrelated.read_bytes() == b"keep"


def test_history_clear_never_deletes_contained_non_bundle_path(history_root: Path) -> None:
    repository = HistoryRepository(history_root)
    protected = history_root / "protected.txt"
    protected.write_text("keep")
    with sqlite3.connect(repository.database_path) as connection:
        connection.execute(
            """INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                _record(1).id,
                _record(1).created_at.isoformat(),
                "data.csv",
                "upload",
                "regression",
                None,
                "predict",
                1,
                "{}",
                "[]",
                100.0,
                "cpu",
                "protected.txt",
                "available",
                None,
            ),
        )

    repository.clear()

    assert protected.read_text() == "keep"
    assert repository.list() == []


def test_history_clear_database_failure_preserves_rows_and_files(
    history_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = HistoryRepository(history_root)
    created = repository.create(_record(1), bundle_data=_zip_bytes())

    def fail_delete(connection: sqlite3.Connection) -> None:
        raise sqlite3.OperationalError("delete failed")

    monkeypatch.setattr(repository, "_delete_all", fail_delete)
    with pytest.raises(sqlite3.OperationalError, match="delete failed"):
        repository.clear()

    assert repository.get(_record(1).id) is not None
    assert created.bundle_path is not None and created.bundle_path.exists()


def test_history_clear_continues_after_bundle_unlink_failure(
    history_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = HistoryRepository(history_root)
    first = repository.create(_record(1), bundle_data=_zip_bytes("first"))
    second = repository.create(_record(2), bundle_data=_zip_bytes("second"))
    assert first.bundle_path is not None and second.bundle_path is not None
    original_unlink = Path.unlink

    def fail_first(path: Path, missing_ok: bool = False) -> None:
        if path == first.bundle_path:
            raise PermissionError("bundle is locked")
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_first)

    warnings = repository.clear()

    assert repository.list() == []
    assert first.bundle_path.exists()
    assert not second.bundle_path.exists()
    assert len(warnings) == 1
    assert first.bundle_path.name in warnings[0]
    assert "bundle is locked" in warnings[0]


def test_history_rejects_unsafe_bundle_paths(history_root: Path) -> None:
    repository = HistoryRepository(history_root)
    unsafe = replace(_record(1), bundle_relative_path="../escape.zip")

    with pytest.raises(ValueError, match="inside history root"):
        repository.create(unsafe, bundle_data=_zip_bytes())

    for invalid_path in ("history.sqlite3", "bundles/not-a-zip.txt"):
        invalid = replace(_record(2), bundle_relative_path=invalid_path)
        with pytest.raises(ValueError, match=r"bundles directory.*\.zip"):
            repository.create(invalid, bundle_data=_zip_bytes())


def test_history_removes_atomic_bundle_when_database_insert_fails(
    history_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = HistoryRepository(history_root)

    def fail_insert(connection: sqlite3.Connection, record: RunRecord) -> None:
        raise RuntimeError("database failed")

    monkeypatch.setattr(repository, "_insert", fail_insert)
    with pytest.raises(RuntimeError, match="database failed"):
        repository.create(_record(1), bundle_data=_zip_bytes())

    assert list((history_root / "bundles").glob("*.zip")) == []


def test_history_rejects_incompatible_schema_version(history_root: Path) -> None:
    initialized = HistoryRepository(history_root)
    with sqlite3.connect(initialized.database_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
    incompatible_root = history_root / "incompatible"
    incompatible_root.mkdir()
    database = incompatible_root / "history.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA user_version = 99")

    with pytest.raises(RuntimeError, match="schema version 99.*expected 1"):
        HistoryRepository(incompatible_root)


def test_history_marks_malformed_and_tampered_rows_unavailable(history_root: Path) -> None:
    repository = HistoryRepository(history_root)
    repository.create(_record(1), bundle_data=_zip_bytes())
    repository.create(_record(3), bundle_data=_zip_bytes())
    with sqlite3.connect(repository.database_path) as connection:
        connection.execute(
            "UPDATE runs SET metrics_json = ?, created_at = ? WHERE id = ?",
            ("[]", "not-a-time", _record(1).id),
        )
        connection.execute(
            """INSERT INTO runs
            (id, created_at, dataset_name, dataset_source, task, target, prediction_mode,
             row_count, metrics_json, warnings_json, latency_ms, device,
             bundle_relative_path, status, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                _record(2).id, _record(2).created_at.isoformat(), "safe", "upload",
                "regression", None, "predict", 2, "{}", "[]", 10.0, "cpu",
                f"bundles/{_record(1).id}.zip", "available", None,
            ),
        )

    rows = repository.list()

    assert len(rows) == 3
    assert sum(row.status == "unavailable" for row in rows) == 2
    assert sum(row.status == "available" for row in rows) == 1


@pytest.mark.parametrize(
    "change",
    [
        {"latency_ms": float("nan")},
        {"latency_ms": float("inf")},
        {"task": "invalid"},
        {"prediction_mode": "invalid"},
        {"metrics": {"mae": float("nan")}},
    ],
)
def test_history_validates_metadata_before_side_effects(
    history_root: Path, change: dict[str, object]
) -> None:
    repository = HistoryRepository(history_root)
    invalid = replace(_record(1), **change)

    with pytest.raises(ValueError):
        repository.create(invalid, bundle_data=_zip_bytes())

    assert repository.list() == []
    assert list(repository.bundle_root.glob("*.zip")) == []


def test_report_rejects_invalid_metadata_before_generation() -> None:
    with pytest.raises(ValueError, match="latency_ms"):
        generate_report_bundle(replace(_report_input(), latency_ms=float("nan")))


def test_pdf_handles_long_adversarial_cells() -> None:
    report = _report_input()
    long_value = "<unsafe>&" + "wrapped-value-" * 80
    predictions = report.predictions.copy()
    predictions["adversarial"] = long_value
    report = replace(
        report, dataset_name=long_value, predictions=predictions, warnings=(long_value,)
    )

    with zipfile.ZipFile(io.BytesIO(generate_report_bundle(report).data)) as archive:
        pdf = PdfReader(io.BytesIO(archive.read("report.pdf")))
        assert len(pdf.pages) >= 2
        assert all(page.extract_text() is not None for page in pdf.pages)
