"""Pure report generation and durable local run history."""

from __future__ import annotations

import base64
import html
import io
import json
import math
import os
import sqlite3
import tempfile
import uuid
import zipfile
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import altair as alt
import pandas as pd
import vl_convert as vlc
from reportlab.lib import colors  # type: ignore[import-untyped]
from reportlab.lib.enums import TA_CENTER  # type: ignore[import-untyped]
from reportlab.lib.pagesizes import A4  # type: ignore[import-untyped]
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # type: ignore[import-untyped]
from reportlab.lib.units import mm  # type: ignore[import-untyped]
from reportlab.pdfgen.canvas import Canvas  # type: ignore[import-untyped]
from reportlab.platypus import (  # type: ignore[import-untyped]
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .analytics import EdaSnapshot, EvaluationDiagnostics, TaskType

PredictionMode = Literal["predict", "evaluation"]
BundleStatus = Literal["available", "failed", "unavailable"]
DIAGNOSTIC_TITLES = {
    "confusion_matrix": "Confusion matrix",
    "per_class": "Per-class metrics",
    "roc_curve": "ROC curve",
    "probability_diagnostics": "Probability diagnostics preview",
    "actual_vs_predicted": "Actual vs predicted preview",
    "residuals": "Residual diagnostics preview",
    "error_quantiles": "Error quantiles",
}
ROW_PREVIEW_LIMIT = 100


@dataclass(frozen=True)
class ReportInput:
    dataset_name: str
    dataset_source: str
    task: TaskType
    target: str | None
    prediction_mode: PredictionMode
    predictions: pd.DataFrame
    eda: EdaSnapshot
    diagnostics: EvaluationDiagnostics
    created_at: datetime
    latency_ms: float
    device: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReportBundle:
    data: bytes

    def write_atomic(self, destination: Path) -> Path:
        """Write the complete ZIP atomically and return its resolved destination."""
        destination = destination.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(self.data)
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return destination


@dataclass(frozen=True)
class RunRecord:
    id: str
    created_at: datetime
    dataset_name: str
    dataset_source: str
    task: TaskType
    target: str | None
    prediction_mode: PredictionMode
    row_count: int
    metrics: dict[str, float | int]
    warnings: tuple[str, ...]
    latency_ms: float
    device: str
    bundle_relative_path: str | None
    status: BundleStatus
    error: str | None = None
    bundle_path: Path | None = field(default=None, compare=False)

    @classmethod
    def failed(cls, report: ReportInput, error: str, *, record_id: str | None = None) -> RunRecord:
        return cls.from_report(
            report, record_id=record_id, status="failed", error=error, bundle_path=None
        )

    @classmethod
    def from_report(
        cls,
        report: ReportInput,
        *,
        record_id: str | None = None,
        status: BundleStatus = "available",
        error: str | None = None,
        bundle_path: str | None = None,
    ) -> RunRecord:
        identifier = record_id or str(uuid.uuid4())
        relative = bundle_path
        if status == "available" and relative is None:
            relative = f"bundles/{identifier}.zip"
        return cls(
            id=identifier,
            created_at=_utc(report.created_at),
            dataset_name=report.dataset_name,
            dataset_source=report.dataset_source,
            task=report.task,
            target=report.target,
            prediction_mode=report.prediction_mode,
            row_count=len(report.predictions),
            metrics=dict(report.diagnostics.metrics),
            warnings=tuple((*report.warnings, *report.diagnostics.warnings)),
            latency_ms=report.latency_ms,
            device=report.device,
            bundle_relative_path=relative,
            status=status,
            error=error,
        )


def generate_report_bundle(report: ReportInput) -> ReportBundle:
    """Generate a deterministic, self-contained report archive in memory."""
    _validate_report(report)
    chart_png = _chart_png(report)
    html_bytes = _html_report(report, chart_png).encode("utf-8")
    pdf_bytes = _pdf_report(report, chart_png)
    metrics = {
        "metrics": report.diagnostics.metrics,
        "warnings": [*report.warnings, *report.diagnostics.warnings],
        "diagnostics": serialize_diagnostics(report.diagnostics),
    }
    output = io.BytesIO()
    members = (
        ("report.html", html_bytes),
        ("report.pdf", pdf_bytes),
        ("predictions.csv", report.predictions.to_csv(index=False).encode("utf-8")),
        (
            "metrics.json",
            json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False).encode("utf-8"),
        ),
    )
    with zipfile.ZipFile(output, "w") as archive:
        for name, data in members:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    return ReportBundle(output.getvalue())


def serialize_diagnostics(
    diagnostics: EvaluationDiagnostics,
) -> dict[str, dict[str, object]]:
    """Serialize available diagnostic tables with bounded row-level previews."""
    output: dict[str, dict[str, object]] = {}
    for name in DIAGNOSTIC_TITLES:
        frame = getattr(diagnostics, name)
        if frame is None:
            continue
        row_limit = 50 if name == "confusion_matrix" else ROW_PREVIEW_LIMIT
        output[name] = _serialize_frame(frame, row_limit=row_limit, column_limit=50)
    return output


class HistoryRepository:
    """SQLite metadata index with ZIP bundles stored alongside it."""

    PAGE_SIZE = 10
    SCHEMA_VERSION = 1

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.bundle_root = (self.root / "bundles").resolve()
        if not self.bundle_root.is_relative_to(self.root):
            raise ValueError("Bundle directory must remain inside history root.")
        self.bundle_root.mkdir(parents=True, exist_ok=True)
        self.database_path = self.root / "history.sqlite3"
        self._initialize()

    def create(self, record: RunRecord, bundle_data: bytes | None = None) -> RunRecord:
        """Reserve metadata transactionally before publishing its canonical bundle."""
        _validate_record(record)
        if record.status != "available" and record.bundle_relative_path is not None:
            raise ValueError(f"A {record.status} record cannot reference a bundle.")
        if record.status == "available" and bundle_data is None:
            raise ValueError("An available record requires bundle data.")
        if record.status != "available" and bundle_data is not None:
            raise ValueError("Only available records can have bundle data.")
        if record.bundle_relative_path is not None:
            self._safe_bundle_path(record.bundle_relative_path)

        relative = f"bundles/{record.id}.zip" if bundle_data is not None else None
        if record.bundle_relative_path != relative:
            raise ValueError("Bundle path must exactly match bundles/{record.id}.zip.")
        destination = self._safe_bundle_path(relative) if relative else None
        persisted = replace(record, bundle_path=destination)
        temporary: Path | None = None
        published = False
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._insert(connection, persisted)
            if bundle_data is not None and destination is not None:
                descriptor, name = tempfile.mkstemp(
                    prefix=f".{record.id}.", suffix=".tmp", dir=self.bundle_root
                )
                temporary = Path(name)
                with os.fdopen(descriptor, "wb") as output:
                    output.write(bundle_data)
                    output.flush()
                    os.fsync(output.fileno())
                if destination.exists():
                    raise FileExistsError(f"Bundle already exists: {destination.name}")
                temporary.replace(destination)
                temporary = None
                published = True
            connection.commit()
        except BaseException:
            connection.rollback()
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            # Only a canonical file published by this uncommitted reservation is removed.
            if published and destination is not None and self.get(record.id) is None:
                destination.unlink(missing_ok=True)
            raise
        finally:
            connection.close()
        return persisted

    def list(self, page: int = 1, *, page_size: int = PAGE_SIZE) -> list[RunRecord]:
        if page < 1 or page_size < 1:
            raise ValueError("page and page_size must be positive")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM runs ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
                (page_size, (page - 1) * page_size),
            ).fetchall()
        return [self._materialize(row) for row in rows]

    def get(self, record_id: str) -> RunRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE id = ?", (record_id,)).fetchone()
        return self._materialize(row) if row is not None else None

    def clear(self) -> tuple[str, ...]:
        """Commit metadata deletion before best-effort canonical bundle cleanup."""
        paths: list[Path] = []
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            indexed = connection.execute(
                "SELECT id, bundle_relative_path FROM runs "
                "WHERE bundle_relative_path IS NOT NULL"
            ).fetchall()
            for row in indexed:
                try:
                    path = self._safe_bundle_path(str(row[1]), record_id=str(row[0]))
                except ValueError:
                    continue
                paths.append(path)
            self._delete_all(connection)
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
        warnings: list[str] = []
        for path in paths:
            if isinstance(path, Path):
                try:
                    path.unlink(missing_ok=True)
                except OSError as exc:
                    warnings.append(f"Could not delete orphan bundle {path.name}: {exc}")
        return tuple(warnings)

    def _delete_all(self, connection: sqlite3.Connection) -> None:
        connection.execute("DELETE FROM runs")

    def _initialize(self) -> None:
        with self._connect() as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            has_runs = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='runs'"
            ).fetchone()
            if version == 0 and has_runs is not None:
                raise RuntimeError(
                    "Unversioned existing history schema is incompatible; expected 1."
                )
            if version == self.SCHEMA_VERSION and has_runs is None:
                raise RuntimeError("Versioned history database is missing its runs table.")
            if version not in (0, self.SCHEMA_VERSION):
                raise RuntimeError(
                    f"History schema version {version} is incompatible; "
                    f"expected {self.SCHEMA_VERSION}."
                )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    dataset_name TEXT NOT NULL,
                    dataset_source TEXT NOT NULL,
                    task TEXT NOT NULL,
                    target TEXT,
                    prediction_mode TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    metrics_json TEXT NOT NULL,
                    warnings_json TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    device TEXT NOT NULL,
                    bundle_relative_path TEXT,
                    status TEXT NOT NULL,
                    error TEXT
                )
                """
            )
            connection.execute(f"PRAGMA user_version = {self.SCHEMA_VERSION}")
            columns = tuple(
                row[1] for row in connection.execute("PRAGMA table_info(runs)").fetchall()
            )
            expected = (
                "id", "created_at", "dataset_name", "dataset_source", "task", "target",
                "prediction_mode", "row_count", "metrics_json", "warnings_json",
                "latency_ms", "device", "bundle_relative_path", "status", "error",
            )
            if columns != expected:
                raise RuntimeError("History schema columns are incompatible with version 1.")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _insert(self, connection: sqlite3.Connection, record: RunRecord) -> None:
        connection.execute(
                """INSERT INTO runs
                (id, created_at, dataset_name, dataset_source, task, target, prediction_mode,
                 row_count, metrics_json, warnings_json, latency_ms, device,
                 bundle_relative_path, status, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.id,
                    _utc(record.created_at).isoformat(),
                    record.dataset_name,
                    record.dataset_source,
                    record.task,
                    record.target,
                    record.prediction_mode,
                    record.row_count,
                    json.dumps(record.metrics, sort_keys=True),
                    json.dumps(record.warnings),
                    record.latency_ms,
                    record.device,
                    record.bundle_relative_path,
                    record.status,
                    record.error,
                ),
            )

    def _materialize(self, row: sqlite3.Row) -> RunRecord:
        try:
            return self._materialize_valid(row)
        except (TypeError, ValueError, json.JSONDecodeError, OverflowError) as exc:
            return RunRecord(
                id=str(row["id"] or "malformed"),
                created_at=datetime(1970, 1, 1, tzinfo=UTC),
                dataset_name=str(row["dataset_name"] or "Unavailable run"),
                dataset_source=str(row["dataset_source"] or "unknown"),
                task="regression",
                target=None,
                prediction_mode="predict",
                row_count=0,
                metrics={},
                warnings=(),
                latency_ms=0.0,
                device=str(row["device"] or "unknown"),
                bundle_relative_path=None,
                status="unavailable",
                error=f"Malformed history metadata: {exc}",
            )

    def _materialize_valid(self, row: sqlite3.Row) -> RunRecord:
        relative = row["bundle_relative_path"]
        path: Path | None = None
        status: BundleStatus = row["status"]
        error = row["error"]
        if relative is not None:
            try:
                path = self._safe_bundle_path(str(relative), record_id=str(row["id"]))
                if not path.is_file() or not zipfile.is_zipfile(path):
                    status = "unavailable"
                    error = "Bundle is missing or corrupt."
                    path = None
            except ValueError:
                status = "unavailable"
                error = "Bundle path is unsafe."
        metrics = json.loads(row["metrics_json"])
        warnings = json.loads(row["warnings_json"])
        if not isinstance(metrics, dict) or not isinstance(warnings, list):
            raise ValueError("Metrics must be an object and warnings must be an array.")
        record = RunRecord(
            id=row["id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            dataset_name=row["dataset_name"],
            dataset_source=row["dataset_source"],
            task=row["task"],
            target=row["target"],
            prediction_mode=row["prediction_mode"],
            row_count=row["row_count"],
            metrics=metrics,
            warnings=tuple(warnings),
            latency_ms=row["latency_ms"],
            device=row["device"],
            bundle_relative_path=relative,
            status=status,
            error=error,
            bundle_path=path,
        )
        _validate_record(record, allow_unavailable=True)
        return record

    def _safe_bundle_path(self, relative: str, *, record_id: str | None = None) -> Path:
        candidate = (self.root / relative).resolve()
        if candidate == self.root or not candidate.is_relative_to(self.root):
            raise ValueError("Bundle path must remain inside history root.")
        if not candidate.is_relative_to(self.bundle_root) or candidate.suffix.lower() != ".zip":
            raise ValueError("Bundle path must be in the bundles directory and end with .zip.")
        if record_id is not None and candidate != (self.bundle_root / f"{record_id}.zip").resolve():
            raise ValueError("Bundle path does not match its record id.")
        return candidate


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("Timestamps must be timezone-aware.")
    return value.astimezone(UTC)


def _serialize_frame(
    frame: pd.DataFrame, *, row_limit: int, column_limit: int
) -> dict[str, object]:
    total_rows = len(frame)
    total_columns = len(frame.columns)
    preview = frame.iloc[:row_limit, :column_limit]
    index_name = str(frame.index.name) if frame.index.name is not None else "index"
    columns = [index_name, *[_json_scalar(column) for column in preview.columns]]
    rows = [
        [_json_scalar(index), *[_json_scalar(value) for value in row]]
        for index, row in zip(preview.index, preview.itertuples(index=False), strict=True)
    ]
    return {
        "columns": columns,
        "rows": rows,
        "total_rows": total_rows,
        "total_columns": total_columns,
        "rows_truncated": len(preview) < total_rows,
        "columns_truncated": len(preview.columns) < total_columns,
        "truncated": len(preview) < total_rows or len(preview.columns) < total_columns,
    }


def _json_scalar(value: object) -> object:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    return repr(value)


def _validate_report(report: ReportInput) -> None:
    if report.task not in ("classification", "regression"):
        raise ValueError("task must be classification or regression")
    if report.prediction_mode not in ("predict", "evaluation"):
        raise ValueError("prediction_mode must be predict or evaluation")
    _validate_numbers(report.latency_ms, report.diagnostics.metrics)
    _utc(report.created_at)


def _validate_record(record: RunRecord, *, allow_unavailable: bool = False) -> None:
    try:
        uuid.UUID(record.id)
    except (ValueError, AttributeError) as exc:
        raise ValueError("id must be a UUID") from exc
    if record.task not in ("classification", "regression"):
        raise ValueError("task must be classification or regression")
    if record.prediction_mode not in ("predict", "evaluation"):
        raise ValueError("prediction_mode must be predict or evaluation")
    if record.status not in ("available", "failed", "unavailable"):
        raise ValueError("status is invalid")
    if (
        record.status != "available"
        and record.bundle_relative_path is not None
        and not (allow_unavailable and record.status == "unavailable")
    ):
        raise ValueError(f"A {record.status} record cannot reference a bundle.")
    if record.row_count < 0:
        raise ValueError("row_count cannot be negative")
    if not all(isinstance(item, str) for item in record.warnings):
        raise ValueError("warnings must contain strings")
    _validate_numbers(record.latency_ms, record.metrics)
    _utc(record.created_at)


def _validate_numbers(latency_ms: float, metrics: dict[str, float | int]) -> None:
    if not isinstance(latency_ms, (float, int)) or not math.isfinite(float(latency_ms)):
        raise ValueError("latency_ms must be finite")
    for name, value in metrics.items():
        if (
            not isinstance(name, str)
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise ValueError("metrics must map strings to numeric values")
        if not math.isfinite(float(value)):
            raise ValueError(f"metric {name} must be finite")


def _chart_png(report: ReportInput) -> bytes | None:
    numeric = report.eda.numeric_summary
    if numeric.empty:
        return None
    chart_data = numeric.loc[:, ["column", "mean"]].head(12).copy()
    chart_data["mean"] = pd.to_numeric(chart_data["mean"], errors="coerce")
    chart = (
        alt.Chart(chart_data)
        .mark_bar(color="#2563eb")
        .encode(
            x=alt.X("column:N", sort=None, title="Column"),
            y=alt.Y("mean:Q", title="Mean"),
        )
        .properties(width=620, height=260, title="Numeric feature means")
    )
    return vlc.vegalite_to_png(chart.to_json(), scale=1.5)


def _html_report(report: ReportInput, chart_png: bytes | None) -> str:
    overview = report.eda.overview
    warnings = [*report.warnings, *report.diagnostics.warnings]
    image = ""
    if chart_png is not None:
        encoded = base64.b64encode(chart_png).decode("ascii")
        image = f'<img alt="Numeric feature means" src="data:image/png;base64,{encoded}">'
    metrics_rows = "".join(
        f"<tr><th>{html.escape(name)}</th><td>{html.escape(str(value))}</td></tr>"
        for name, value in sorted(report.diagnostics.metrics.items())
    )
    warning_items = "".join(f"<li>{html.escape(item)}</li>" for item in warnings) or "<li>None</li>"
    quality = report.eda.column_quality.head(100).to_html(index=False, escape=True)
    preview = report.predictions.head(20).to_html(index=False, escape=True)
    dataset_name = html.escape(report.dataset_name)
    generated_at = html.escape(_utc(report.created_at).isoformat())
    source = html.escape(report.dataset_source)
    target = html.escape(report.target or "None")
    device = html.escape(report.device)
    task = html.escape(report.task)
    mode = html.escape(report.prediction_mode)
    diagnostic_sections = "".join(
        _diagnostic_html(name, table)
        for name, table in serialize_diagnostics(report.diagnostics).items()
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>TabFM Workbench Report</title><style>
body{{font:14px system-ui,sans-serif;color:#172033;max-width:1100px;
margin:40px auto;padding:0 24px}}
h1,h2{{color:#123b73}} .meta{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}
.card{{background:#f4f7fb;padding:14px;border-radius:8px}}
table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #d8e0ea;padding:6px;text-align:left}} img{{max-width:100%;height:auto}}
</style></head><body><h1>TabFM Workbench Report</h1>
<p>{dataset_name} - generated {generated_at}</p>
<h2>Dataset overview</h2><div class="meta"><div class="card">Rows: {overview.rows}</div>
<div class="card">Columns: {overview.columns}</div>
<div class="card">Missing cells: {overview.missing_cell_percent}%</div></div>
<h2>Run provenance</h2><p>Source: {source} | Task: {task} |
Target: {target} | Mode: {mode} | Device: {device} |
Latency: {report.latency_ms:.1f}ms</p><h2>Schema and quality</h2>{quality}{image}
<h2>Metrics and diagnostics</h2><table>{metrics_rows}</table>
{diagnostic_sections}
<h2>Warnings</h2><ul>{warning_items}</ul>
<h2>Prediction preview</h2>{preview}</body></html>"""


def _diagnostic_html(name: str, table: dict[str, object]) -> str:
    columns = table["columns"]
    rows = table["rows"]
    assert isinstance(columns, list) and isinstance(rows, list)
    header = "".join(f"<th>{html.escape(str(value))}</th>" for value in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    notes: list[str] = []
    if table["rows_truncated"]:
        notes.append(f"{len(rows)} of {table['total_rows']} rows")
    if table["columns_truncated"]:
        notes.append(f"{len(columns) - 1} of {table['total_columns']} data columns")
    note = f"<p>Showing {', '.join(notes)}.</p>" if notes else ""
    return f"<h3>{DIAGNOSTIC_TITLES[name]}</h3>{note}<table><tr>{header}</tr>{body}</table>"


def _pdf_report(report: ReportInput, chart_png: bytes | None) -> bytes:
    output = io.BytesIO()
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            textColor=colors.HexColor("#123b73"),
            alignment=TA_CENTER,
        )
    )
    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=20 * mm,
        bottomMargin=18 * mm,
        title="TabFM Workbench Report",
    )
    overview = report.eda.overview

    def cell(value: object, *, heading: bool = False) -> Paragraph:
        style = styles["BodyText"] if not heading else styles["BodyText"].clone("TableHeading")
        if heading:
            style.fontName = "Helvetica-Bold"
        style.fontSize = 7
        style.leading = 9
        return Paragraph(html.escape(str(value)), style)

    story: list[object] = [
        Paragraph("TabFM Workbench Report", styles["ReportTitle"]),
        Paragraph(html.escape(report.dataset_name), styles["Heading2"]),
        Spacer(1, 4 * mm),
        Paragraph("Dataset overview", styles["Heading2"]),
        Table(
            [
                [
                    cell("Rows", heading=True),
                    cell(overview.rows),
                    cell("Columns", heading=True),
                    cell(overview.columns),
                ],
                [
                    cell("Missing cells", heading=True),
                    cell(f"{overview.missing_cell_percent}%"),
                    cell("Duplicates", heading=True),
                    cell(overview.duplicate_rows),
                ],
            ],
            colWidths=[35 * mm, 30 * mm, 35 * mm, 30 * mm],
        ),
        Spacer(1, 5 * mm),
        Paragraph("Run provenance", styles["Heading2"]),
        Paragraph(
            f"Task: {report.task} | Target: {html.escape(report.target or 'None')} | "
            f"Mode: {report.prediction_mode} | Device: {html.escape(report.device)} | "
            f"Latency: {report.latency_ms:.1f}ms",
            styles["BodyText"],
        ),
        Paragraph("Metrics and diagnostics", styles["Heading2"]),
    ]
    metric_data = [
        [cell("Metric", heading=True), cell("Value", heading=True)],
        *[
            [cell(key), cell(value)]
            for key, value in sorted(report.diagnostics.metrics.items())
        ],
    ]
    metrics_table = Table(metric_data, colWidths=[80 * mm, 70 * mm], repeatRows=1)
    metrics_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(metrics_table)
    if chart_png is not None:
        story.extend(
            [Spacer(1, 5 * mm), Image(io.BytesIO(chart_png), width=165 * mm, height=69 * mm)]
        )
    for name, table_data in serialize_diagnostics(report.diagnostics).items():
        columns = table_data["columns"]
        rows = table_data["rows"]
        assert isinstance(columns, list) and isinstance(rows, list)
        pdf_columns = columns[:10]
        pdf_rows = [row[:10] for row in rows[:30]]
        story.append(Paragraph(DIAGNOSTIC_TITLES[name], styles["Heading3"]))
        if table_data["truncated"] or len(columns) > 10 or len(rows) > 30:
            story.append(
                Paragraph(
                    f"PDF preview shows {len(pdf_rows)} of {table_data['total_rows']} rows "
                    f"and {len(pdf_columns) - 1} of {table_data['total_columns']} data columns.",
                    styles["BodyText"],
                )
            )
        table_rows = [[cell(value, heading=True) for value in pdf_columns]] + [
            [cell(value) for value in row] for row in pdf_rows
        ]
        if table_rows:
            diagnostic_table = Table(
                table_rows,
                repeatRows=1,
                colWidths=[160 * mm / len(pdf_columns)] * len(pdf_columns),
            )
            diagnostic_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")),
                        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                    ]
                )
            )
            story.append(diagnostic_table)
    story.extend([PageBreak(), Paragraph("Schema and quality", styles["Heading2"])])
    quality_columns = ["column", "dtype", "missing_count", "unique_count"]
    available_columns = [
        column for column in quality_columns if column in report.eda.column_quality
    ]
    quality_values = report.eda.column_quality.loc[:, available_columns].head(100).values.tolist()
    quality_rows = [[cell(value, heading=True) for value in available_columns]] + [
        [cell(value) for value in row] for row in quality_values
    ]
    quality_table = Table(
        quality_rows, repeatRows=1, colWidths=[45 * mm, 35 * mm, 35 * mm, 35 * mm]
    )
    quality_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(quality_table)
    preview_columns = list(report.predictions.columns[:6])
    preview_values = report.predictions.loc[:, preview_columns].head(12).values.tolist()
    preview_rows = [[cell(value, heading=True) for value in preview_columns]] + [
        [cell(value) for value in row] for row in preview_values
    ]
    story.extend([Spacer(1, 4 * mm), Paragraph("Prediction preview", styles["Heading2"])])
    if preview_columns:
        preview_table = Table(
            preview_rows,
            repeatRows=1,
            colWidths=[160 * mm / len(preview_columns)] * len(preview_columns),
        )
        preview_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                ]
            )
        )
        story.append(preview_table)
    all_warnings = [*report.warnings, *report.diagnostics.warnings]
    if all_warnings:
        story.append(
            KeepTogether(
                [
                    Paragraph("Warnings", styles["Heading2"]),
                    *[
                        Paragraph(f"- {html.escape(item)}", styles["BodyText"])
                        for item in all_warnings
                    ],
                ]
            )
        )

    def decorate(canvas: object, doc: object) -> None:
        canvas.saveState()  # type: ignore[attr-defined]
        canvas.setFont("Helvetica", 8)  # type: ignore[attr-defined]
        canvas.setFillColor(colors.grey)  # type: ignore[attr-defined]
        canvas.drawString(18 * mm, 287 * mm, "TabFM Workbench Report")  # type: ignore[attr-defined]
        canvas.drawString(18 * mm, 10 * mm, "TabFM Workbench - local research use")  # type: ignore[attr-defined]
        canvas.drawRightString(192 * mm, 10 * mm, f"Page {doc.page}")  # type: ignore[attr-defined]
        canvas.restoreState()  # type: ignore[attr-defined]

    def invariant_canvas(*args: object, **kwargs: object) -> Canvas:
        kwargs["invariant"] = 1
        return Canvas(*args, **kwargs)

    document.build(
        story,
        onFirstPage=decorate,
        onLaterPages=decorate,
        canvasmaker=invariant_canvas,
    )
    return output.getvalue()
