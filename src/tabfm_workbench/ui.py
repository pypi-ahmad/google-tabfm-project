"""Streamlit views and session orchestration."""

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from .analytics import EvaluationDiagnostics, build_eda_snapshot
from .config import Settings
from .integrations import (
    discover_via_mcp,
    download_huggingface_file,
    download_kaggle_dataset,
    list_huggingface_files,
    list_kaggle_files,
    provider_status,
    search_huggingface_datasets,
    search_kaggle_datasets,
)
from .loader import DatasetArtifact, load_many, load_table
from .predictor import (
    PreparedPredictor,
    TaskType,
    context_fingerprint,
    load_tabfm_predictor,
    suggest_task,
)
from .remote import fetch_dataset
from .reports import (
    HistoryRepository,
    ReportInput,
    RunRecord,
    generate_report_bundle,
    sanitize_for_csv_export,
)

PREPARED_STATE_KEYS = (
    "prepared_predictor",
    "prepared_signature",
    "prepared_target",
    "context_signature",
    "batch_result",
    "batch_features",
    "single_result",
    "single_features",
)


def task_widget_key(name: str, generation: int) -> str:
    """Return a widget key invalidated by a new-task generation."""
    return f"task_{generation}_{name}"


def _task_key(name: str) -> str:
    return task_widget_key(name, int(st.session_state.get("task_generation", 0)))


def clear_loaded_datasets_state(state: Any) -> None:
    """Clear in-memory datasets and every task derived from them."""
    state["artifacts"] = {}
    exact = {
        "active_artifact",
        "context_uploads",
        "target_column",
        "task_type",
        "prediction_mode",
        "current_bundle",
        "current_record",
        *PREPARED_STATE_KEYS,
    }
    prefixes = ("context_uploads_", "batch_", "single_", "eda_", "report_")
    for key in list(state):
        if (
            key in exact
            or key.startswith(prefixes)
            or (key.startswith("task_") and key not in {"task_generation", "task_type"})
        ):
            state.pop(key, None)
    state["upload_generation"] = int(state.get("upload_generation", 0)) + 1
    state["task_generation"] = int(state.get("task_generation", 0)) + 1


def start_new_task_state(state: Any) -> None:
    """Reset task-local inputs and outputs while retaining datasets and history."""
    exact = {
        "target_column",
        "task_type",
        "batch_test_upload",
        "current_bundle",
        "current_record",
        "prediction_mode",
        *PREPARED_STATE_KEYS,
    }
    prefixes = ("single_", "eda_", "report_", "batch_")
    for key in list(state):
        if (
            key in exact
            or key.startswith(prefixes)
            or (key.startswith("task_") and key not in {"task_generation", "task_type"})
        ):
            state.pop(key, None)
    state["task_generation"] = int(state.get("task_generation", 0)) + 1


def activate_artifact_state(state: Any, name: str) -> None:
    """Select a dataset and invalidate all state derived from the prior dataset."""
    if state.get("active_artifact") != name:
        start_new_task_state(state)
        state["active_artifact"] = name


def register_artifact_state(state: Any, artifact: DatasetArtifact) -> str:
    """Register and activate a dataset after fully invalidating the prior task."""
    artifacts = dict(state.get("artifacts", {}))
    name = artifact.name
    if name in artifacts:
        stem, suffix = Path(name).stem, Path(name).suffix
        index = 2
        while f"{stem} ({index}){suffix}" in artifacts:
            index += 1
        name = f"{stem} ({index}){suffix}"
        artifact = DatasetArtifact(name, artifact.dataframe, artifact.source, artifact.warnings)
    start_new_task_state(state)
    artifacts[name] = artifact
    state["artifacts"] = artifacts
    state["active_artifact"] = name
    return name


def update_batch_input_state(state: Any, signature: str) -> None:
    """Invalidate batch output when its source or input data changes."""
    if state.get("batch_input_signature") == signature:
        return
    batch_bundle = state.get("batch_bundle")
    batch_record = state.get("batch_record")
    latest_is_batch = (
        batch_bundle is not None and state.get("current_bundle") == batch_bundle
    ) or (batch_record is not None and state.get("current_record") is batch_record)
    if latest_is_batch:
        state.pop("current_bundle", None)
        state.pop("current_record", None)
    for key in (
        "batch_result",
        "batch_features",
        "batch_bundle",
        "batch_record",
    ):
        state.pop(key, None)
    state["batch_input_signature"] = signature


def batch_input_signature(mode: str, features: pd.DataFrame, expected: pd.Series | None) -> str:
    """Fingerprint batch mode, ordered schema, values, index, and optional labels."""
    digest = hashlib.sha256(mode.encode("utf-8"))
    schema = [
        (str(column), str(dtype))
        for column, dtype in zip(features.columns, features.dtypes, strict=True)
    ]
    digest.update(json.dumps(schema, ensure_ascii=False).encode("utf-8"))
    digest.update(np.asarray(pd.util.hash_pandas_object(features, index=True)).tobytes())
    if expected is not None:
        expected_schema = (str(expected.name), str(expected.dtype))
        digest.update(json.dumps(expected_schema, ensure_ascii=False).encode("utf-8"))
        digest.update(np.asarray(pd.util.hash_pandas_object(expected, index=True)).tobytes())
    return digest.hexdigest()


def update_provider_filter_state(state: Any, field: str, value: str) -> None:
    """Clear provider results that depend on a changed filter."""
    value_key = f"{field}_value"
    if state.get(value_key) == value:
        return
    dependent = {
        "hf_query": ("hf_results", "hf_files"),
        "hf_repository": ("hf_files",),
        "kaggle_query": ("kaggle_results", "kaggle_files"),
        "kaggle_reference": ("kaggle_files",),
        "mcp_provider": ("mcp_results",),
        "mcp_query": ("mcp_results",),
    }
    for key in dependent[field]:
        state.pop(key, None)
    state[value_key] = value


def initialize_session_state(settings: Settings) -> None:
    """Initialize state shared by all direct page scripts."""
    st.session_state.setdefault("artifacts", {})
    st.session_state.setdefault("upload_generation", 0)
    st.session_state.setdefault("task_generation", 0)
    st.session_state.setdefault("history_repository", HistoryRepository(settings.tabfm_history_dir))


def render_sidebar(settings: Settings) -> None:
    """Show runtime and credential readiness without exposing secret values."""
    with st.sidebar:
        st.header("Runtime status")
        _status_line("License", settings.tabfm_accept_non_commercial_license)
        st.caption(f"Device preference: {settings.tabfm_device} · Ensemble: 8")
        hf = provider_status("huggingface", token=settings.hf_token)
        kaggle = provider_status(
            "kaggle",
            token=settings.kaggle_api_token,
            username=settings.kaggle_username,
            key=settings.kaggle_key,
        )
        _status_line("Hugging Face", hf.configured)
        _status_line("Kaggle", kaggle.configured)
        _status_line("HF MCP endpoint", bool(settings.hf_mcp_url))
        _status_line("Kaggle MCP endpoint", bool(settings.kaggle_mcp_url))
        if settings.tabfm_allow_insecure_http:
            st.warning("Plain HTTP imports enabled. Remote content can be altered in transit.")
        else:
            st.caption("Remote imports require HTTPS.")
        st.caption("Credentials are read from environment variables only.")
        if st.button("Start new task", icon=":material/restart_alt:", width="stretch"):
            start_new_task_state(st.session_state)
            st.rerun()


def render_data_loading(settings: Settings) -> None:
    """Load datasets from local files, URLs, providers, or MCP discovery."""
    st.subheader("Data Loading")
    st.caption("Load multiple files independently, then choose one active context dataset.")

    with st.expander("Local files", expanded=True):
        uploads = st.file_uploader(
            "Upload CSV, Parquet, or XLSX files",
            type=["csv", "parquet", "xlsx"],
            accept_multiple_files=True,
            key=f"context_uploads_{st.session_state.get('upload_generation', 0)}",
        )
        if st.button("Parse uploaded files", disabled=not uploads):
            oversized = [
                item.name
                for item in uploads
                if item.size > settings.tabfm_max_upload_mb * 1024 * 1024
            ]
            allowed = [item for item in uploads if item.name not in oversized]
            result = load_many([(item.name, item.getvalue()) for item in allowed])
            for artifact in result.artifacts:
                _register_artifact(artifact)
            for failure in result.failures:
                st.error(f"{failure.name}: {failure.message}")
            for name in oversized:
                st.error(f"{name}: upload exceeds configured size limit.")

    with st.expander("Direct URL"):
        with st.form("url_source"):
            url = st.text_input("HTTP or HTTPS table URL")
            submitted = st.form_submit_button("Fetch URL")
        if submitted:
            remote = fetch_dataset(
                url,
                max_bytes=settings.tabfm_max_download_mb * 1024 * 1024,
                allow_insecure_http=settings.tabfm_allow_insecure_http,
            )
            table = load_table(BytesIO(remote.content), remote.filename)
            _register_artifact(DatasetArtifact(remote.filename, table, source="url"))

    with st.expander("Hugging Face Hub"):
        query = st.text_input("Search Hugging Face datasets", key="hf_query")
        update_provider_filter_state(st.session_state, "hf_query", query)
        if st.button("Search Hugging Face", disabled=not query):
            st.session_state.hf_results = search_huggingface_datasets(
                query, token=settings.hf_token
            )
        repositories = st.session_state.get("hf_results", [])
        if repositories:
            repository = st.selectbox("Dataset repository", repositories, key="hf_repository")
            update_provider_filter_state(st.session_state, "hf_repository", repository)
            if st.button("List Hugging Face files"):
                st.session_state.hf_files = list_huggingface_files(
                    repository, token=settings.hf_token
                )
            files = st.session_state.get("hf_files", [])
            if files:
                filename = st.selectbox("Dataset file", files, key="hf_file")
                if st.button("Download Hugging Face file"):
                    path = download_huggingface_file(
                        repository,
                        filename,
                        token=settings.hf_token,
                        destination=Path("data/downloads/huggingface"),
                    )
                    _register_path(path, source="huggingface")

    with st.expander("Kaggle"):
        query = st.text_input("Search Kaggle datasets", key="kaggle_query")
        update_provider_filter_state(st.session_state, "kaggle_query", query)
        if st.button("Search Kaggle", disabled=not query):
            st.session_state.kaggle_results = search_kaggle_datasets(query)
        references = st.session_state.get("kaggle_results", [])
        if references:
            reference = st.selectbox("Kaggle dataset", references, key="kaggle_reference")
            update_provider_filter_state(st.session_state, "kaggle_reference", reference)
            if st.button("List Kaggle files"):
                st.session_state.kaggle_files = list_kaggle_files(reference)
            files = st.session_state.get("kaggle_files", [])
            if files and st.button("Download Kaggle dataset"):
                folder = download_kaggle_dataset(
                    reference,
                    destination=Path("data/downloads/kaggle") / reference.replace("/", "_"),
                )
                for filename in files:
                    path = folder / filename
                    if path.is_file():
                        _register_path(path, source="kaggle")

    with st.expander("MCP discovery"):
        st.caption("MCP is read-only discovery; downloads use native provider APIs.")
        provider = st.selectbox("MCP provider", ["Hugging Face", "Kaggle"])
        update_provider_filter_state(st.session_state, "mcp_provider", provider)
        endpoint = settings.hf_mcp_url if provider == "Hugging Face" else settings.kaggle_mcp_url
        query = st.text_input("Discovery query", key="mcp_query")
        update_provider_filter_state(st.session_state, "mcp_query", query)
        if st.button("Search MCP", disabled=not endpoint or not query) and endpoint:
            st.session_state.mcp_results = asyncio.run(discover_via_mcp(endpoint, query))
        if st.session_state.get("mcp_results"):
            st.json(st.session_state.mcp_results)

    artifacts: dict[str, DatasetArtifact] = st.session_state.get("artifacts", {})
    if not artifacts:
        st.info("No datasets loaded yet.")
        return
    if st.button("Clear loaded datasets", icon=":material/delete_sweep:"):
        _confirm_clear_loaded_datasets()
    names = list(artifacts)
    current = st.session_state.get("active_artifact", names[0])
    selected = st.selectbox("Active context dataset", names, index=names.index(current))
    activate_artifact_state(st.session_state, selected)
    artifact = artifacts[selected]
    left, middle, right = st.columns(3)
    left.metric("Rows", f"{len(artifact.dataframe):,}")
    middle.metric("Columns", len(artifact.dataframe.columns))
    right.metric("Source", artifact.source)
    st.dataframe(artifact.dataframe.head(100), width="stretch")


def render_model_context(settings: Settings) -> None:
    """Choose target/task and prepare in-context examples without weight updates."""
    table = _active_table()
    st.subheader("Model & Context")
    if table is None:
        st.info("Load and select a dataset first.")
        return
    st.caption(f"Active dataset: {st.session_state.get('active_artifact')}")
    if st.button("Clear loaded datasets", icon=":material/delete_sweep:", key="model_clear"):
        _confirm_clear_loaded_datasets()
    st.dataframe(table.head(20), width="stretch")
    prior_target = st.session_state.get("target_column")
    target_index = (
        list(table.columns).index(prior_target)
        if isinstance(prior_target, str) and prior_target in table.columns
        else 0
    )
    target = st.selectbox(
        "Target column",
        table.columns,
        index=target_index,
        key=_task_key("target_column"),
    )
    st.session_state.target_column = target
    labeled = table.loc[table[target].notna()].copy()
    suggestion = suggest_task(labeled[target])
    st.caption(f"Suggested {suggestion.task}: {suggestion.rationale}")
    tasks = ["classification", "regression"]
    prior_task = st.session_state.get("task_type")
    default_task = prior_task if prior_task in tasks else suggestion.task
    task = cast(
        TaskType,
        st.segmented_control(
            "Task type",
            tasks,
            default=default_task,
            key=_task_key("task_type"),
        ),
    )
    st.session_state.task_type = task
    features = labeled.drop(columns=[target])
    signature = context_fingerprint(features, labeled[target], task)
    if st.session_state.get("context_signature") not in {None, signature}:
        _invalidate_prepared_state()
    st.session_state.context_signature = signature

    left, middle, right = st.columns(3)
    left.metric("Labeled context rows", f"{len(labeled):,}")
    middle.metric("Features", len(features.columns))
    right.metric("Target values", labeled[target].nunique())
    st.info(
        "TabFM `fit()` prepares encoders and in-context examples. It does not train or modify "
        "pretrained weights."
    )
    if not settings.tabfm_accept_non_commercial_license:
        st.warning(
            "Review the TabFM non-commercial license, then set "
            "TABFM_ACCEPT_NON_COMMERCIAL_LICENSE=true to enable model loading."
        )
        return
    if st.button("Load TabFM and prepare context", type="primary", key=_task_key("prepare")):
        with st.spinner("Loading checkpoint and preparing context…"):
            predictor = load_tabfm_predictor(
                task,
                device=settings.tabfm_device,
                accept_non_commercial_license=settings.tabfm_accept_non_commercial_license,
            )
            predictor.prepare(features, labeled[target])
            st.session_state.prepared_predictor = predictor
            st.session_state.prepared_signature = signature
            st.session_state.prepared_target = target
        st.success("Context prepared. Batch and single-row predictions are ready.")
    if st.session_state.get("prepared_signature") == signature:
        st.success("Prepared context is current.")


def render_batch_predictions() -> None:
    """Predict blank-target rows or a separately uploaded table."""
    st.subheader("Batch Predictions")
    predictor = _current_predictor()
    table = _active_table()
    if predictor is None or table is None:
        st.info("Prepare a model context first.")
        return
    target = st.session_state.prepared_target
    mode = st.segmented_control(
        "Prediction source",
        ["Blank targets in context dataset", "Separate test file"],
        default="Blank targets in context dataset",
        key=_task_key("batch_source"),
    )
    expected: pd.Series | None = None
    if mode == "Blank targets in context dataset":
        tests = table.loc[table[target].isna()].copy()
        features = tests.drop(columns=[target])
    else:
        upload = st.file_uploader(
            "Test CSV, Parquet, or XLSX",
            type=["csv", "parquet", "xlsx"],
            key=_task_key("batch_test_upload"),
        )
        if upload is None:
            st.info("Upload a test table to continue.")
            return
        tests = load_table(BytesIO(upload.getvalue()), upload.name)
        expected = tests[target] if target in tests.columns else None
        features = tests.drop(columns=[target], errors="ignore")
    signature = batch_input_signature(str(mode), features, expected)
    update_batch_input_state(st.session_state, signature)
    if features.empty:
        st.warning("No prediction rows are available for this source.")
        return
    st.caption(f"Prediction rows: {len(features):,}")
    if st.button("Run batch prediction", type="primary", key=_task_key("batch_submit")):
        st.session_state.batch_result = predictor.predict(features, expected=expected)
        st.session_state.batch_features = features
        _archive_prediction(st.session_state.batch_result, features, expected, result_kind="batch")
    result = st.session_state.get("batch_result")
    if result is not None:
        _render_result(
            result,
            st.session_state.batch_features,
            "tabfm_batch_predictions.csv",
            st.session_state.get("batch_bundle"),
        )


def render_single_prediction() -> None:
    """Build one typed row from context schema and predict immediately."""
    st.subheader("Single Test Case")
    predictor = _current_predictor()
    table = _active_table()
    if predictor is None or table is None:
        st.info("Prepare a model context first.")
        return
    target = st.session_state.prepared_target
    features = table.drop(columns=[target])
    with st.form(_task_key("single_case")):
        row = {column: _feature_input(column, features[column]) for column in features.columns}
        submitted = st.form_submit_button("Predict single case", type="primary")
    if submitted:
        st.session_state.single_features = pd.DataFrame([row])
        st.session_state.single_result = predictor.predict(st.session_state.single_features)
        _archive_prediction(
            st.session_state.single_result,
            st.session_state.single_features,
            None,
            result_kind="single",
        )
    result = st.session_state.get("single_result")
    if result is not None:
        st.caption("Latest submitted case")
        _render_result(
            result,
            st.session_state.single_features,
            "tabfm_single_prediction.csv",
            st.session_state.get("single_bundle"),
        )


def _feature_input(name: str, values: pd.Series) -> Any:
    non_null = values.dropna()
    if pd.api.types.is_bool_dtype(values):
        return st.selectbox(name, [False, True], key=_task_key(f"single_{name}"))
    if pd.api.types.is_numeric_dtype(values):
        numeric_default = float(non_null.median()) if not non_null.empty else 0.0
        return st.number_input(name, value=numeric_default, key=_task_key(f"single_{name}"))
    if pd.api.types.is_datetime64_any_dtype(values):
        date_default = (
            non_null.iloc[0].date() if not non_null.empty else pd.Timestamp.today().date()
        )
        return pd.Timestamp(
            st.date_input(name, value=date_default, key=_task_key(f"single_{name}"))
        )
    categories = sorted(map(str, non_null.unique()))
    if categories and len(categories) <= 50:
        return st.selectbox(name, categories, key=_task_key(f"single_{name}"))
    text_default = str(non_null.iloc[0]) if not non_null.empty else ""
    return st.text_input(name, value=text_default, key=_task_key(f"single_{name}"))


def _render_result(
    result: Any, features: pd.DataFrame, filename: str, bundle: bytes | None
) -> None:
    for warning in collect_result_warnings(result):
        st.warning(warning)
    if result.metrics:
        with st.container(horizontal=True):
            for name, value in result.metrics.items():
                st.metric(name.replace("_", " ").title(), f"{value:.4g}", border=True)
    if result.diagnostics is not None:
        render_evaluation_diagnostics(result.diagnostics)
    st.caption(f"Device: {result.device} · Inference latency: {result.latency_ms:.1f} ms")
    output = features.copy()
    output["prediction"] = result.predictions
    if result.probabilities is not None:
        output = output.join(result.probabilities.add_prefix("probability_"))
    st.dataframe(output, width="stretch")
    st.download_button(
        "Download predictions",
        sanitize_for_csv_export(output).to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )
    if bundle is not None:
        st.download_button(
            "Download report bundle",
            bundle,
            file_name="tabfm_report_bundle.zip",
            mime="application/zip",
            icon=":material/download:",
        )


def collect_result_warnings(result: Any) -> tuple[str, ...]:
    """Merge immediate and metric warnings once, preserving display order."""
    warnings = list(result.warnings)
    diagnostics = result.diagnostics
    if diagnostics is not None:
        warnings.extend(diagnostics.warnings)
    return tuple(dict.fromkeys(warnings))


def render_evaluation_diagnostics(diagnostics: EvaluationDiagnostics) -> None:
    """Render available evaluation details with bounded responsive previews."""
    st.subheader("Evaluation diagnostics")
    if diagnostics.confusion_matrix is not None:
        left, right = st.columns(2)
        with left:
            st.markdown("**Confusion matrix**")
            st.dataframe(
                diagnostics.confusion_matrix.iloc[:50, :50], width="stretch"
            )
        with right:
            st.markdown("**Per-class metrics**")
            if diagnostics.per_class is not None:
                st.dataframe(diagnostics.per_class.head(100), hide_index=True, width="stretch")
        if diagnostics.roc_curve is not None:
            st.altair_chart(build_roc_curve_chart(diagnostics.roc_curve), width="stretch")
        if diagnostics.probability_diagnostics is not None:
            st.markdown("**Probability diagnostics preview**")
            st.dataframe(
                diagnostics.probability_diagnostics.head(100), width="stretch"
            )
        return
    if diagnostics.actual_vs_predicted is not None:
        left, right = st.columns(2)
        with left:
            st.altair_chart(
                build_actual_vs_predicted_chart(diagnostics.actual_vs_predicted.head(5_000)),
                width="stretch",
            )
        with right:
            if diagnostics.residuals is not None:
                st.altair_chart(
                    build_residual_chart(diagnostics.residuals.head(5_000)),
                    width="stretch",
                )
        st.markdown("**Actual vs predicted preview**")
        st.dataframe(diagnostics.actual_vs_predicted.head(100), width="stretch")
    if diagnostics.error_quantiles is not None:
        st.markdown("**Error quantiles**")
        st.dataframe(diagnostics.error_quantiles, hide_index=True, width="stretch")


def build_roc_curve_chart(frame: pd.DataFrame) -> alt.Chart:
    return (  # type: ignore[no-any-return]
        alt.Chart(frame)
        .mark_line()
        .encode(
            x=alt.X(field="false_positive_rate", type="quantitative", title="False positive rate"),
            y=alt.Y(field="true_positive_rate", type="quantitative", title="True positive rate"),
            tooltip=["false_positive_rate:Q", "true_positive_rate:Q", "threshold:Q"],
        )
        .properties(title="ROC curve")
    )


def build_actual_vs_predicted_chart(frame: pd.DataFrame) -> alt.Chart:
    return (  # type: ignore[no-any-return]
        alt.Chart(frame)
        .mark_circle(opacity=0.65)
        .encode(
            x=alt.X(field="actual", type="quantitative", title="Actual"),
            y=alt.Y(field="predicted", type="quantitative", title="Predicted"),
            tooltip=["actual:Q", "predicted:Q"],
        )
        .properties(title="Actual vs predicted")
    )


def build_residual_chart(frame: pd.DataFrame) -> alt.Chart:
    return (  # type: ignore[no-any-return]
        alt.Chart(frame)
        .mark_circle(opacity=0.65)
        .encode(
            x=alt.X(field="predicted", type="quantitative", title="Predicted"),
            y=alt.Y(field="residual", type="quantitative", title="Residual"),
            tooltip=["predicted:Q", "residual:Q", "absolute_error:Q"],
        )
        .properties(title="Residual diagnostics")
    )


def _register_path(path: Path, *, source: str) -> None:
    table = load_table(BytesIO(path.read_bytes()), path.name)
    _register_artifact(DatasetArtifact(path.name, table, source=source))


def _register_artifact(artifact: DatasetArtifact) -> None:
    register_artifact_state(st.session_state, artifact)


def _active_table() -> pd.DataFrame | None:
    artifacts = st.session_state.get("artifacts", {})
    active = st.session_state.get("active_artifact")
    artifact = artifacts.get(active) if isinstance(active, str) else None
    return artifact.dataframe if artifact else None


def _current_predictor() -> PreparedPredictor | None:
    if st.session_state.get("prepared_signature") != st.session_state.get("context_signature"):
        return None
    return st.session_state.get("prepared_predictor")


def _invalidate_prepared_state() -> None:
    for key in (
        "prepared_predictor",
        "prepared_signature",
        "prepared_target",
        "batch_result",
        "single_result",
    ):
        st.session_state.pop(key, None)


def _status_line(label: str, ready: bool) -> None:
    st.write(f"{'Ready' if ready else 'Not configured'} · {label}")


@st.dialog("Clear loaded datasets?")
def _confirm_clear_loaded_datasets() -> None:
    st.warning("This removes loaded datasets and all in-memory model and prediction state.")
    if st.button("Clear datasets", type="primary", icon=":material/delete:"):
        clear_loaded_datasets_state(st.session_state)
        st.rerun()


def _archive_prediction(
    result: Any,
    features: pd.DataFrame,
    expected: pd.Series | None,
    *,
    result_kind: str,
) -> None:
    """Build and persist one report bundle, called only from prediction submit actions."""
    artifacts: dict[str, DatasetArtifact] = st.session_state.get("artifacts", {})
    active = st.session_state.get("active_artifact")
    artifact = artifacts.get(active) if isinstance(active, str) else None
    if artifact is None:
        return
    output = features.copy()
    output["prediction"] = result.predictions
    if result.probabilities is not None:
        output = output.join(result.probabilities.add_prefix("probability_"))
    diagnostics = result.diagnostics or EvaluationDiagnostics(dict(result.metrics), ())
    report = ReportInput(
        dataset_name=artifact.name,
        dataset_source=artifact.source,
        task=st.session_state.get("task_type", "regression"),
        target=st.session_state.get("prepared_target"),
        prediction_mode=(
            "evaluation" if expected is not None and expected.notna().any() else "predict"
        ),
        predictions=output,
        eda=_cached_eda(artifact.dataframe),
        diagnostics=diagnostics,
        created_at=datetime.now(UTC),
        latency_ms=result.latency_ms,
        device=result.device,
        warnings=tuple(result.warnings),
    )
    repository: HistoryRepository = st.session_state.history_repository
    try:
        with st.spinner("Generating and saving report bundle…"):
            bundle = generate_report_bundle(report)
            record = repository.create(RunRecord.from_report(report), bundle.data)
        st.session_state[f"{result_kind}_bundle"] = bundle.data
        st.session_state[f"{result_kind}_record"] = record
        st.session_state.current_bundle = bundle.data
        st.session_state.current_record = record
    except (OSError, RuntimeError, ValueError) as exc:
        st.session_state.pop("current_bundle", None)
        st.session_state.pop(f"{result_kind}_bundle", None)
        st.session_state.pop(f"{result_kind}_record", None)
        try:
            failed = repository.create(RunRecord.failed(report, str(exc)))
            st.session_state[f"{result_kind}_record"] = failed
            st.session_state.current_record = failed
        except (OSError, RuntimeError, ValueError):
            st.session_state.pop("current_record", None)
        st.warning(f"Predictions are ready, but the report bundle could not be saved: {exc}")


@st.cache_data(max_entries=8, show_spinner="Building EDA snapshot…")
def _cached_eda(frame: pd.DataFrame) -> Any:
    return build_eda_snapshot(frame)


def build_numeric_histogram_chart(frame: pd.DataFrame, column: str) -> alt.Chart:
    return (  # type: ignore[no-any-return]
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X(field=column, type="quantitative", bin=True),
            y=alt.Y(aggregate="count", type="quantitative"),
        )
    )


def build_numeric_box_chart(frame: pd.DataFrame, column: str) -> alt.Chart:
    return (  # type: ignore[no-any-return]
        alt.Chart(frame).mark_boxplot().encode(x=alt.X(field=column, type="quantitative"))
    )


def build_categorical_numeric_box_chart(
    frame: pd.DataFrame, category: str, value: str
) -> alt.Chart:
    return (  # type: ignore[no-any-return]
        alt.Chart(frame)
        .mark_boxplot()
        .encode(
            x=alt.X(field=category, type="nominal"),
            y=alt.Y(field=value, type="quantitative"),
        )
    )


def read_bundle_bytes(path: Path) -> tuple[bytes | None, str | None]:
    """Read a history bundle without crashing if it disappears after listing."""
    try:
        return path.read_bytes(), None
    except OSError as exc:
        return None, f"The report bundle became unavailable: {exc}"


def render_eda_reports() -> None:
    """Render bounded exploratory analysis and the latest report bundle."""
    frame = _active_table()
    if frame is None:
        st.info("Load and select a dataset first.")
        return
    snapshot = _cached_eda(frame)
    with st.container(horizontal=True):
        st.metric("Rows", f"{snapshot.overview.rows:,}", border=True)
        st.metric("Columns", snapshot.overview.columns, border=True)
        st.metric("Missing cells", f"{snapshot.overview.missing_cell_percent:.2f}%", border=True)
        st.metric("Duplicates", f"{snapshot.overview.duplicate_rows:,}", border=True)
    st.caption(
        "Charts use deterministic samples capped at 10,000 rows; scatter plots use "
        "5,000 rows and correlations use at most 20 numeric columns."
    )
    st.subheader("Schema and quality")
    st.dataframe(snapshot.column_quality, hide_index=True, width="stretch")
    missing = snapshot.column_quality.loc[:, ["column", "missing_percent"]]
    if not missing.empty:
        st.bar_chart(missing, x="column", y="missing_percent")
    left, right = st.columns(2)
    with left:
        st.subheader("Numeric summary")
        st.dataframe(snapshot.numeric_summary, hide_index=True, width="stretch")
    with right:
        st.subheader("Categorical summary")
        st.dataframe(snapshot.categorical_summary, hide_index=True, width="stretch")
    sample = snapshot.sample
    numeric = [str(column) for column in snapshot.numeric_summary["column"].tolist()]
    categorical = [str(column) for column in snapshot.categorical_summary["column"].tolist()]
    st.subheader("Univariate analysis")
    if numeric:
        column = st.selectbox("Numeric column", numeric, key=_task_key("eda_numeric"))
        kind = st.segmented_control(
            "Numeric chart",
            ["Histogram", "Box"],
            default="Histogram",
            key=_task_key("eda_numeric_chart"),
        )
        if kind == "Histogram":
            chart = build_numeric_histogram_chart(sample, column)
        else:
            chart = build_numeric_box_chart(sample, column)
        st.altair_chart(chart, width="stretch")
    if categorical:
        column = st.selectbox("Categorical column", categorical, key=_task_key("eda_categorical"))
        counts = (
            sample[column]
            .astype("string")
            .value_counts(dropna=False)
            .head(20)
            .rename_axis(column)
            .reset_index(name="count")
        )
        st.bar_chart(counts, x=column, y="count")
    st.subheader("Bivariate analysis")
    if len(numeric) >= 2:
        x = st.selectbox("X axis", numeric, key=_task_key("eda_x"))
        y = st.selectbox("Y axis", numeric, index=1, key=_task_key("eda_y"))
        st.scatter_chart(snapshot.scatter_sample, x=x, y=y)
    if categorical and numeric:
        category = st.selectbox("Category", categorical, key=_task_key("eda_bivariate_category"))
        value = st.selectbox("Value", numeric, key=_task_key("eda_bivariate_value"))
        top = set(sample[category].value_counts().head(20).index)
        chart_data = sample.loc[sample[category].isin(top), [category, value]]
        st.altair_chart(
            build_categorical_numeric_box_chart(chart_data, category, value),
            width="stretch",
        )
    if not snapshot.correlations.empty:
        st.subheader("Correlation heatmap")
        corr = (
            snapshot.correlations.rename_axis("feature")
            .reset_index()
            .melt("feature", var_name="compared", value_name="correlation")
        )
        heatmap = (
            alt.Chart(corr)
            .mark_rect()
            .encode(
                x=alt.X(field="feature", type="nominal"),
                y=alt.Y(field="compared", type="nominal"),
                color=alt.Color(
                    field="correlation",
                    type="quantitative",
                    scale=alt.Scale(domain=[-1, 1]),
                ),
                tooltip=["feature", "compared", "correlation"],
            )
        )
        st.altair_chart(heatmap, width="stretch")
    target = st.session_state.get("prepared_target") or st.session_state.get("target_column")
    if target in frame.columns:
        st.subheader("Target analysis")
        task = st.session_state.get("task_type")
        if task == "classification" or target not in numeric:
            counts = (
                frame[target]
                .astype("string")
                .value_counts(dropna=False)
                .head(20)
                .rename_axis("target")
                .reset_index(name="count")
            )
            st.bar_chart(counts, x="target", y="count")
        else:
            st.altair_chart(
                build_numeric_histogram_chart(snapshot.sample, target),
                width="stretch",
            )
    if st.session_state.get("current_bundle") is not None:
        st.download_button(
            "Download latest report bundle",
            st.session_state.current_bundle,
            file_name="tabfm_report_bundle.zip",
            mime="application/zip",
            icon=":material/download:",
        )


def render_history(settings: Settings) -> None:
    """Render durable newest-first run history."""
    repository: HistoryRepository = st.session_state.history_repository
    st.warning(
        f"History is stored permanently on this computer at {settings.tabfm_history_dir.resolve()}."
    )
    for warning in st.session_state.pop("history_cleanup_warnings", ()):
        st.warning(f"History metadata was cleared, but cleanup was incomplete: {warning}")
    page = int(st.number_input("Page", min_value=1, step=1, value=1, key="history_page"))
    records = repository.list(page)
    if not records:
        st.info("No saved runs on this page.")
    else:
        summary = pd.DataFrame(
            [
                {
                    "Created": record.created_at,
                    "Dataset": record.dataset_name,
                    "Task": record.task,
                    "Rows": record.row_count,
                    "Status": record.status,
                }
                for record in records
            ]
        )
        st.dataframe(summary, hide_index=True, width="stretch")
        labels = {
            f"{record.created_at:%Y-%m-%d %H:%M} · {record.dataset_name} · {record.id[:8]}": record
            for record in records
        }
        selected = labels[st.selectbox("Run", list(labels), key="history_run")]
        st.json(
            {
                "task": selected.task,
                "target": selected.target,
                "mode": selected.prediction_mode,
                "metrics": selected.metrics,
                "warnings": selected.warnings,
                "latency_ms": selected.latency_ms,
                "device": selected.device,
                "status": selected.status,
                "error": selected.error,
            }
        )
        if selected.status == "available" and selected.bundle_path is not None:
            bundle, error = read_bundle_bytes(selected.bundle_path)
            if bundle is not None:
                st.download_button(
                    "Download report bundle",
                    bundle,
                    file_name=f"tabfm-{selected.id}.zip",
                    mime="application/zip",
                    icon=":material/download:",
                )
            else:
                st.warning(error)
        else:
            st.warning(selected.error or "The report bundle is unavailable.")
    if st.button("Clear history", icon=":material/delete_forever:"):
        _confirm_clear_history(repository)


@st.dialog("Clear permanent history?")
def _confirm_clear_history(repository: HistoryRepository) -> None:
    st.warning("This permanently removes indexed run metadata and its report bundles.")
    if st.button("Clear history permanently", type="primary", icon=":material/delete_forever:"):
        warnings = repository.clear()
        st.session_state.pop("current_bundle", None)
        st.session_state.pop("current_record", None)
        if warnings:
            st.session_state.history_cleanup_warnings = warnings
        st.rerun()
