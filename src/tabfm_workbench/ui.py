"""Streamlit views and session orchestration."""

import asyncio
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import pandas as pd
import streamlit as st

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
        if st.button("Clear session", use_container_width=True):
            st.session_state.clear()
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
            key="context_uploads",
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
        if st.button("Search Hugging Face", disabled=not query):
            st.session_state.hf_results = search_huggingface_datasets(
                query, token=settings.hf_token
            )
        repositories = st.session_state.get("hf_results", [])
        if repositories:
            repository = st.selectbox("Dataset repository", repositories, key="hf_repository")
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
        if st.button("Search Kaggle", disabled=not query):
            st.session_state.kaggle_results = search_kaggle_datasets(query)
        references = st.session_state.get("kaggle_results", [])
        if references:
            reference = st.selectbox("Kaggle dataset", references, key="kaggle_reference")
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
        endpoint = settings.hf_mcp_url if provider == "Hugging Face" else settings.kaggle_mcp_url
        query = st.text_input("Discovery query", key="mcp_query")
        if st.button("Search MCP", disabled=not endpoint or not query) and endpoint:
            st.session_state.mcp_results = asyncio.run(discover_via_mcp(endpoint, query))
        if st.session_state.get("mcp_results"):
            st.json(st.session_state.mcp_results)

    artifacts: dict[str, DatasetArtifact] = st.session_state.get("artifacts", {})
    if not artifacts:
        st.info("No datasets loaded yet.")
        return
    names = list(artifacts)
    current = st.session_state.get("active_artifact", names[0])
    selected = st.selectbox("Active context dataset", names, index=names.index(current))
    if selected != current:
        _invalidate_prepared_state()
    st.session_state.active_artifact = selected
    artifact = artifacts[selected]
    left, middle, right = st.columns(3)
    left.metric("Rows", f"{len(artifact.dataframe):,}")
    middle.metric("Columns", len(artifact.dataframe.columns))
    right.metric("Source", artifact.source)
    st.dataframe(artifact.dataframe.head(100), use_container_width=True)


def render_model_context(settings: Settings) -> None:
    """Choose target/task and prepare in-context examples without weight updates."""
    table = _active_table()
    st.subheader("Model & Context")
    if table is None:
        st.info("Load and select a dataset first.")
        return
    target = st.selectbox("Target column", table.columns, key="target_column")
    labeled = table.loc[table[target].notna()].copy()
    suggestion = suggest_task(labeled[target])
    st.caption(f"Suggested {suggestion.task}: {suggestion.rationale}")
    tasks = ["classification", "regression"]
    task = cast(
        TaskType,
        st.radio(
            "Task type",
            tasks,
            index=tasks.index(suggestion.task),
            horizontal=True,
            key="task_type",
        ),
    )
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
    if st.button("Load TabFM and prepare context", type="primary"):
        with st.spinner("Loading checkpoint and preparing context…"):
            predictor = load_tabfm_predictor(task, device=settings.tabfm_device)
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
    mode = st.radio(
        "Prediction source",
        ["Blank targets in context dataset", "Separate test file"],
        horizontal=True,
    )
    expected: pd.Series | None = None
    if mode == "Blank targets in context dataset":
        tests = table.loc[table[target].isna()].copy()
        features = tests.drop(columns=[target])
    else:
        upload = st.file_uploader(
            "Test CSV, Parquet, or XLSX",
            type=["csv", "parquet", "xlsx"],
            key="batch_test_upload",
        )
        if upload is None:
            st.info("Upload a test table to continue.")
            return
        tests = load_table(BytesIO(upload.getvalue()), upload.name)
        expected = tests[target] if target in tests.columns else None
        features = tests.drop(columns=[target], errors="ignore")
    if features.empty:
        st.warning("No prediction rows are available for this source.")
        return
    st.caption(f"Prediction rows: {len(features):,}")
    if st.button("Run batch prediction", type="primary"):
        st.session_state.batch_result = predictor.predict(features, expected=expected)
        st.session_state.batch_features = features
    result = st.session_state.get("batch_result")
    if result is not None:
        _render_result(result, st.session_state.batch_features, "tabfm_batch_predictions.csv")


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
    with st.form("single_case"):
        row = {column: _feature_input(column, features[column]) for column in features.columns}
        submitted = st.form_submit_button("Predict single case", type="primary")
    if submitted:
        st.session_state.single_features = pd.DataFrame([row])
        st.session_state.single_result = predictor.predict(st.session_state.single_features)
    result = st.session_state.get("single_result")
    if result is not None:
        _render_result(result, st.session_state.single_features, "tabfm_single_prediction.csv")


def _feature_input(name: str, values: pd.Series) -> Any:
    non_null = values.dropna()
    if pd.api.types.is_bool_dtype(values):
        return st.selectbox(name, [False, True], key=f"single_{name}")
    if pd.api.types.is_numeric_dtype(values):
        numeric_default = float(non_null.median()) if not non_null.empty else 0.0
        return st.number_input(name, value=numeric_default, key=f"single_{name}")
    if pd.api.types.is_datetime64_any_dtype(values):
        date_default = (
            non_null.iloc[0].date() if not non_null.empty else pd.Timestamp.today().date()
        )
        return pd.Timestamp(st.date_input(name, value=date_default, key=f"single_{name}"))
    categories = sorted(map(str, non_null.unique()))
    if categories and len(categories) <= 50:
        return st.selectbox(name, categories, key=f"single_{name}")
    text_default = str(non_null.iloc[0]) if not non_null.empty else ""
    return st.text_input(name, value=text_default, key=f"single_{name}")


def _render_result(result: Any, features: pd.DataFrame, filename: str) -> None:
    for warning in result.warnings:
        st.warning(warning)
    if result.metrics:
        columns = st.columns(len(result.metrics))
        for column, (name, value) in zip(columns, result.metrics.items(), strict=True):
            column.metric(name.replace("_", " ").title(), f"{value:.4g}")
    st.caption(f"Device: {result.device} · Inference latency: {result.latency_ms:.1f} ms")
    output = features.copy()
    output["prediction"] = result.predictions
    if result.probabilities is not None:
        output = output.join(result.probabilities.add_prefix("probability_"))
    st.dataframe(output, use_container_width=True)
    st.download_button(
        "Download predictions",
        output.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


def _register_path(path: Path, *, source: str) -> None:
    table = load_table(BytesIO(path.read_bytes()), path.name)
    _register_artifact(DatasetArtifact(path.name, table, source=source))


def _register_artifact(artifact: DatasetArtifact) -> None:
    artifacts = dict(st.session_state.get("artifacts", {}))
    name = artifact.name
    if name in artifacts:
        stem, suffix = Path(name).stem, Path(name).suffix
        index = 2
        while f"{stem} ({index}){suffix}" in artifacts:
            index += 1
        name = f"{stem} ({index}){suffix}"
        artifact = DatasetArtifact(name, artifact.dataframe, artifact.source, artifact.warnings)
    artifacts[name] = artifact
    st.session_state.artifacts = artifacts
    st.session_state.active_artifact = name
    _invalidate_prepared_state()


def _active_table() -> pd.DataFrame | None:
    artifacts = st.session_state.get("artifacts", {})
    active = st.session_state.get("active_artifact")
    artifact = artifacts.get(active)
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
