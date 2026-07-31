from pathlib import Path

import pandas as pd

from tabfm_workbench.loader import DatasetArtifact
from tabfm_workbench.ui import (
    activate_artifact_state,
    batch_input_signature,
    clear_loaded_datasets_state,
    read_bundle_bytes,
    register_artifact_state,
    start_new_task_state,
    task_widget_key,
    update_batch_input_state,
    update_provider_filter_state,
)


def test_task_widget_key_rotates_with_generation() -> None:
    assert task_widget_key("target", 2) == "task_2_target"
    assert task_widget_key("target", 3) != task_widget_key("target", 2)


def test_clear_loaded_datasets_removes_artifacts_and_invalidates_widgets() -> None:
    state = {
        "artifacts": {"a.csv": object()},
        "active_artifact": "a.csv",
        "prepared_predictor": object(),
        "batch_result": object(),
        "context_uploads": object(),
        "upload_generation": 2,
        "task_generation": 7,
        "provider_download": "keep",
        "current_bundle": b"zip",
        "current_record": object(),
        "target_column": "label",
        "task_type": "classification",
        "prediction_mode": "Batch",
        "eda_numeric": "x",
        "context_uploads_2": object(),
        "task_7_batch_test_upload": object(),
        "history_repository": object(),
    }

    clear_loaded_datasets_state(state)

    assert state["artifacts"] == {}
    assert "active_artifact" not in state
    assert "prepared_predictor" not in state
    assert "batch_result" not in state
    assert "context_uploads" not in state
    assert state["upload_generation"] == 3
    assert state["task_generation"] == 8
    assert state["provider_download"] == "keep"
    assert "current_bundle" not in state
    assert "current_record" not in state
    assert "target_column" not in state
    assert "task_type" not in state
    assert "prediction_mode" not in state
    assert "eda_numeric" not in state
    assert "context_uploads_2" not in state
    assert "task_7_batch_test_upload" not in state
    assert "history_repository" in state


def test_start_new_task_keeps_artifacts_and_history_but_clears_task_outputs() -> None:
    artifacts = {"a.csv": object()}
    repository = object()
    state = {
        "artifacts": artifacts,
        "active_artifact": "a.csv",
        "history_repository": repository,
        "target_column": "label",
        "task_type": "classification",
        "prepared_predictor": object(),
        "batch_test_upload": object(),
        "batch_result": object(),
        "current_bundle": b"zip",
        "eda_numeric": "x",
        "task_generation": 4,
        "prediction_mode": "Batch",
    }

    start_new_task_state(state)

    assert state["artifacts"] is artifacts
    assert state["active_artifact"] == "a.csv"
    assert state["history_repository"] is repository
    assert state["task_generation"] == 5
    for key in (
        "target_column",
        "task_type",
        "prepared_predictor",
        "batch_result",
        "current_bundle",
        "eda_numeric",
        "prediction_mode",
    ):
        assert key not in state


def test_active_artifact_change_clears_task_outputs_but_keeps_artifacts() -> None:
    artifacts = {"a.csv": object(), "b.csv": object()}
    state = {
        "artifacts": artifacts,
        "active_artifact": "a.csv",
        "current_bundle": b"stale",
        "current_record": object(),
        "prepared_predictor": object(),
        "target_column": "label",
        "task_generation": 1,
    }

    activate_artifact_state(state, "b.csv")

    assert state["artifacts"] is artifacts
    assert state["active_artifact"] == "b.csv"
    assert state["task_generation"] == 2
    assert "current_bundle" not in state
    assert "prepared_predictor" not in state


def test_batch_input_change_clears_only_batch_output_and_latest_alias() -> None:
    state = {
        "batch_input_signature": "old",
        "batch_result": object(),
        "batch_features": object(),
        "batch_bundle": b"batch",
        "batch_record": object(),
        "current_bundle": b"batch",
        "current_record": object(),
        "single_result": object(),
        "single_bundle": b"single",
    }

    update_batch_input_state(state, "new")

    assert state["batch_input_signature"] == "new"
    for key in (
        "batch_result",
        "batch_features",
        "batch_bundle",
        "batch_record",
        "current_bundle",
        "current_record",
    ):
        assert key not in state
    assert "single_result" in state
    assert state["single_bundle"] == b"single"


def test_batch_signature_includes_ordered_columns_and_dtypes() -> None:
    values = [[1], [2]]
    column_a = pd.DataFrame(values, columns=["a"], dtype="int64")
    column_b = pd.DataFrame(values, columns=["b"], dtype="int64")
    floats = pd.DataFrame(values, columns=["a"], dtype="float64")

    signature = batch_input_signature("Separate test file", column_a, None)

    assert signature != batch_input_signature("Separate test file", column_b, None)
    assert signature != batch_input_signature("Separate test file", floats, None)

    state = {"batch_input_signature": signature, "batch_result": object(), "batch_bundle": b"zip"}
    update_batch_input_state(state, batch_input_signature("Separate test file", column_b, None))
    assert "batch_result" not in state
    assert "batch_bundle" not in state


def test_provider_filter_changes_clear_only_dependent_results() -> None:
    state = {
        "hf_query_value": "old",
        "hf_results": ["repo"],
        "hf_files": ["file"],
        "kaggle_results": ["keep"],
    }
    update_provider_filter_state(state, "hf_query", "new")
    assert "hf_results" not in state
    assert "hf_files" not in state
    assert state["kaggle_results"] == ["keep"]

    state.update({"mcp_provider_value": "Hugging Face", "mcp_results": [1]})
    update_provider_filter_state(state, "mcp_provider", "Kaggle")
    assert "mcp_results" not in state


def test_registering_new_artifact_invalidates_full_prior_task() -> None:
    state = {
        "artifacts": {"old.csv": object()},
        "active_artifact": "old.csv",
        "target_column": "label",
        "current_bundle": b"stale",
        "eda_numeric": "x",
        "task_generation": 2,
    }
    artifact = DatasetArtifact("new.csv", pd.DataFrame({"x": [1]}))
    register_artifact_state(state, artifact)
    assert state["active_artifact"] == "new.csv"
    assert state["task_generation"] == 3
    assert "target_column" not in state
    assert "current_bundle" not in state
    assert "eda_numeric" not in state


def test_history_bundle_read_handles_file_disappearing() -> None:
    missing = Path("data/definitely-missing-history-bundle.zip")
    data, error = read_bundle_bytes(missing)
    assert data is None
    assert error is not None
    assert "unavailable" in error
