from types import SimpleNamespace

import pytest

from tabfm_workbench.integrations import (
    ProviderError,
    list_huggingface_files,
    list_kaggle_files,
    normalize_dataset_filename,
    provider_status,
)


def test_normalize_dataset_filename_blocks_traversal() -> None:
    with pytest.raises(ProviderError, match="filename"):
        normalize_dataset_filename("../secret.csv")


def test_provider_status_never_exposes_secret() -> None:
    status = provider_status("huggingface", token="hf_super_secret")
    assert status.configured is True
    assert "secret" not in status.message


def test_provider_status_reports_missing_kaggle_credentials() -> None:
    status = provider_status("kaggle")
    assert status.configured is False


def test_huggingface_file_listing_keeps_supported_tables() -> None:
    api = SimpleNamespace(
        list_repo_files=lambda **kwargs: ["README.md", "data/train.csv", "data/test.parquet"]
    )
    assert list_huggingface_files("owner/data", token=None, api=api) == [
        "data/test.parquet",
        "data/train.csv",
    ]


def test_kaggle_file_listing_keeps_supported_tables() -> None:
    api = SimpleNamespace(
        authenticate=lambda: None,
        dataset_list_files=lambda reference: SimpleNamespace(
            files=[SimpleNamespace(name="notes.txt"), SimpleNamespace(name="train.xlsx")]
        ),
    )
    assert list_kaggle_files("owner/data", api=api) == ["train.xlsx"]
