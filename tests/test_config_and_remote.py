from pathlib import Path

import httpx
import pytest

from tabfm_workbench.config import Settings
from tabfm_workbench.remote import RemoteFetchError, fetch_dataset, validate_remote_url


def test_history_directory_has_local_default_and_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert Settings().tabfm_history_dir == Path("data/sessions/history")
    monkeypatch.setenv("TABFM_HISTORY_DIR", "custom/history")
    assert Settings().tabfm_history_dir == Path("custom/history")


def test_settings_require_license_acknowledgement_for_model_use() -> None:
    settings = Settings(
        tabfm_accept_non_commercial_license=False,
    )
    with pytest.raises(ValueError, match="non-commercial"):
        settings.assert_model_use_allowed()


def test_remote_fetch_rejects_http() -> None:
    with pytest.raises(RemoteFetchError, match="HTTPS"):
        fetch_dataset("http://example.com/data.csv")


def test_remote_fetch_allows_http_only_when_explicitly_enabled() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, content=b"x\n1\n", request=request)
    )
    with httpx.Client(transport=transport) as client:
        result = fetch_dataset(
            "http://example.com/data.csv",
            allow_insecure_http=True,
            client=client,
        )
    assert result.content == b"x\n1\n"


def test_remote_fetch_limits_streamed_size() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, content=b"a" * 12, request=request)
    )
    with (
        httpx.Client(transport=transport) as client,
        pytest.raises(RemoteFetchError, match="size limit"),
    ):
        fetch_dataset("https://example.com/data.csv", max_bytes=10, client=client)


def test_remote_fetch_returns_filename_and_bytes() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, content=b"x,y\n1,2\n", request=request)
    )
    with httpx.Client(transport=transport) as client:
        result = fetch_dataset("https://example.com/files/data.csv", client=client)
    assert result.filename == "data.csv"
    assert result.content == b"x,y\n1,2\n"


def test_remote_url_rejects_hostname_resolving_to_loopback() -> None:
    def loopback_resolver(hostname: str, port: int) -> list[tuple[object, ...]]:
        return [(None, None, None, None, ("127.0.0.1", port))]

    with pytest.raises(RemoteFetchError, match="[Pp]rivate or local"):
        validate_remote_url(
            "https://internal.example/data.csv",
            resolver=loopback_resolver,
        )
