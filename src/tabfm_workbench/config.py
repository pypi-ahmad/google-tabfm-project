"""Environment-backed application settings."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration loaded from environment variables or an optional env file."""

    model_config = SettingsConfigDict(env_file=None, extra="ignore")

    tabfm_accept_non_commercial_license: bool = False
    tabfm_device: Literal["auto", "cuda", "cpu"] = "auto"
    tabfm_model_cache_dir: Path = Path("data/cache/models")
    tabfm_session_ttl_hours: int = Field(default=24, ge=1, le=168)
    tabfm_max_upload_mb: int = Field(default=500, ge=1, le=2048)
    tabfm_max_download_mb: int = Field(default=500, ge=1, le=2048)
    tabfm_allow_insecure_http: bool = False
    hf_token: str | None = None
    kaggle_api_token: str | None = None
    kaggle_username: str | None = None
    kaggle_key: str | None = None
    hf_mcp_url: str | None = None
    kaggle_mcp_url: str | None = "https://www.kaggle.com/mcp"

    def assert_model_use_allowed(self) -> None:
        """Block weight loading until user accepts license constraints."""
        if not self.tabfm_accept_non_commercial_license:
            raise ValueError(
                "TabFM weights use a non-commercial license. Set "
                "TABFM_ACCEPT_NON_COMMERCIAL_LICENSE=true only after reviewing it."
            )
