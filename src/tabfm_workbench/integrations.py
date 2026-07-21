"""Native Kaggle/Hugging Face adapters and optional MCP discovery."""

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

SUPPORTED_TABLE_SUFFIXES = (".csv", ".parquet", ".xlsx")


class ProviderError(RuntimeError):
    """Raised when a dataset provider operation fails."""


@dataclass(frozen=True)
class ProviderStatus:
    configured: bool
    message: str


def normalize_dataset_filename(filename: str) -> str:
    """Return a leaf filename suitable for workspace storage."""
    leaf = PurePosixPath(filename.replace("\\", "/"))
    if leaf.name != str(leaf) or leaf.name in {"", ".", ".."}:
        raise ProviderError("Invalid dataset filename.")
    return leaf.name


def provider_status(
    provider: Literal["huggingface", "kaggle"],
    *,
    token: str | None = None,
    username: str | None = None,
    key: str | None = None,
) -> ProviderStatus:
    """Describe credential availability without echoing secret material."""
    configured = bool(token) if provider == "huggingface" else bool(token or (username and key))
    return ProviderStatus(
        configured=configured,
        message="Credentials configured" if configured else "Credentials not configured",
    )


def search_huggingface_datasets(query: str, *, token: str | None, limit: int = 20) -> list[str]:
    try:
        from huggingface_hub import HfApi

        return [item.id for item in HfApi(token=token).list_datasets(search=query, limit=limit)]
    except ImportError as exc:
        raise ProviderError("Install integrations extra to use Hugging Face.") from exc
    except Exception as exc:
        raise ProviderError(f"Hugging Face search failed: {exc}") from exc


def list_huggingface_files(
    repo_id: str,
    *,
    token: str | None,
    api: Any | None = None,
) -> list[str]:
    """List supported table files in a Hugging Face dataset repository."""
    try:
        if api is None:
            from huggingface_hub import HfApi

            api = HfApi(token=token)
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=token)
        return sorted(name for name in files if name.lower().endswith(SUPPORTED_TABLE_SUFFIXES))
    except ImportError as exc:
        raise ProviderError("Install integrations extra to use Hugging Face.") from exc
    except Exception as exc:
        raise ProviderError(f"Hugging Face file listing failed: {exc}") from exc


def download_huggingface_file(
    repo_id: str,
    filename: str,
    *,
    token: str | None,
    destination: Path,
) -> Path:
    try:
        from huggingface_hub import hf_hub_download

        safe_name = normalize_dataset_filename(filename)
        destination.mkdir(parents=True, exist_ok=True)
        cached = hf_hub_download(repo_id, filename=filename, repo_type="dataset", token=token)
        output = destination / safe_name
        output.write_bytes(Path(cached).read_bytes())
        return output
    except ImportError as exc:
        raise ProviderError("Install integrations extra to use Hugging Face.") from exc
    except ProviderError:
        raise
    except Exception as exc:
        raise ProviderError(f"Hugging Face download failed: {exc}") from exc


def search_kaggle_datasets(query: str, *, limit: int = 20) -> list[str]:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        return [item.ref for item in api.dataset_list(search=query)[:limit]]
    except ImportError as exc:
        raise ProviderError("Install integrations extra to use Kaggle.") from exc
    except Exception as exc:
        raise ProviderError(f"Kaggle search failed: {exc}") from exc


def list_kaggle_files(reference: str, *, api: Any | None = None) -> list[str]:
    """List supported table files in a Kaggle dataset."""
    try:
        if api is None:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
        api.authenticate()
        response = api.dataset_list_files(reference)
        files = getattr(response, "files", response)
        names = [item.name for item in files]
        return sorted(name for name in names if name.lower().endswith(SUPPORTED_TABLE_SUFFIXES))
    except ImportError as exc:
        raise ProviderError("Install integrations extra to use Kaggle.") from exc
    except Exception as exc:
        raise ProviderError(f"Kaggle file listing failed: {exc}") from exc


def download_kaggle_dataset(reference: str, *, destination: Path) -> Path:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        destination.mkdir(parents=True, exist_ok=True)
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(reference, path=str(destination), unzip=True, quiet=True)
        return destination
    except ImportError as exc:
        raise ProviderError("Install integrations extra to use Kaggle.") from exc
    except Exception as exc:
        raise ProviderError(f"Kaggle download failed: {exc}") from exc


async def discover_via_mcp(endpoint: str, query: str) -> list[dict[str, Any]]:
    """Call allowlisted read-only search tool on a Streamable HTTP MCP server."""
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        async with (
            streamable_http_client(endpoint) as (read, write, _),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            tools = await session.list_tools()
            allowed = {"search_datasets", "dataset_search", "list_datasets"}
            tool = next((item for item in tools.tools if item.name in allowed), None)
            if tool is None:
                raise ProviderError("MCP server exposes no allowlisted dataset search tool.")
            result = await session.call_tool(tool.name, {"query": query})
            return [{"type": block.type, "content": str(block)} for block in result.content]
    except ImportError as exc:
        raise ProviderError("Install integrations extra to use MCP discovery.") from exc
    except ProviderError:
        raise
    except Exception as exc:
        raise ProviderError(f"MCP discovery failed: {exc}") from exc
