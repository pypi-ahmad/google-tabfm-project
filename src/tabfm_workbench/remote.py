"""Bounded HTTPS dataset retrieval."""

from collections.abc import Callable
from dataclasses import dataclass
from ipaddress import ip_address
from pathlib import PurePosixPath
from socket import getaddrinfo
from typing import Any
from urllib.parse import unquote, urlparse

import httpx


class RemoteFetchError(ValueError):
    """Raised when a remote source violates download policy."""


@dataclass(frozen=True)
class RemoteDataset:
    filename: str
    content: bytes


Resolver = Callable[[str, int], list[tuple[Any, ...]]]


def validate_remote_url(
    url: str,
    *,
    allow_insecure_http: bool = False,
    resolver: Resolver = getaddrinfo,
) -> None:
    """Reject unsafe schemes, credentials, and local/private destinations."""
    parsed = urlparse(url)
    allowed_schemes = {"https", "http"} if allow_insecure_http else {"https"}
    if parsed.scheme not in allowed_schemes:
        raise RemoteFetchError(
            "Only HTTPS dataset URLs are allowed unless HTTP is explicitly enabled."
        )
    if not parsed.hostname:
        raise RemoteFetchError("Dataset URL must include a hostname.")
    if parsed.username or parsed.password:
        raise RemoteFetchError("Credentials in dataset URLs are not allowed.")
    if parsed.hostname.lower() == "localhost":
        raise RemoteFetchError("Local network dataset URLs are not allowed.")
    try:
        address = ip_address(parsed.hostname)
    except ValueError:
        address = None
    if address is None:
        try:
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            resolved = resolver(parsed.hostname, port)
        except OSError as exc:
            raise RemoteFetchError("Dataset hostname could not be resolved.") from exc
        addresses = {ip_address(item[4][0]) for item in resolved if len(item) >= 5}
        if not addresses:
            raise RemoteFetchError("Dataset hostname did not resolve to an IP address.")
        if any(not item.is_global for item in addresses):
            raise RemoteFetchError("Private or local network dataset URLs are not allowed.")
        return
    if not address.is_global:
        raise RemoteFetchError("Private or local network dataset URLs are not allowed.")


def fetch_dataset(
    url: str,
    *,
    max_bytes: int = 500 * 1024 * 1024,
    allow_insecure_http: bool = False,
    client: httpx.Client | None = None,
) -> RemoteDataset:
    """Fetch dataset into memory with scheme and byte-count controls."""
    validate_remote_url(url, allow_insecure_http=allow_insecure_http)
    parsed = urlparse(url)
    filename = unquote(PurePosixPath(parsed.path).name) or "dataset.csv"
    owned_client = client is None
    active_client = client or httpx.Client(
        timeout=httpx.Timeout(30, read=120), follow_redirects=False
    )
    try:
        with active_client.stream("GET", url) as response:
            response.raise_for_status()
            declared = response.headers.get("content-length")
            if declared and int(declared) > max_bytes:
                raise RemoteFetchError("Remote dataset exceeds configured size limit.")
            content = bytearray()
            for chunk in response.iter_bytes():
                content.extend(chunk)
                if len(content) > max_bytes:
                    raise RemoteFetchError("Remote dataset exceeds configured size limit.")
        return RemoteDataset(filename=filename, content=bytes(content))
    except httpx.HTTPError as exc:
        raise RemoteFetchError(f"Remote dataset request failed: {exc}") from exc
    finally:
        if owned_client:
            active_client.close()
