"""Bounded HTTPS dataset retrieval."""

from collections.abc import Callable
from dataclasses import dataclass
from ipaddress import IPv4Address, IPv6Address, ip_address
from pathlib import PurePosixPath
from socket import getaddrinfo
from typing import Any
from urllib.parse import unquote, urlparse, urlunparse

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
) -> IPv4Address | IPv6Address:
    """Reject unsafe schemes, credentials, and local/private destinations.

    Returns the single globally-routable address that was actually validated,
    so the caller can connect to that exact address instead of re-resolving
    the hostname a second time (which would reopen a DNS-rebinding window
    between this check and the real request).
    """
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
        return min(addresses, key=str)
    if not address.is_global:
        raise RemoteFetchError("Private or local network dataset URLs are not allowed.")
    return address


def fetch_dataset(
    url: str,
    *,
    max_bytes: int = 500 * 1024 * 1024,
    allow_insecure_http: bool = False,
    client: httpx.Client | None = None,
    resolver: Resolver = getaddrinfo,
) -> RemoteDataset:
    """Fetch dataset into memory with scheme and byte-count controls."""
    pinned_address = validate_remote_url(
        url, allow_insecure_http=allow_insecure_http, resolver=resolver
    )
    parsed = urlparse(url)
    filename = unquote(PurePosixPath(parsed.path).name) or "dataset.csv"
    hostname = parsed.hostname or ""
    port_suffix = f":{parsed.port}" if parsed.port is not None else ""
    pinned_host = (
        f"[{pinned_address}]" if isinstance(pinned_address, IPv6Address) else str(pinned_address)
    )
    pinned_url = urlunparse(parsed._replace(netloc=f"{pinned_host}{port_suffix}"))
    owned_client = client is None
    active_client = client or httpx.Client(
        timeout=httpx.Timeout(30, read=120), follow_redirects=False
    )
    try:
        with active_client.stream(
            "GET",
            pinned_url,
            headers={"Host": hostname},
            extensions={"sni_hostname": hostname},
        ) as response:
            if 300 <= response.status_code < 400:
                raise RemoteFetchError(
                    f"Remote dataset request was redirected ({response.status_code}); "
                    "provide the direct file URL instead of one that redirects."
                )
            response.raise_for_status()
            declared = response.headers.get("content-length")
            declared_bytes: int | None
            try:
                declared_bytes = int(declared) if declared else None
            except ValueError:
                declared_bytes = None
            if declared_bytes is not None and declared_bytes > max_bytes:
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
