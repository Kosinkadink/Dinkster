"""No-network coverage for public acquisition receipts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from http.client import HTTPMessage
from pathlib import Path
from typing import cast
from unittest.mock import patch

import dinkster_assets.fetch as fetch_module
import pytest
from dinkster_assets import (
    AssetVault,
    FetchResult,
    PublicFetchDenied,
    digest_bytes,
    fetch_public_asset,
)
from dinkster_assets.fetch import HTTPSResponse

ASSET = b"public model bytes"
DIGEST = digest_bytes(ASSET)
LISTED_URL = "https://models.example/model.safetensors"
PUBLIC_IP = "93.184.216.34"


class Response:
    def __init__(
        self,
        status: int,
        data: bytes = b"",
        headers: dict[str, str] | None = None,
        *,
        final_url: str = LISTED_URL,
    ) -> None:
        self.status = status
        self.headers = HTTPMessage()
        for name, value in (headers or {}).items():
            self.headers[name] = value
        self._chunks = iter((data, b""))
        self._final_url = final_url

    def read(self, _amount: int | None = None) -> bytes:
        return next(self._chunks)

    def geturl(self) -> str:
        return self._final_url

    def __enter__(self) -> Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class HTTPSFixture:
    def __init__(self, responses: dict[str, Response]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, tuple[str, ...], dict[str, str]]] = []

    def __call__(
        self,
        url: str,
        addresses: Sequence[str],
        headers: Mapping[str, str],
        _timeout: float,
    ) -> AbstractContextManager[HTTPSResponse]:
        self.calls.append((url, tuple(addresses), dict(headers)))
        return cast("AbstractContextManager[HTTPSResponse]", nullcontext(self.responses[url]))


def public_dns(_host: str, _port: int) -> tuple[str, ...]:
    return (PUBLIC_IP,)


def no_network_public_fetch(
    digest: str,
    listed_url: str,
    expected_size: int,
    vault: AssetVault,
    *,
    opener: HTTPSFixture,
    dns: Callable[[str, int], Sequence[str]] = public_dns,
    headers: Mapping[str, str] | None = None,
) -> FetchResult:
    with (
        patch.object(fetch_module, "_globally_routable_addresses", dns),
        patch.object(fetch_module, "_open_pinned_https", opener),
    ):
        return fetch_public_asset(digest, listed_url, expected_size, vault, headers=headers)


@pytest.mark.parametrize(
    ("listed_url", "headers", "response_headers", "message"),
    [
        ("https://user:secret@models.example/model", {}, {}, "without userinfo"),
        (LISTED_URL, {"Authorization": "Bearer secret"}, {}, "credentials or cookies"),
        (LISTED_URL, {"Cookie": "session=secret"}, {}, "credentials or cookies"),
        (LISTED_URL, {"Proxy-Authorization": "Basic secret"}, {}, "credentials or cookies"),
        (LISTED_URL, {"X-Api-Key": "secret"}, {}, "custom request headers"),
        (LISTED_URL, {}, {"Set-Cookie": "session=secret"}, "set a cookie"),
        ("https://models.example/model?token=secret", {}, {}, "without userinfo, query"),
    ],
)
def test_credentials_and_cookies_deny_public_fetch(
    tmp_path: Path,
    listed_url: str,
    headers: dict[str, str],
    response_headers: dict[str, str],
    message: str,
) -> None:
    opener = HTTPSFixture(
        {listed_url: Response(200, ASSET, {**response_headers, "Content-Length": str(len(ASSET))})}
    )
    with pytest.raises(PublicFetchDenied, match=message):
        no_network_public_fetch(
            DIGEST,
            listed_url,
            len(ASSET),
            AssetVault(tmp_path / "vault"),
            opener=opener,
            headers=headers,
        )


def test_http_redirect_private_dns_and_rebinding_deny_public_fetch(tmp_path: Path) -> None:
    vault = AssetVault(tmp_path / "vault")
    with pytest.raises(PublicFetchDenied, match="requires HTTPS"):
        no_network_public_fetch(
            DIGEST,
            LISTED_URL,
            len(ASSET),
            vault,
            opener=HTTPSFixture(
                {LISTED_URL: Response(302, headers={"Location": "http://cdn.example/model"})}
            ),
        )

    def private(_host: str, _port: int) -> tuple[str, ...]:
        return ("127.0.0.1",)

    with pytest.raises(PublicFetchDenied, match="non-global"):
        no_network_public_fetch(
            DIGEST, LISTED_URL, len(ASSET), vault, opener=HTTPSFixture({}), dns=private
        )

    responses = {
        LISTED_URL: Response(302, headers={"Location": LISTED_URL + ".redirected"}),
        LISTED_URL + ".redirected": Response(200, ASSET),
    }
    answers = iter(((PUBLIC_IP,), ("10.0.0.2",)))
    with pytest.raises(PublicFetchDenied, match="non-global"):
        no_network_public_fetch(
            DIGEST,
            LISTED_URL,
            len(ASSET),
            vault,
            opener=HTTPSFixture(responses),
            dns=lambda _host, _port: next(answers),
        )
    assert vault.resolve(DIGEST) is None
