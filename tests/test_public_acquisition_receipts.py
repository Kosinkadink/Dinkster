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


@pytest.mark.parametrize("provider", ["huggingface", "civitai"])
@pytest.mark.parametrize("foreign_start", [False, True])
def test_model_credentials_stay_with_provider_and_do_not_return_after_cdn_redirect(
    provider: str,
    foreign_start: bool,
) -> None:
    origin = "huggingface.co" if provider == "huggingface" else "civitai.com"
    foreign = "civitai.com" if provider == "huggingface" else "huggingface.co"
    initial = f"https://{foreign if foreign_start else origin}/model/file"
    same_origin = f"https://{origin}/model/pinned/file"
    cdn = "https://cdn.example/file?signature=synthetic"
    back = f"https://{origin}/model/pinned/returned"
    opener = HTTPSFixture(
        {
            initial: Response(302, headers={"Location": same_origin}),
            same_origin: Response(302, headers={"Location": cdn}),
            cdn: Response(302, headers={"Location": back}),
            back: Response(200, ASSET),
        }
    )
    with (
        patch.object(fetch_module, "_globally_routable_addresses", public_dns),
        patch.object(fetch_module, "_open_pinned_https", opener),
        fetch_module.open_model_response(
            initial,
            huggingface_token="synthetic-canary" if provider == "huggingface" else None,
            civitai_token="synthetic-canary" if provider == "civitai" else None,
        ) as response,
    ):
        assert response.read() == ASSET
    assert [call[2].get("Authorization") for call in opener.calls] == (
        [None, None, None, None]
        if foreign_start
        else ["Bearer synthetic-canary", "Bearer synthetic-canary", None, None]
    )


@pytest.mark.parametrize("provider", ["huggingface", "civitai"])
def test_provider_form_exchange_never_redirects_or_retries_ambiguous_post(
    provider: str, caplog: pytest.LogCaptureFixture
) -> None:
    from urllib.parse import parse_qs

    from dinkster_assets import AssetError

    caplog.set_level("INFO", logger=fetch_module.__name__)
    requests: list[tuple[str, bytes]] = []
    opened: list[str] = []
    status = 200
    broken_body = False

    class ProviderResponse(Response):
        def close(self) -> None:
            pass

        def read(self, amount: int | None = None) -> bytes:
            if broken_body:
                from http.client import IncompleteRead

                raise IncompleteRead(b"synthetic-private-response")
            return super().read(amount)

    class Connection:
        peer_verified = True

        def __init__(self, _host: str, _port: int, address: str, _timeout: float) -> None:
            assert _host == ("huggingface.co" if provider == "huggingface" else "auth.civitai.com")
            opened.append(address)

        def request(self, method: str, path: str, *, body: bytes, headers: dict[str, str]) -> None:
            assert method == "POST"
            assert path == (
                "/oauth/token" if provider == "huggingface" else "/api/auth/oauth/token"
            )
            assert headers["Content-Type"] == "application/x-www-form-urlencoded"
            requests.append((path, body))
            if status == 0:
                raise OSError("synthetic ambiguous exchange")

        def getresponse(self) -> ProviderResponse:
            return ProviderResponse(
                status,
                b'{"access_token":"synthetic-secret"}',
                {"Location": "https://elsewhere.example/token"},
            )

        def close(self) -> None:
            pass

    with (
        patch.object(
            fetch_module, "_globally_routable_addresses", lambda *_args: (PUBLIC_IP, "1.1.1.1")
        ),
        patch.object(fetch_module, "_PinnedHTTPSConnection", Connection),
    ):
        result = fetch_module.post_provider_form(
            "token", {"code": "a+b&c", "code_verifier": "synthetic-verifier"}, provider=provider
        )
        assert result == {"access_token": "synthetic-secret"}
        assert parse_qs(requests[-1][1].decode()) == {
            "code": ["a+b&c"],
            "code_verifier": ["synthetic-verifier"],
        }
        status = 302
        with pytest.raises(AssetError, match="exchange unavailable"):
            fetch_module.post_provider_form("token", {"code": "synthetic-code"}, provider=provider)
        assert len(opened) == 2
        status = 0
        with pytest.raises(PublicFetchDenied, match="outcome is unknown"):
            fetch_module.post_provider_form("token", {"code": "synthetic-code"}, provider=provider)
        assert len(opened) == 3
        status = 200
        broken_body = True
        with pytest.raises(AssetError, match="response interrupted") as refused:
            fetch_module.post_provider_form("token", {"code": "synthetic-code"}, provider=provider)
        assert "synthetic-private-response" not in str(refused.value)
        assert len(opened) == 4
    assert f"provider={provider} endpoint=token status=200" in caplog.text
    assert f"provider={provider} endpoint=token status=302" in caplog.text
    for private in ("a+b&c", "synthetic-verifier", "synthetic-code", "synthetic-secret"):
        assert private not in caplog.text


@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.parametrize("provider", ["huggingface", "civitai"])
def test_model_provider_refusal_retains_status_without_private_response(
    status: int, provider: str
) -> None:
    origin = "huggingface.co" if provider == "huggingface" else "civitai.com"
    url = f"https://{origin}/model/file"
    opener = HTTPSFixture({url: Response(status, b"private-response-canary")})
    with (
        patch.object(fetch_module, "_globally_routable_addresses", public_dns),
        patch.object(fetch_module, "_open_pinned_https", opener),
        pytest.raises(fetch_module.ModelAccessDenied) as refused,
        fetch_module.open_model_response(
            url,
            huggingface_token="synthetic-canary" if provider == "huggingface" else None,
            civitai_token="synthetic-canary" if provider == "civitai" else None,
        ),
    ):
        pytest.fail("refused provider bytes must not reach a writer")
    assert refused.value.status == status
    assert refused.value.provider == provider
    assert refused.value.authenticated
    assert "canary" not in str(refused.value)
