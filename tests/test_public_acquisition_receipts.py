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
    AssetError,
    AssetNeed,
    AssetVault,
    FetchResult,
    PublicAcquisitionReceiptStore,
    PublicAcquisitionSourceV1,
    PublicFetchDenied,
    RemoteSource,
    acquire_need,
    digest_bytes,
    fetch_public_asset,
)
from dinkster_assets.fetch import HTTPSResponse

ASSET = b"public model bytes"
DIGEST = digest_bytes(ASSET)
LISTED_URL = "https://models.example/model.safetensors"
FINAL_URL = "https://cdn.example/model.safetensors"
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


def public_source() -> PublicAcquisitionSourceV1:
    return PublicAcquisitionSourceV1(
        digest=DIGEST,
        size_bytes=len(ASSET),
        source_type="declarative-resolver",
        source_id="resolver-id",
        source_revision="sha256:" + "1" * 64,
        listed_urls=(LISTED_URL,),
    )


def test_verified_public_https_redirect_records_exact_receipt(tmp_path: Path) -> None:
    opener = HTTPSFixture(
        {
            LISTED_URL: Response(302, headers={"Location": FINAL_URL}),
            FINAL_URL: Response(200, ASSET, {"Content-Length": str(len(ASSET))}),
        }
    )
    vault = AssetVault(tmp_path / "vault")
    result = no_network_public_fetch(DIGEST, LISTED_URL, len(ASSET), vault, opener=opener)
    store_path = tmp_path / "receipts.json"
    receipt = PublicAcquisitionReceiptStore(
        store_path, clock=lambda: 123.0, id_factory=lambda: "a" * 32
    ).record(public_source(), result)

    assert result.path.read_bytes() == ASSET
    assert result.public_https_verified
    assert (result.listed_url, result.final_url, result.size_bytes) == (
        LISTED_URL,
        FINAL_URL,
        len(ASSET),
    )
    assert receipt.to_wire() == {
        "version": 1,
        "receiptId": "a" * 32,
        "digest": DIGEST,
        "sizeBytes": len(ASSET),
        "sourceType": "declarative-resolver",
        "sourceId": "resolver-id",
        "sourceRevision": "sha256:" + "1" * 64,
        "listedUrl": LISTED_URL,
        "finalUrl": FINAL_URL,
        "fetchedAt": 123.0,
    }
    assert PublicAcquisitionReceiptStore(store_path).records() == (receipt,)
    assert [call[0] for call in opener.calls] == [LISTED_URL, FINAL_URL]


def test_separate_receipt_store_instances_preserve_each_append(tmp_path: Path) -> None:
    result = no_network_public_fetch(
        DIGEST,
        LISTED_URL,
        len(ASSET),
        AssetVault(tmp_path / "vault"),
        opener=HTTPSFixture({LISTED_URL: Response(200, ASSET)}),
    )
    path = tmp_path / "receipts.json"
    first = PublicAcquisitionReceiptStore(path, clock=lambda: 100.0, id_factory=lambda: "a" * 32)
    second = PublicAcquisitionReceiptStore(path, clock=lambda: 101.0, id_factory=lambda: "b" * 32)

    first.record(public_source(), result)
    second.record(public_source(), result)

    assert [receipt.receipt_id for receipt in second.records()] == ["a" * 32, "b" * 32]
    assert PublicAcquisitionReceiptStore(path).records() == second.records()


def test_acquisition_records_receipt_only_on_eligible_verified_public_path(tmp_path: Path) -> None:
    opener = HTTPSFixture({LISTED_URL: Response(200, ASSET)})
    vault = AssetVault(tmp_path / "vault")
    receipts = PublicAcquisitionReceiptStore(
        tmp_path / "receipts.json", clock=lambda: 123.0, id_factory=lambda: "a" * 32
    )
    with (
        patch.object(fetch_module, "_globally_routable_addresses", public_dns),
        patch.object(fetch_module, "_open_pinned_https", opener),
    ):
        result = acquire_need(
            AssetNeed("model", DIGEST, sources=(RemoteSource(LISTED_URL),)),
            vault,
            public_sources=(public_source(),),
            receipts=receipts,
        )

    assert result.status == "acquired"
    assert result.path == vault.resolve(DIGEST)
    assert len(receipts.records()) == 1


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


def test_actual_peer_must_match_a_globally_routable_dns_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed: list[bool] = []

    class ReboundConnection:
        def __init__(self, *_args: object) -> None:
            self.peer_verified = False

        def request(self, *_args: object, **_kwargs: object) -> None:
            return None

        def getresponse(self) -> Response:
            return Response(200, ASSET)

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(fetch_module, "_PinnedHTTPSConnection", ReboundConnection)
    monkeypatch.setattr(fetch_module, "_globally_routable_addresses", public_dns)
    with pytest.raises(PublicFetchDenied, match="unapproved peer"):
        fetch_public_asset(DIGEST, LISTED_URL, len(ASSET), AssetVault(tmp_path / "vault"))
    assert closed


@pytest.mark.parametrize(
    ("digest", "expected_size", "message"),
    [
        ("blake3:" + "0" * 64, len(ASSET), "did not verify"),
        (DIGEST, len(ASSET) + 1, "size does not match"),
    ],
)
def test_wrong_digest_or_size_creates_no_receipt(
    tmp_path: Path, digest: str, expected_size: int, message: str
) -> None:
    with pytest.raises(AssetError, match=message):
        no_network_public_fetch(
            digest,
            LISTED_URL,
            expected_size,
            AssetVault(tmp_path / "vault"),
            opener=HTTPSFixture({LISTED_URL: Response(200, ASSET)}),
        )
    assert PublicAcquisitionReceiptStore(tmp_path / "receipts.json").records() == ()


def test_preexisting_match_gets_no_receipt_and_denied_public_path_keeps_http_use(
    tmp_path: Path,
) -> None:
    source = public_source()
    receipts = PublicAcquisitionReceiptStore(tmp_path / "receipts.json")
    held_vault = AssetVault(tmp_path / "held")
    with held_vault.writer(DIGEST) as writer:
        writer.write(ASSET)
        writer.commit()
    held = acquire_need(
        AssetNeed("model", DIGEST, sources=(RemoteSource(LISTED_URL),)),
        held_vault,
        public_sources=(source,),
        receipts=receipts,
    )
    assert held.status == "held"
    assert receipts.records() == ()

    fallback_vault = AssetVault(tmp_path / "fallback")
    with (
        patch(
            "dinkster_assets.acquire.fetch_public_asset",
            side_effect=PublicFetchDenied("private DNS answer"),
        ),
        patch("dinkster_assets.fetch.urllib.request.urlopen", return_value=Response(200, ASSET)),
    ):
        acquired = acquire_need(
            AssetNeed("model", DIGEST, sources=(RemoteSource(LISTED_URL),)),
            fallback_vault,
            public_sources=(source,),
            receipts=receipts,
        )
    assert acquired.status == "acquired"
    assert fallback_vault.has(DIGEST)
    assert receipts.records() == ()
