"""No-network coverage for public acquisition and P2P authorization policy."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import replace
from http.client import HTTPMessage
from pathlib import Path
from typing import cast
from unittest.mock import patch

import dinkster_assets.fetch as fetch_module
import pytest
from dinkster_assets import (
    P2P_GRANT_VERSION,
    P2P_REMOTE_GRANT_MAX_SECONDS,
    AssetError,
    AssetNeed,
    AssetVault,
    FetchResult,
    P2PDescriptorV1,
    P2PGrantReconciler,
    ProvenanceStore,
    PublicAcquisitionReceiptStore,
    PublicAcquisitionReceiptV1,
    PublicAcquisitionSourceV1,
    PublicFetchDenied,
    PublicSourceType,
    PublicSwarmDeclarationV1,
    RemoteSource,
    ResolverSubscriptionError,
    ResolverSubscriptionStore,
    acquire_need,
    derive_p2p_descriptor,
    digest_bytes,
    fetch_public_asset,
    parse_resolver_index,
)
from dinkster_assets.fetch import HTTPSResponse
from dinkster_assets.p2p_descriptor import canonical_p2p_info
from dinkster_assets.p2p_grants import public_swarm_grant_id

from dinkster.p2p_diagnostics import main as p2p_diagnostics_main

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
        return fetch_public_asset(
            digest,
            listed_url,
            expected_size,
            vault,
            headers=headers,
        )


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
    result = no_network_public_fetch(
        DIGEST,
        LISTED_URL,
        len(ASSET),
        vault,
        opener=opener,
    )
    store_path = tmp_path / "receipts.json"
    receipt = PublicAcquisitionReceiptStore(
        store_path,
        clock=lambda: 123.0,
        id_factory=lambda: "a" * 32,
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
    first = PublicAcquisitionReceiptStore(
        path,
        clock=lambda: 100.0,
        id_factory=lambda: "a" * 32,
    )
    second = PublicAcquisitionReceiptStore(
        path,
        clock=lambda: 101.0,
        id_factory=lambda: "b" * 32,
    )

    first.record(public_source(), result)
    second.record(public_source(), result)

    assert [receipt.receipt_id for receipt in second.records()] == ["a" * 32, "b" * 32]
    assert PublicAcquisitionReceiptStore(path).records() == second.records()


def test_acquisition_records_receipt_only_on_eligible_verified_public_path(
    tmp_path: Path,
) -> None:
    opener = HTTPSFixture({LISTED_URL: Response(200, ASSET)})
    vault = AssetVault(tmp_path / "vault")
    receipts = PublicAcquisitionReceiptStore(
        tmp_path / "receipts.json",
        clock=lambda: 123.0,
        id_factory=lambda: "a" * 32,
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
        (
            "https://user:secret@models.example/model",
            {},
            {},
            "without userinfo",
        ),
        (LISTED_URL, {"Authorization": "Bearer secret"}, {}, "credentials or cookies"),
        (LISTED_URL, {"Cookie": "session=secret"}, {}, "credentials or cookies"),
        (
            LISTED_URL,
            {"Proxy-Authorization": "Basic secret"},
            {},
            "credentials or cookies",
        ),
        (LISTED_URL, {"X-Api-Key": "secret"}, {}, "custom request headers"),
        (LISTED_URL, {}, {"Set-Cookie": "session=secret"}, "set a cookie"),
        (
            "https://models.example/model?token=secret",
            {},
            {},
            "without userinfo, query",
        ),
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
    downgrade = HTTPSFixture(
        {LISTED_URL: Response(302, headers={"Location": "http://cdn.example/model"})}
    )
    with pytest.raises(PublicFetchDenied, match="requires HTTPS"):
        no_network_public_fetch(
            DIGEST,
            LISTED_URL,
            len(ASSET),
            vault,
            opener=downgrade,
        )

    def private(_host: str, _port: int) -> tuple[str, ...]:
        return ("127.0.0.1",)

    with pytest.raises(PublicFetchDenied, match="non-global"):
        no_network_public_fetch(
            DIGEST,
            LISTED_URL,
            len(ASSET),
            vault,
            opener=HTTPSFixture({}),
            dns=private,
        )

    def mixed(_host: str, _port: int) -> tuple[str, ...]:
        return (PUBLIC_IP, "192.168.1.5")

    with pytest.raises(PublicFetchDenied, match="non-global"):
        no_network_public_fetch(
            DIGEST,
            LISTED_URL,
            len(ASSET),
            vault,
            opener=HTTPSFixture({}),
            dns=mixed,
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
        fetch_public_asset(
            DIGEST,
            LISTED_URL,
            len(ASSET),
            AssetVault(tmp_path / "vault"),
        )
    assert closed


@pytest.mark.parametrize(
    ("digest", "expected_size", "message"),
    [
        ("blake3:" + "0" * 64, len(ASSET), "did not verify"),
        (DIGEST, len(ASSET) + 1, "size does not match"),
    ],
)
def test_wrong_digest_or_size_creates_no_receipt(
    tmp_path: Path,
    digest: str,
    expected_size: int,
    message: str,
) -> None:
    opener = HTTPSFixture({LISTED_URL: Response(200, ASSET)})
    with pytest.raises(AssetError, match=message):
        no_network_public_fetch(
            digest,
            LISTED_URL,
            expected_size,
            AssetVault(tmp_path / "vault"),
            opener=opener,
        )
    store = PublicAcquisitionReceiptStore(tmp_path / "receipts.json")
    assert store.records() == ()


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
        patch(
            "dinkster_assets.fetch.urllib.request.urlopen",
            return_value=Response(200, ASSET, final_url=LISTED_URL),
        ),
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


def declaration(
    descriptor: P2PDescriptorV1,
    *,
    source_type: PublicSourceType = "declarative-resolver",
    source_id: str = "resolver-id",
    source_revision: str = "sha256:" + "1" * 64,
    license_name: str = "apache-2.0",
    refreshed_at: float = 100.0,
    expires_at: float = 100.0 + P2P_REMOTE_GRANT_MAX_SECONDS,
    evidence_id: str = "",
) -> PublicSwarmDeclarationV1:
    return PublicSwarmDeclarationV1(
        version=P2P_GRANT_VERSION,
        digest=DIGEST,
        size_bytes=len(ASSET),
        source_type=source_type,
        source_id=source_id,
        source_revision=source_revision,
        license=license_name,
        descriptor=descriptor,
        refreshed_at=refreshed_at,
        expires_at=expires_at,
        evidence_id=evidence_id,
        listed_urls=(LISTED_URL,) if source_type == "declarative-resolver" else (),
    )


def local_asset(tmp_path: Path) -> tuple[AssetVault, P2PDescriptorV1]:
    vault = AssetVault(tmp_path / "grant-vault")
    with vault.writer(DIGEST) as writer:
        writer.write(ASSET)
        path = writer.commit()
    return vault, derive_p2p_descriptor(path).descriptor


def receipt_for() -> PublicAcquisitionReceiptV1:
    return PublicAcquisitionReceiptV1(
        version=1,
        receipt_id="b" * 32,
        digest=DIGEST,
        size_bytes=len(ASSET),
        source_type="declarative-resolver",
        source_id="resolver-id",
        source_revision="sha256:" + "1" * 64,
        listed_url=LISTED_URL,
        final_url=FINAL_URL,
        fetched_at=101.0,
    )


def receipt_for_declaration(
    source: PublicSwarmDeclarationV1,
    *,
    receipt_id: str = "c" * 32,
    fetched_at: float = 101.0,
) -> PublicAcquisitionReceiptV1:
    return PublicAcquisitionReceiptV1(
        version=1,
        receipt_id=receipt_id,
        digest=source.digest,
        size_bytes=source.size_bytes,
        source_type=source.source_type,
        source_id=source.source_id,
        source_revision=source.source_revision,
        listed_url=source.listed_urls[0],
        final_url=source.listed_urls[0],
        fetched_at=fetched_at,
    )


def resolver_document(
    descriptor: P2PDescriptorV1,
    *,
    name: str = "model.safetensors",
    include_p2p: bool = True,
) -> bytes:
    entry: dict[str, object] = {
        "digest": DIGEST,
        "name": name,
        "urls": [LISTED_URL],
        "size": len(ASSET),
        "license": "Apache-2.0",
    }
    if include_p2p:
        entry["p2p"] = descriptor.to_wire()
    return json.dumps(
        {
            "dinksterResolver": 1,
            "name": "fixture-provider",
            "updated": "2026-09-01T00:00:00Z",
            "entries": [entry],
        }
    ).encode()


def test_four_source_types_normalize_and_seed_with_matching_complete_evidence(
    tmp_path: Path,
) -> None:
    vault, descriptor = local_asset(tmp_path)
    declarations = (
        declaration(
            descriptor,
            source_type="official-provider",
            source_id="provider/model",
            evidence_id="enumeration-42",
        ),
        declaration(descriptor),
        declaration(
            descriptor,
            source_type="code-resolver",
            source_id="resolver-code",
            evidence_id="consented-set-7",
        ),
        declaration(
            descriptor,
            source_type="manual",
            source_id="manual-model",
            evidence_id="attestation-9",
        ),
    )
    snapshot = (
        P2PGrantReconciler(clock=lambda: 102.0)
        .reconcile(
            declarations,
            (receipt_for(),),
            vault.resolve,
            enabled=True,
        )
        .snapshot
    )

    assert len(snapshot.public_grants) == 4
    assert {grant.source_type for grant in snapshot.seed_grants} == {
        "official-provider",
        "declarative-resolver",
        "code-resolver",
        "manual",
    }
    assert {grant.evidence_type for grant in snapshot.seed_grants} == {
        "provider-enumeration",
        "public-acquisition-receipt",
        "consented-set",
        "manual-attestation",
    }
    assert set(snapshot.public_grants[0].to_wire()) == {
        "version",
        "grantId",
        "digest",
        "sourceType",
        "sourceId",
        "sourceRevision",
        "license",
        "descriptor",
        "expiresAt",
    }
    assert set(snapshot.seed_grants[0].to_wire()) == {
        *snapshot.public_grants[0].to_wire(),
        "evidenceType",
        "evidenceId",
    }


def test_disabled_p2p_has_no_grants_and_enabled_download_is_staging_only(tmp_path: Path) -> None:
    vault, descriptor = local_asset(tmp_path)
    local_path = vault.resolve(DIGEST)
    assert local_path is not None
    local_path.unlink()
    grant_source = declaration(descriptor)
    reconciler = P2PGrantReconciler(clock=lambda: 102.0)

    assert (
        reconciler.reconcile(
            (grant_source,), (), vault.resolve, enabled=False
        ).snapshot.public_grants
        == ()
    )
    active = reconciler.reconcile((grant_source,), (), vault.resolve, enabled=True).snapshot
    assert len(active.staging_download_grants(DIGEST, already_local=False)) == 1
    assert active.staging_download_grants(DIGEST, already_local=True) == ()
    assert active.seed_grants == ()

    with vault.writer(DIGEST) as writer:
        writer.write(ASSET)
        writer.commit()
    complete = reconciler.reconcile((grant_source,), (), vault.resolve, enabled=True).snapshot
    assert complete.staging_download_grants(DIGEST, already_local=True) == ()
    assert complete.seed_grants == ()


def test_receipt_must_match_current_source_size_url_revision_and_time(tmp_path: Path) -> None:
    vault, descriptor = local_asset(tmp_path)
    source = declaration(descriptor)
    receipt = receipt_for()
    mismatches = (
        replace(receipt, size_bytes=len(ASSET) + 1),
        replace(receipt, source_revision="sha256:" + "2" * 64),
        replace(receipt, listed_url="https://other.example/model.safetensors"),
        replace(receipt, fetched_at=103.0),
    )
    for mismatch in mismatches:
        snapshot = (
            P2PGrantReconciler(clock=lambda: 102.0)
            .reconcile(
                (source,),
                (mismatch,),
                vault.resolve,
                enabled=True,
            )
            .snapshot
        )
        assert snapshot.public_grants
        assert snapshot.seed_grants == ()


def test_multiple_grants_compose_and_last_removal_or_consent_revokes(
    tmp_path: Path,
) -> None:
    vault, descriptor = local_asset(tmp_path)
    first = declaration(
        descriptor,
        source_type="manual",
        source_id="first",
        evidence_id="attestation-first",
    )
    second = declaration(
        descriptor,
        source_type="manual",
        source_id="second",
        evidence_id="attestation-second",
    )
    reconciler = P2PGrantReconciler(clock=lambda: 102.0)
    initial = reconciler.reconcile((first, second), (), vault.resolve, enabled=True)
    assert len(initial.snapshot.public_grants) == len(initial.snapshot.seed_grants) == 2

    one_left = reconciler.reconcile((second,), (), vault.resolve, enabled=True)
    assert len(one_left.revoked_public_grant_ids) == 1
    assert len(one_left.revoked_seed_grant_ids) == 1
    assert one_left.revoked_download_digests == frozenset()
    assert one_left.revoked_seed_digests == frozenset()

    last_removed = reconciler.reconcile((), (), vault.resolve, enabled=True)
    assert last_removed.revoked_download_digests == {DIGEST}
    assert last_removed.revoked_seed_digests == {DIGEST}

    reconciler.reconcile((first,), (), vault.resolve, enabled=True)
    withdrawn = reconciler.reconcile((first,), (), vault.resolve, enabled=False)
    assert withdrawn.snapshot.public_grants == withdrawn.snapshot.seed_grants == ()
    assert withdrawn.revoked_download_digests == {DIGEST}
    assert withdrawn.revoked_seed_digests == {DIGEST}


def test_remote_grant_expires_from_last_successful_refresh(tmp_path: Path) -> None:
    vault, descriptor = local_asset(tmp_path)
    now = [100.0]
    grant_source = declaration(
        descriptor,
        source_type="manual",
        source_id="expiring",
        evidence_id="attestation-expiring",
    )
    reconciler = P2PGrantReconciler(clock=lambda: now[0])
    assert reconciler.reconcile(
        (grant_source,), (), vault.resolve, enabled=True
    ).snapshot.seed_grants

    now[0] = 100.0 + P2P_REMOTE_GRANT_MAX_SECONDS
    expired = reconciler.reconcile((grant_source,), (), vault.resolve, enabled=True)
    assert expired.snapshot.public_grants == expired.snapshot.seed_grants == ()
    assert expired.revoked_download_digests == {DIGEST}
    assert expired.revoked_seed_digests == {DIGEST}

    with pytest.raises(AssetError, match="six hours"):
        declaration(
            descriptor,
            source_type="manual",
            source_id="too-long",
            evidence_id="attestation-too-long",
            expires_at=100.0 + P2P_REMOTE_GRANT_MAX_SECONDS + 1,
        )


@pytest.mark.parametrize("license_name", ["", "custom-model-license", "All rights reserved"])
@pytest.mark.parametrize("source_type", ["official-provider", "manual", "declarative-resolver"])
def test_unknown_or_conflicting_license_preserves_verified_seed_grants(
    tmp_path: Path, license_name: str, source_type: PublicSourceType
) -> None:
    vault, descriptor = local_asset(tmp_path)
    unknown = declaration(
        descriptor,
        source_type=source_type,
        source_id="unknown-license",
        license_name=license_name,
        evidence_id="" if source_type == "declarative-resolver" else "attestation-unknown",
    )
    reconciler = P2PGrantReconciler(clock=lambda: 102.0)
    receipts = (receipt_for_declaration(unknown),) if source_type == "declarative-resolver" else ()
    if source_type == "declarative-resolver":
        assert (
            reconciler.reconcile((unknown,), (), vault.resolve, enabled=True).snapshot.seed_grants
            == ()
        )
    snapshot = reconciler.reconcile((unknown,), receipts, vault.resolve, enabled=True).snapshot
    assert len(snapshot.public_grants) == len(snapshot.seed_grants) == 1
    assert snapshot.seed_grants[0].license == license_name
    assert snapshot.seed_grants[0].grant_id == public_swarm_grant_id(unknown)

    conflicting = declaration(
        descriptor,
        source_type=source_type,
        source_id="conflicting-license",
        license_name="mit",
        evidence_id="" if source_type == "declarative-resolver" else "attestation-conflict",
    )
    if source_type == "declarative-resolver":
        receipts += (receipt_for_declaration(conflicting, receipt_id="d" * 32),)
    snapshot = reconciler.reconcile(
        (unknown, conflicting),
        receipts,
        vault.resolve,
        enabled=True,
    ).snapshot
    assert len(snapshot.public_grants) == len(snapshot.seed_grants) == 2


@pytest.mark.parametrize("conflict", ["size", "file-root"])
def test_cross_provider_byte_identity_conflicts_revoke_both_grants(
    tmp_path: Path, conflict: str
) -> None:
    vault, descriptor = local_asset(tmp_path)
    first = declaration(
        descriptor, source_type="official-provider", source_id="first", evidence_id="first-set"
    )
    size = len(ASSET) + (conflict == "size")
    file_root = "c" * 64 if conflict == "file-root" else descriptor.file_root
    conflicting_descriptor = P2PDescriptorV1(
        "bittorrent-v2",
        hashlib.sha256(
            canonical_p2p_info(asset_digest=DIGEST, size=size, file_root=file_root)
        ).hexdigest(),
        file_root,
        descriptor.piece_length,
    )
    second = replace(first, source_id="second", size_bytes=size, descriptor=conflicting_descriptor)
    reconciler = P2PGrantReconciler(clock=lambda: 102.0)
    assert reconciler.reconcile((first,), (), vault.resolve, enabled=True).snapshot.seed_grants
    rejected = reconciler.reconcile((first, second), (), vault.resolve, enabled=True)
    assert rejected.snapshot.public_grants == rejected.snapshot.seed_grants == ()
    assert rejected.revoked_download_digests == rejected.revoked_seed_digests == {DIGEST}


def test_forged_complete_file_or_receipt_binding_does_not_seed(tmp_path: Path) -> None:
    vault, descriptor = local_asset(tmp_path)
    path = vault.resolve(DIGEST)
    assert path is not None
    path.write_bytes(b"mutated model bytes")
    source = declaration(descriptor)
    seed = (
        P2PGrantReconciler(clock=lambda: 102.0)
        .reconcile(
            (source,),
            (receipt_for(),),
            vault.resolve,
            enabled=True,
        )
        .snapshot.seed_grants
    )
    assert seed == ()

    unverified = FetchResult(path, DIGEST, LISTED_URL, FINAL_URL, len(ASSET))
    with pytest.raises(AssetError, match="does not prove"):
        PublicAcquisitionReceiptStore(tmp_path / "unverified.json").record(
            public_source(), unverified
        )

    verified = no_network_public_fetch(
        DIGEST,
        LISTED_URL,
        len(ASSET),
        AssetVault(tmp_path / "verified"),
        opener=HTTPSFixture({LISTED_URL: Response(200, ASSET)}),
    )
    with pytest.raises(AssetError, match="does not prove"):
        PublicAcquisitionReceiptStore(tmp_path / "altered.json").record(
            public_source(), replace(verified, final_url="https://other.example/model")
        )


def test_resolver_p2p_descriptor_is_bound_to_entry_identity_and_size(tmp_path: Path) -> None:
    _vault, descriptor = local_asset(tmp_path)
    parsed = parse_resolver_index(resolver_document(descriptor))
    assert parsed.entries[0].p2p == descriptor
    assert parse_resolver_index(json.dumps(parsed.to_wire())) == parsed

    missing_size = json.loads(resolver_document(descriptor))
    del missing_size["entries"][0]["size"]
    assert parse_resolver_index(json.dumps(missing_size)).entries[0].p2p is None

    null_descriptor = json.loads(resolver_document(descriptor))
    null_descriptor["entries"][0]["p2p"] = None
    assert parse_resolver_index(json.dumps(null_descriptor)).entries[0].p2p is None

    wrong_binding = json.loads(resolver_document(descriptor))
    wrong_binding["entries"][0]["size"] = len(ASSET) + 1
    assert parse_resolver_index(json.dumps(wrong_binding)).entries[0].p2p is None


def test_resolver_set_change_tombstone_and_unsubscribe_revoke_grants(
    tmp_path: Path,
) -> None:
    vault, descriptor = local_asset(tmp_path)
    now = [100.0]
    index_path = tmp_path / "resolver.json"
    index_path.write_bytes(resolver_document(descriptor))
    subscriptions = ResolverSubscriptionStore(
        tmp_path / "subscriptions.json",
        ProvenanceStore(tmp_path / "provenance.json"),
        clock=lambda: now[0],
    )
    subscription = subscriptions.subscribe(str(index_path))
    assert subscriptions.public_swarm_declarations() == ()
    subscriptions.set_p2p_trust(
        subscription.id,
        trusted_for_p2p=True,
        license_authoritative=True,
    )
    current = subscriptions.public_swarm_declarations()
    assert len(current) == 1
    assert current[0].source_type == "official-provider"
    reconciler = P2PGrantReconciler(clock=lambda: now[0])
    initial = reconciler.reconcile(
        current,
        (),
        vault.resolve,
        enabled=True,
    )
    assert initial.snapshot.seed_grants

    now[0] = 101.0
    index_path.write_bytes(resolver_document(descriptor, name="renamed.safetensors"))
    subscriptions.refresh(subscription.id)
    changed = subscriptions.public_swarm_declarations()
    set_change = reconciler.reconcile(changed, (), vault.resolve, enabled=True)
    assert set_change.revoked_public_grant_ids
    assert set_change.revoked_seed_grant_ids
    assert set_change.revoked_download_digests == frozenset()
    assert set_change.revoked_seed_digests == frozenset()
    assert set_change.snapshot.public_grants
    assert set_change.snapshot.seed_grants

    now[0] = 102.0
    index_path.write_bytes(
        resolver_document(descriptor, name="renamed.safetensors", include_p2p=False)
    )
    subscriptions.refresh(subscription.id)
    tombstone = reconciler.reconcile(
        subscriptions.public_swarm_declarations(), (), vault.resolve, enabled=True
    )
    assert tombstone.revoked_download_digests == {DIGEST}
    assert tombstone.revoked_seed_digests == {DIGEST}

    now[0] = 103.0
    index_path.write_bytes(resolver_document(descriptor))
    subscriptions.refresh(subscription.id)
    restored = subscriptions.public_swarm_declarations()
    reconciler.reconcile(restored, (), vault.resolve, enabled=True)
    assert subscriptions.unsubscribe(subscription.id)
    unsubscribed = reconciler.reconcile(
        subscriptions.public_swarm_declarations(), (), vault.resolve, enabled=True
    )
    assert unsubscribed.revoked_download_digests == {DIGEST}
    assert unsubscribed.revoked_seed_digests == {DIGEST}


def test_failed_hosted_refresh_keeps_provenance_but_expires_p2p_trust(
    tmp_path: Path,
) -> None:
    vault, descriptor = local_asset(tmp_path)
    now = [100.0]
    provenance = ProvenanceStore(tmp_path / "provenance.json")
    subscriptions = ResolverSubscriptionStore(
        tmp_path / "subscriptions.json",
        provenance,
        clock=lambda: now[0],
    )
    with patch(
        "dinkster_assets.resolver_subscription._fetch_url",
        return_value=(resolver_document(descriptor), '"etag-1"'),
    ):
        subscription = subscriptions.subscribe("https://indexes.example/models.json")
    subscriptions.set_p2p_trust(
        subscription.id,
        trusted_for_p2p=True,
        license_authoritative=True,
    )
    declarations = subscriptions.public_swarm_declarations()
    assert declarations[0].refreshed_at == 100.0
    assert declarations[0].expires_at == 100.0 + P2P_REMOTE_GRANT_MAX_SECONDS
    reconciler = P2PGrantReconciler(clock=lambda: now[0])
    assert reconciler.reconcile(
        declarations,
        (),
        vault.resolve,
        enabled=True,
    ).snapshot.seed_grants

    now[0] = 100.0 + P2P_REMOTE_GRANT_MAX_SECONDS
    with patch(
        "dinkster_assets.resolver_subscription._fetch_url",
        side_effect=ResolverSubscriptionError("offline"),
    ):
        subscriptions.refresh(subscription.id)
    stale = subscriptions.subscriptions()[0]
    assert stale.checked_at == now[0]
    assert stale.refreshed_at == 100.0
    assert stale.error == "offline"
    assert provenance.sources(DIGEST) == (LISTED_URL,)
    assert subscriptions.public_sources(DIGEST)
    expired = reconciler.reconcile(
        subscriptions.public_swarm_declarations(),
        (),
        vault.resolve,
        enabled=True,
    )
    assert expired.snapshot.public_grants == expired.snapshot.seed_grants == ()
    assert expired.revoked_download_digests == {DIGEST}
    assert expired.revoked_seed_digests == {DIGEST}


def test_failed_local_refresh_keeps_provenance_but_expires_p2p_trust(
    tmp_path: Path,
) -> None:
    vault, descriptor = local_asset(tmp_path)
    now = [100.0]
    index_path = tmp_path / "resolver.json"
    index_path.write_bytes(resolver_document(descriptor))
    provenance = ProvenanceStore(tmp_path / "provenance.json")
    subscriptions = ResolverSubscriptionStore(
        tmp_path / "subscriptions.json",
        provenance,
        clock=lambda: now[0],
    )
    subscription = subscriptions.subscribe(str(index_path))
    subscriptions.set_p2p_trust(
        subscription.id,
        trusted_for_p2p=True,
        license_authoritative=True,
    )
    declarations = subscriptions.public_swarm_declarations()
    assert declarations[0].refreshed_at == 100.0
    reconciler = P2PGrantReconciler(clock=lambda: now[0])
    assert reconciler.reconcile(
        declarations,
        (),
        vault.resolve,
        enabled=True,
    ).snapshot.seed_grants

    now[0] = 100.0 + P2P_REMOTE_GRANT_MAX_SECONDS
    index_path.unlink()
    subscriptions.refresh(subscription.id)
    stale = subscriptions.subscriptions()[0]
    assert stale.checked_at == now[0]
    assert stale.refreshed_at == 100.0
    assert stale.error
    assert provenance.sources(DIGEST) == (LISTED_URL,)
    expired = reconciler.reconcile(
        subscriptions.public_swarm_declarations(),
        (),
        vault.resolve,
        enabled=True,
    )
    assert expired.snapshot.public_grants == expired.snapshot.seed_grants == ()
    assert expired.revoked_download_digests == {DIGEST}
    assert expired.revoked_seed_digests == {DIGEST}


def test_p2p_diagnostics_is_no_network_and_does_not_persist_opt_in(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "library"
    root.mkdir()
    now = time.time()
    source_file = tmp_path / "source.bin"
    source_file.write_bytes(ASSET)
    descriptor = derive_p2p_descriptor(source_file).descriptor
    index_path = tmp_path / "resolver.json"
    index_path.write_bytes(resolver_document(descriptor))
    subscriptions = ResolverSubscriptionStore(
        root / "resolver-indexes.json",
        ProvenanceStore(root / "provenance.json"),
        clock=lambda: now,
    )
    subscription = subscriptions.subscribe(str(index_path))
    subscriptions.set_p2p_trust(
        subscription.id,
        trusted_for_p2p=True,
        license_authoritative=True,
    )
    public_source = subscriptions.public_sources()[0]
    fetch_result = no_network_public_fetch(
        DIGEST,
        LISTED_URL,
        len(ASSET),
        AssetVault(root / "vault"),
        opener=HTTPSFixture({LISTED_URL: Response(200, ASSET)}),
    )
    PublicAcquisitionReceiptStore(
        root / "public-acquisition-receipts.json",
        clock=lambda: now,
        id_factory=lambda: "d" * 32,
    ).record(public_source, fetch_result)
    before = {
        path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }

    with (
        patch(
            "dinkster_assets.resolver_subscription._fetch_url",
            side_effect=AssertionError("diagnostics attempted network I/O"),
        ),
        patch.object(
            fetch_module,
            "_globally_routable_addresses",
            side_effect=AssertionError("diagnostics attempted DNS"),
        ),
    ):
        assert p2p_diagnostics_main(["--library-root", str(root), "--json"]) == 0
        disabled = json.loads(capsys.readouterr().out)
        assert disabled["enabled"] is False
        assert len(disabled["receipts"]) == 1
        assert disabled["publicGrants"] == disabled["seedGrants"] == []

        assert p2p_diagnostics_main(["--library-root", str(root), "--enabled", "--json"]) == 0
        enabled = json.loads(capsys.readouterr().out)
        assert enabled["enabled"] is True
        assert len(enabled["publicGrants"]) == len(enabled["seedGrants"]) == 1

    after = {
        path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }
    assert after == before
