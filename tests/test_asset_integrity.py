"""Descriptor-bound asset reads preserve ingest-time content identity."""

from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path
from typing import BinaryIO, cast

import aiohttp
import pytest
from aiohttp.test_utils import TestClient, TestServer, make_mocked_request
from dinkster_assets import (
    AssetError,
    AssetIntegrityError,
    AssetVault,
    LibraryStore,
    LocalAssetLibrary,
    clear_verified_cache,
    digest_bytes,
    open_verified,
    verified_local_path,
)
from dinkster_assets import integrity as integrity_module
from dinkster_assets import vault as vault_module
from dinkster_server import ServerLibrary, create_app
from dinkster_server import asset_stream as asset_stream_module
from dinkster_server.asset_stream import open_verified_sized, stream_verified
from test_server import SCHEMAS, make_engine


@pytest.fixture(autouse=True)
def _fresh_verification_cache() -> None:
    """Each test proves its own hashing behavior; no cross-test trust."""
    clear_verified_cache()


def write_asset(tmp_path: Path, data: bytes, name: str = "model.safetensors") -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


@pytest.mark.parametrize("result", ["stable", "changing", "corrupt"])
def test_concurrent_vault_publication_requires_stable_matching_reverification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, result: str
) -> None:
    vault = AssetVault(tmp_path)
    data = b"verified image document"
    digest = digest_bytes(data)
    with vault.writer(digest) as writer:
        writer.write(data)
        target, created = writer.commit_with_result()
    assert created
    attempts = 0
    original = integrity_module.digest_file_with_record

    def verify(path: Path) -> tuple[str, integrity_module.AssetVerificationRecord | None]:
        nonlocal attempts
        attempts += 1
        if attempts == 1 or result == "changing":
            raise AssetError("asset changed while being ingested")
        if result == "corrupt":
            path.write_bytes(b"corrupt image document")
        return original(path)

    monkeypatch.setattr(vault_module, "digest_file_with_record", verify)
    with vault.writer(digest) as writer:
        writer.write(data)
        if result == "stable":
            assert writer.commit_with_result() == (target, False)
        else:
            with pytest.raises(AssetError):
                writer.commit_with_result()
    assert attempts == 2
    assert target.exists()
    assert not list(tmp_path.glob(".ingest-*"))


# --- the stale-index attack the (size, mtime) heuristic cannot see -----------


@pytest.mark.skipif(
    os.name != "posix",
    reason="requires POSIX ctime as a content-change stamp",
)
def test_same_size_same_mtime_replacement_is_rejected_at_read(tmp_path: Path) -> None:
    """The library heuristic is deliberately fooled (same size, restored
    mtime) and still resolves the path - but every byte-consumption method
    refuses to serve the impostor bytes."""
    original = b"tiny checkpoint bytes"
    root = tmp_path / "models"
    root.mkdir()
    path = root / "tiny.safetensors"
    path.write_bytes(original)
    library = LocalAssetLibrary(root)
    library.scan()
    ref = library.ref("models/tiny.safetensors")
    assert ref.digest == digest_bytes(original)

    impostor = b"X" * len(original)  # equal size, different content
    stat = path.stat()
    path.write_bytes(impostor)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    # The discovery heuristic passes: resolve still names the path.
    assert library.resolve(ref.digest) == path

    # Consumption does not: every read boundary raises, none returns bytes.
    with pytest.raises(AssetIntegrityError) as excinfo:
        ref.open()
    assert excinfo.value.reason == "stale_ingest_record"
    assert excinfo.value.expected_digest == ref.digest
    assert excinfo.value.actual_digest is None
    with pytest.raises(AssetIntegrityError):
        ref.read_bytes()
    with pytest.raises(AssetIntegrityError):
        ref.local_path()


def test_replacement_between_resolve_and_open_is_rejected(tmp_path: Path) -> None:
    """resolve() answered honestly, then the path was rebound before
    open(): the descriptor-time hash catches it."""
    original = b"the bytes the digest names"
    path = write_asset(tmp_path, original)
    digest = digest_bytes(original)

    replacement = write_asset(tmp_path, b"swapped in afterwards", "swap.tmp")
    os.replace(replacement, path)

    with pytest.raises(AssetIntegrityError) as excinfo:
        open_verified(path, digest)
    assert excinfo.value.reason == "digest_mismatch"


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX open-file replacement semantics")
def test_open_verified_handle_pins_the_verified_inode(tmp_path: Path) -> None:
    """Rename-replacement AFTER verification cannot swap the bytes: the
    returned handle and the proof share one descriptor."""
    original = b"verified content"
    path = write_asset(tmp_path, original)
    handle = open_verified(path, digest_bytes(original))
    try:
        replacement = write_asset(tmp_path, b"late replacement", "late.tmp")
        os.replace(replacement, path)
        assert handle.read() == original  # still the verified inode
    finally:
        handle.close()
    assert path.read_bytes() == b"late replacement"  # the name did move on


def test_read_bytes_round_trips_verified_content(tmp_path: Path) -> None:
    data = b"honest bytes"
    path = write_asset(tmp_path, data)
    with open_verified(path, digest_bytes(data)) as handle:
        assert handle.tell() == 0
        assert handle.read() == data


def test_ingest_record_skips_payload_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = b"verified during ingest"
    path = write_asset(tmp_path, data)
    digest = digest_bytes(data)
    record = integrity_module.verification_record(digest, path.stat())
    assert record is not None

    calls = _counting_hasher(monkeypatch)
    with open_verified(path, digest, record) as handle:
        assert handle.read() == data
    assert calls == [0]


def test_stale_ingest_record_fails_without_rehash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = b"verified during ingest"
    path = write_asset(tmp_path, data)
    digest = digest_bytes(data)
    record = integrity_module.verification_record(digest, path.stat())
    assert record is not None
    path.write_bytes(b"changed after ingest with a different size")

    calls = _counting_hasher(monkeypatch)
    with pytest.raises(AssetIntegrityError) as excinfo:
        open_verified(path, digest, record)
    assert excinfo.value.reason == "stale_ingest_record"
    assert calls == [0]


# --- the verification cache: speeds up honest reopens, never authorizes ------


def _counting_hasher(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls = [0]
    real = integrity_module._hash_handle  # noqa: SLF001 - designated test seam

    def counted(handle: BinaryIO) -> str:
        calls[0] += 1
        return real(handle)

    monkeypatch.setattr(integrity_module, "_hash_handle", counted)
    return calls


@pytest.mark.skipif(os.name != "posix", reason="verified fingerprint cache is POSIX-only")
def test_cache_skips_rehash_until_the_file_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _counting_hasher(monkeypatch)
    data = b"stable content"
    path = write_asset(tmp_path, data)
    digest = digest_bytes(data)

    with open_verified(path, digest) as handle:
        assert handle.read() == data
    with open_verified(path, digest) as handle:  # cache hit: no second hash
        assert handle.read() == data
    assert calls[0] == 1

    replacement = write_asset(tmp_path, b"fresh content!", "next.tmp")
    os.replace(replacement, path)  # new inode -> new fingerprint
    with pytest.raises(AssetIntegrityError):
        open_verified(path, digest)
    assert calls[0] == 2  # the mutation forced a rehash


def test_cache_entry_never_authorizes_a_different_digest(tmp_path: Path) -> None:
    """The cache records what the content IS, not what a caller asked for:
    a hit for the honest digest must still refuse a different expectation."""
    data = b"one identity"
    path = write_asset(tmp_path, data)
    open_verified(path, digest_bytes(data)).close()  # populate the cache

    with pytest.raises(AssetIntegrityError) as excinfo:
        open_verified(path, digest_bytes(b"a different identity"))
    assert excinfo.value.reason == "digest_mismatch"
    assert excinfo.value.actual_digest == digest_bytes(data)


def test_in_place_rewrite_with_restored_mtime_forces_rehash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same inode, same size, same mtime - only the POSIX change stamp
    (ctime) moves. The cached fingerprint must miss and the rehash must
    reject; this is exactly the rewrite the (size, mtime) heuristic and a
    creation-time st_ctime could not see."""
    calls = _counting_hasher(monkeypatch)
    data = b"cache me if you can"
    path = write_asset(tmp_path, data)
    digest = digest_bytes(data)
    open_verified(path, digest).close()
    assert calls[0] == 1

    stat = path.stat()
    with path.open("r+b") as writer:
        writer.write(b"Y" * len(data))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(AssetIntegrityError):
        open_verified(path, digest)
    assert calls[0] == 2  # cache miss: the rewrite was rehashed, not trusted


def test_cache_disabled_where_fingerprints_are_untrusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On platforms where st_ctime is creation time (Windows), a matching
    fingerprint proves nothing: every open rehashes."""
    calls = _counting_hasher(monkeypatch)
    monkeypatch.setattr(integrity_module, "_TRUST_FINGERPRINTS", False)
    data = b"no shortcuts here"
    path = write_asset(tmp_path, data)
    digest = digest_bytes(data)
    open_verified(path, digest).close()
    open_verified(path, digest).close()
    assert calls[0] == 2
    assert len(integrity_module._verified) == 0  # noqa: SLF001 - cache introspection


@pytest.mark.skipif(os.name != "posix", reason="verified fingerprint cache is POSIX-only")
def test_cache_is_bounded_lru(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(integrity_module, "_CACHE_LIMIT", 2)
    for index in range(4):
        data = f"asset number {index}".encode()
        path = write_asset(tmp_path, data, f"asset{index}.bin")
        open_verified(path, digest_bytes(data)).close()
    assert len(integrity_module._verified) == 2  # noqa: SLF001 - cache introspection


def test_mutation_during_hashing_is_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A write landing on the inode mid-hash invalidates the proof: the
    post-hash fingerprint no longer matches the pre-hash one."""
    data = b"about to be appended to"
    path = write_asset(tmp_path, data)
    real = integrity_module._hash_handle  # noqa: SLF001 - designated test seam

    def hash_then_mutate(handle: BinaryIO) -> str:
        result = real(handle)
        with path.open("ab") as writer:
            writer.write(b" oops")
        return result

    monkeypatch.setattr(integrity_module, "_hash_handle", hash_then_mutate)
    with pytest.raises(AssetIntegrityError) as excinfo:
        open_verified(path, digest_bytes(data))
    assert excinfo.value.reason == "changed_during_verification"
    assert excinfo.value.actual_digest is None


# --- verified_local_path: the weaker, path-shaped boundary --------------------


def test_verified_local_path_returns_path_for_honest_content(tmp_path: Path) -> None:
    data = b"checkpoint bytes"
    path = write_asset(tmp_path, data)
    assert verified_local_path(path, digest_bytes(data)) == path


@pytest.mark.skipif(os.name != "nt", reason="requires Windows fingerprint semantics")
def test_windows_path_binding_ignores_creation_time() -> None:
    fingerprint = (1, 2, 3, 4, 5)
    assert integrity_module._path_binding_fingerprint(fingerprint) == (1, 2, 3, 4)  # noqa: SLF001


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX open-file replacement semantics")
def test_verified_local_path_rejects_rebinding_after_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The name is re-checked against the verified descriptor: a rename
    landing between the content proof and the return is a refusal, not a
    stale answer. (A rename DURING hashing is caught even earlier - the
    old inode's ctime bump fails the mutation check.)"""
    data = b"original path target"
    path = write_asset(tmp_path, data)
    real = integrity_module.open_verified

    def verify_then_rebind(target: Path, digest: str) -> BinaryIO:
        handle = real(target, digest)
        os.replace(write_asset(tmp_path, data + b"!", "rebind.tmp"), path)
        return handle

    monkeypatch.setattr(integrity_module, "open_verified", verify_then_rebind)
    with pytest.raises(AssetIntegrityError) as excinfo:
        verified_local_path(path, digest_bytes(data))
    assert excinfo.value.reason == "path_rebound"


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX open-file unlink semantics")
def test_verified_local_path_rejects_deletion_after_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = b"soon to vanish"
    path = write_asset(tmp_path, data)
    real = integrity_module.open_verified

    def verify_then_unlink(target: Path, digest: str) -> BinaryIO:
        handle = real(target, digest)
        path.unlink()
        return handle

    monkeypatch.setattr(integrity_module, "open_verified", verify_then_unlink)
    with pytest.raises(AssetIntegrityError) as excinfo:
        verified_local_path(path, digest_bytes(data))
    assert excinfo.value.reason == "path_rebound"


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX open-file replacement semantics")
def test_verified_local_path_rejects_rename_during_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rename landing mid-hash is also refused - via the mutation check,
    since dropping the old inode's link count bumps its ctime."""
    data = b"renamed away mid-hash"
    path = write_asset(tmp_path, data)
    real = integrity_module._hash_handle  # noqa: SLF001 - designated test seam

    def hash_then_rebind(handle: BinaryIO) -> str:
        result = real(handle)
        os.replace(write_asset(tmp_path, data + b"!", "rebind.tmp"), path)
        return result

    monkeypatch.setattr(integrity_module, "_hash_handle", hash_then_rebind)
    with pytest.raises(AssetIntegrityError) as excinfo:
        verified_local_path(path, digest_bytes(data))
    assert excinfo.value.reason == "changed_during_verification"


# --- the serving boundary: /assets/{digest} never mirrors wrong bytes --------


def test_vault_ingest_persists_verification_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = b"streamed and hashed once"
    digest = digest_bytes(data)
    vault = AssetVault(tmp_path / "vault")
    with vault.writer(digest) as writer:
        writer.write(data)
        writer.commit()

    resolution = vault.resolve_asset(digest)
    assert resolution is not None
    assert resolution.verification is not None
    calls = _counting_hasher(monkeypatch)
    with open_verified(resolution.path, digest, resolution.verification) as handle:
        assert handle.read() == data
    assert calls == [0]


def test_asset_endpoint_serves_verified_bytes_and_refuses_stale(
    tmp_path: Path,
) -> None:
    data = b"distributable model bytes" * 64
    digest = digest_bytes(data)

    async def scenario() -> None:
        vault = AssetVault(tmp_path / "vault")
        with vault.writer(digest) as writer:
            writer.write(data)
            writer.commit()
        server = TestServer(create_app(make_engine, SCHEMAS, asset_export=vault))
        await server.start_server()
        try:
            async with aiohttp.ClientSession() as http:
                good = await http.get(server.make_url(f"/assets/{digest}"))
                assert good.status == 200
                assert good.headers["Content-Length"] == str(len(data))
                assert await good.read() == data

                # Corrupt the stored file in place; resolve still finds it,
                # but the endpoint must refuse rather than mirror garbage.
                stored = vault.resolve(digest)
                assert stored is not None
                stored.write_bytes(b"Z" * len(data))
                stale = await http.get(server.make_url(f"/assets/{digest}"))
                assert stale.status == 409
                assert stale.headers["Cache-Control"] == "no-store"
                body = await stale.json()
                assert digest in body["error"]
        finally:
            await server.close()

    asyncio.run(scenario())


def test_api_asset_get_and_metadata_refuse_stale_vault_content(tmp_path: Path) -> None:
    """The client surface (/api/assets/{digest} + /metadata) holds the same
    invariant as the peer surface: an immutable-cache contract must never
    be stamped on bytes (or probed facts) that no longer hash to the
    digest."""

    async def scenario() -> None:
        vault = AssetVault(tmp_path / "vault")
        library = ServerLibrary(vault=vault, store=LibraryStore(tmp_path / "library.sqlite"))
        client = TestClient(TestServer(create_app(make_engine, SCHEMAS, library=library)))
        await client.start_server()
        try:
            data = b"payload destined for corruption" * 8
            digest = (await (await client.post("/api/assets", data=data)).json())["digest"]
            good = await client.get(f"/api/assets/{digest}")
            assert good.status == 200
            assert await good.read() == data
            assert good.headers["ETag"] == f'"{digest}"'

            stored = vault.resolve(digest)
            assert stored is not None
            stored.write_bytes(b"Q" * len(data))  # corrupt in place

            stale = await client.get(f"/api/assets/{digest}")
            assert stale.status == 409
            assert stale.headers["Cache-Control"] == "no-store"
            assert digest in (await stale.json())["error"]

            stale_meta = await client.get(f"/api/assets/{digest}/metadata")
            assert stale_meta.status == 409
            assert stale_meta.headers["Cache-Control"] == "no-store"
        finally:
            await client.close()

    asyncio.run(scenario())


# --- cancellation: a dead request never leaks or races the descriptor --------


def test_open_helper_closes_descriptor_when_cancelled_mid_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """aiohttp cancels the handler while the worker thread is still
    hashing: the awaiter dies immediately, and the descriptor the worker
    eventually produces is closed by the completion callback, not leaked."""
    data = b"slow to verify"
    path = write_asset(tmp_path, data)
    digest = digest_bytes(data)
    started = threading.Event()
    release = threading.Event()
    opened: list[BinaryIO] = []

    def slow_open(target: Path, expected: str) -> BinaryIO:
        started.set()
        assert release.wait(10)
        handle = open_verified(target, expected)
        opened.append(handle)
        return handle

    monkeypatch.setattr(asset_stream_module, "open_verified", slow_open)

    async def scenario() -> None:
        task = asyncio.ensure_future(open_verified_sized(path, digest))
        await asyncio.to_thread(started.wait, 10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        async with asyncio.timeout(10):
            while not (opened and opened[0].closed):
                await asyncio.sleep(0.01)

    asyncio.run(scenario())


def test_stream_cancellation_defers_close_to_the_worker_read(
    tmp_path: Path,
) -> None:
    """Cancelled mid-read: the handle must be closed AFTER the in-flight
    worker read returns - closing concurrently could block the event loop
    on the reader's internal lock."""
    read_started = threading.Event()
    release = threading.Event()
    events: list[str] = []

    class SlowHandle:
        closed = False

        def read(self, _size: int) -> bytes:
            read_started.set()
            assert release.wait(10)
            events.append("read-returned")
            return b""

        def close(self) -> None:
            self.closed = True
            events.append("closed")

    handle = SlowHandle()

    async def scenario() -> None:
        request = make_mocked_request("GET", "/assets/x")
        task = asyncio.ensure_future(stream_verified(request, cast(BinaryIO, handle), 0))
        await asyncio.to_thread(read_started.wait, 10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not handle.closed  # the worker read still owns the handle
        release.set()
        async with asyncio.timeout(10):
            while not handle.closed:
                await asyncio.sleep(0.01)
        assert events == ["read-returned", "closed"]

    asyncio.run(scenario())
