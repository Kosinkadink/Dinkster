"""Asset distribution: bytes move by identity (DESIGN 3.12).

The vault ingests streams and lands files only when they hash to the
digest the caller already knew; provenance records are leads, never
authorities; fetch tries candidates conservatively; and a peer instance's
/assets endpoints are just more candidates. A wrong or malicious source
can waste bandwidth - it can never plant wrong bytes.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from dinkster_assets import (
    AssetError,
    AssetRef,
    AssetVault,
    ChainResolver,
    FetchingResolver,
    LocalAssetLibrary,
    ProvenanceRecord,
    ProvenanceStore,
    digest_bytes,
    fetch_asset,
)
from dinkster_server import create_app, peer_asset_sources
from test_server import SCHEMAS, make_engine

MODEL_BYTES = b"pretend this is a 7 GB checkpoint" * 64
MODEL_DIGEST = digest_bytes(MODEL_BYTES)


# --- vault: verifying, atomic, idempotent ------------------------------------


def test_vault_streaming_ingest_and_resolve(tmp_path: Path) -> None:
    vault = AssetVault(tmp_path / "vault")
    with vault.writer(MODEL_DIGEST) as writer:
        for offset in range(0, len(MODEL_BYTES), 100):
            writer.write(MODEL_BYTES[offset : offset + 100])
        path = writer.commit()
    assert path.read_bytes() == MODEL_BYTES
    assert vault.resolve(MODEL_DIGEST) == path
    assert vault.has(MODEL_DIGEST)
    assert vault.digests() == [MODEL_DIGEST]


def test_vault_digests_announces_only_canonical_64_hex_files(tmp_path: Path) -> None:
    vault = AssetVault(tmp_path / "vault")
    with vault.writer(MODEL_DIGEST) as writer:
        writer.write(MODEL_BYTES)
        path = writer.commit()
    (path.parent / f"{path.name}.0123456789abcdef.dinkster.safetensors").write_bytes(b"derived")
    (path.parent / ("g" * 64)).write_bytes(b"not hex")
    (path.parent / ("a" * 63)).write_bytes(b"too short")

    assert vault.digests() == [MODEL_DIGEST]


def test_vault_delete_unlinks_derived_siblings(tmp_path: Path) -> None:
    vault = AssetVault(tmp_path / "vault")
    with vault.writer(MODEL_DIGEST) as writer:
        writer.write(MODEL_BYTES)
        path = writer.commit()
    sidecar = path.parent / f"{path.name}.0123456789abcdef.dinkster.safetensors"
    sidecar.write_bytes(b"derived")
    unrelated = path.parent / f"other.{path.name}.dinkster.safetensors"
    unrelated.write_bytes(b"keep")

    assert vault.delete(MODEL_DIGEST) is True
    assert not path.exists()
    assert not sidecar.exists()
    assert unrelated.exists()


def test_vault_refuses_bytes_that_do_not_verify(tmp_path: Path) -> None:
    vault = AssetVault(tmp_path / "vault")
    with pytest.raises(AssetError, match="did not verify"):
        with vault.writer(MODEL_DIGEST) as writer:
            writer.write(b"not the model at all")
            writer.commit()
    assert vault.resolve(MODEL_DIGEST) is None
    assert not list(vault.root.rglob("*.ingest-*"))  # rollback left nothing


def test_vault_abandoned_writer_leaves_nothing(tmp_path: Path) -> None:
    vault = AssetVault(tmp_path / "vault")
    with pytest.raises(RuntimeError, match="download died"):
        with vault.writer(MODEL_DIGEST) as writer:
            writer.write(MODEL_BYTES[:100])
            raise RuntimeError("download died")
    assert vault.resolve(MODEL_DIGEST) is None
    assert not [p for p in vault.root.rglob("*") if p.is_file()]


def test_vault_temp_files_are_ignored_by_library_scans(tmp_path: Path) -> None:
    """A partial ingest must never be cataloged as an asset - the ignore
    rule that protects the index file protects downloads too."""
    root = tmp_path / "shared-folder"
    root.mkdir()
    (root / "real.safetensors").write_bytes(b"real model")
    vault = AssetVault(root)
    writer = vault.writer(MODEL_DIGEST)  # left open: a download in flight
    writer.write(MODEL_BYTES[:100])
    library = LocalAssetLibrary(root)
    assert library.scan() == 1  # only the real file; dot-temp invisible
    writer._discard()  # noqa: SLF001 - test cleanup


# --- provenance: leads, additive, persistent ----------------------------------


def test_provenance_store_merges_and_persists(tmp_path: Path) -> None:
    store_path = tmp_path / "provenance.json"
    store = ProvenanceStore(store_path)
    store.add(
        ProvenanceRecord(
            digest=MODEL_DIGEST,
            sources=("https://hub.example/model.safetensors",),
            license="apache-2.0",
        )
    )
    merged = store.add(
        ProvenanceRecord(
            digest=MODEL_DIGEST,
            sources=(
                "https://mirror.example/model.safetensors",
                "https://hub.example/model.safetensors",  # duplicate: unioned away
            ),
            note="community mirror added",
        )
    )
    assert merged.sources == (
        "https://hub.example/model.safetensors",
        "https://mirror.example/model.safetensors",
    )
    assert merged.license == "apache-2.0"  # earlier scalar survives the merge

    reloaded = ProvenanceStore(store_path)
    assert reloaded.sources(MODEL_DIGEST) == merged.sources
    assert reloaded.get(MODEL_DIGEST) is not None
    assert reloaded.get(MODEL_DIGEST).note == "community mirror added"  # type: ignore[union-attr]


def test_provenance_malformed_file_is_empty_store(tmp_path: Path) -> None:
    store_path = tmp_path / "provenance.json"
    store_path.write_text("{not json", "utf-8")
    assert ProvenanceStore(store_path).records() == ()
    store_path.write_text(json.dumps([{"digest": "not-a-digest"}, 42]), "utf-8")
    assert ProvenanceStore(store_path).records() == ()  # rows dropped, not fatal


# --- fetch: digest is the only authority --------------------------------------


def make_asset_app(assets: dict[str, bytes]) -> web.Application:
    """A plain HTTP host for asset bytes: a hub, a mirror, whatever."""

    async def serve(request: web.Request) -> web.Response:
        data = assets.get(request.match_info["name"])
        if data is None:
            return web.Response(status=404)
        return web.Response(body=data)

    app = web.Application()
    app.router.add_get("/files/{name}", serve)
    return app


def test_fetch_verifies_and_falls_through_mirrors(tmp_path: Path) -> None:
    async def scenario() -> None:
        server = TestServer(make_asset_app({"good": MODEL_BYTES, "lying": b"wrong bytes entirely"}))
        await server.start_server()
        try:
            vault = AssetVault(tmp_path / "vault")
            urls = [
                "file:///etc/passwd",  # non-http lead: skipped, never opened
                str(server.make_url("/files/missing")),  # 404: next
                str(server.make_url("/files/lying")),  # fails verification: next
                str(server.make_url("/files/good")),
            ]
            path = await asyncio.to_thread(fetch_asset, MODEL_DIGEST, urls, vault)
            assert path is not None and path.read_bytes() == MODEL_BYTES
            # Held identities never touch the network again.
            again = await asyncio.to_thread(
                fetch_asset, MODEL_DIGEST, ["http://nowhere.invalid/x"], vault
            )
            assert again == path
        finally:
            await server.close()

    asyncio.run(scenario())


def test_fetch_every_candidate_failing_is_none(tmp_path: Path) -> None:
    vault = AssetVault(tmp_path / "vault")
    assert fetch_asset(MODEL_DIGEST, [], vault) is None
    assert fetch_asset(MODEL_DIGEST, ["file:///etc/passwd"], vault) is None
    assert vault.digests() == []


# --- instance endpoints + peer pull -------------------------------------------


def test_asset_endpoints_and_peer_fetch(tmp_path: Path) -> None:
    """Instance A holds a model; instance B resolves it by digest through
    the same provenance-guided fetch path any mirror would use, verifies,
    and afterwards serves it locally."""

    async def scenario() -> None:
        vault_a = AssetVault(tmp_path / "a")
        with vault_a.writer(MODEL_DIGEST) as writer:
            writer.write(MODEL_BYTES)
            writer.commit()
        server = TestServer(create_app(make_engine, SCHEMAS, asset_export=vault_a))
        await server.start_server()
        try:
            async with aiohttp.ClientSession() as http:
                listing = await http.get(server.make_url("/assets"))
                assert listing.status == 200
                assert (await listing.json())["digests"] == [MODEL_DIGEST]
                bad = await http.get(server.make_url("/assets/not-a-digest"))
                assert bad.status == 400
                missing = await http.get(server.make_url(f"/assets/blake3:{'0' * 64}"))
                assert missing.status == 404

            vault_b = AssetVault(tmp_path / "b")
            endpoint = str(server.make_url("")).rstrip("/")
            resolver = FetchingResolver(vault_b, peer_asset_sources([endpoint]))
            path = await asyncio.to_thread(resolver.resolve, MODEL_DIGEST)
            assert path is not None and path.read_bytes() == MODEL_BYTES
            assert vault_b.has(MODEL_DIGEST)  # promoted: held locally now
        finally:
            await server.close()
        # The peer is gone; B still resolves from its own vault.
        assert vault_b.resolve(MODEL_DIGEST) is not None

    asyncio.run(scenario())


def test_asset_endpoints_absent_without_export(tmp_path: Path) -> None:
    async def scenario() -> None:
        server = TestServer(create_app(make_engine, SCHEMAS))
        await server.start_server()
        try:
            async with aiohttp.ClientSession() as http:
                resp = await http.get(server.make_url("/assets"))
                assert resp.status == 404
        finally:
            await server.close()

    asyncio.run(scenario())


def test_peer_sources_merge_provenance_and_peers(tmp_path: Path) -> None:
    provenance = ProvenanceStore(tmp_path / "provenance.json")
    provenance.add(ProvenanceRecord(digest=MODEL_DIGEST, sources=("https://hub.example/m",)))
    sources = peer_asset_sources(
        ["http://peer1:8188/", "http://peer2:8188"], provenance=provenance.sources
    )
    assert sources(MODEL_DIGEST) == [
        "https://hub.example/m",
        f"http://peer1:8188/assets/{MODEL_DIGEST}",
        f"http://peer2:8188/assets/{MODEL_DIGEST}",
    ]
    lan_first = peer_asset_sources(
        ["http://peer1:8188"], provenance=provenance.sources, peers_first=True
    )
    assert lan_first(MODEL_DIGEST)[0] == f"http://peer1:8188/assets/{MODEL_DIGEST}"


# --- the node-facing path: AssetRef.local_path() through the chain ------------


def test_asset_ref_materializes_through_chain_with_fetch(tmp_path: Path) -> None:
    """What a node actually experiences: local library first, vault next,
    network last - and the ref neither knows nor cares which one answered."""

    async def scenario() -> None:
        vault_a = AssetVault(tmp_path / "a")
        with vault_a.writer(MODEL_DIGEST) as writer:
            writer.write(MODEL_BYTES)
            writer.commit()
        server = TestServer(create_app(make_engine, SCHEMAS, asset_export=vault_a))
        await server.start_server()
        try:
            library_root = tmp_path / "models"
            library_root.mkdir()
            library = LocalAssetLibrary(library_root)
            library.scan()
            vault_b = AssetVault(tmp_path / "b")
            endpoint = str(server.make_url("")).rstrip("/")
            chain = ChainResolver(
                library,
                vault_b,
                FetchingResolver(vault_b, peer_asset_sources([endpoint])),
            )
            ref = AssetRef(
                digest=MODEL_DIGEST,
                name="model.safetensors",
                size=len(MODEL_BYTES),
                resolver=chain,
            )
            path = await asyncio.to_thread(ref.local_path)
            assert path.read_bytes() == MODEL_BYTES
        finally:
            await server.close()

    asyncio.run(scenario())
