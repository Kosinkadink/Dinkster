"""Produced publication recovery and storage route safety."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer
from dinkster_assets import AssetVault, LibraryStore, digest_bytes
from dinkster_assets.declared import produced_output_scratch, use_produced_output_job
from dinkster_assets.scratch import ProducedScratchState, hold_produced_job
from dinkster_server import Principal, ServerLibrary, create_app
from dinkster_server.auth import StaticBearerAuthenticator
from dinkster_server.cleanup import recover_produced_publications
from test_server import SCHEMAS, make_engine

from tests.async_scenarios import run_scenario


def test_recovery_adopts_only_offered_publications(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    vault = AssetVault(tmp_path / "vault")
    offered_data, unoffered_data = b"offered", b"unoffered"
    offered, unoffered = digest_bytes(offered_data), digest_bytes(unoffered_data)

    with closing(hold_produced_job(scratch, "finished")), use_produced_output_job("finished", 1):
        source = produced_output_scratch(
            {"DINKSTER_PACK_SCRATCH": str(scratch / "packs" / "media")}
        )
        assert source is not None
        for digest, data in ((offered, offered_data), (unoffered, unoffered_data)):
            with source.writer(digest) as writer:
                writer.write(data)
                writer.commit()
        ProducedScratchState(source.root).offer([offered])

    result = recover_produced_publications(vault, scratch)
    assert result["verifiedPublications"] == 1
    path = vault.resolve(offered)
    assert path is not None and path.read_bytes() == offered_data
    assert vault.resolve(unoffered) is None


def test_recovery_preserves_corrupt_source_for_a_retry(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    vault = AssetVault(tmp_path / "vault")
    data = b"offered"
    digest = digest_bytes(data)

    with closing(hold_produced_job(scratch, "finished")), use_produced_output_job("finished", 1):
        source = produced_output_scratch(
            {"DINKSTER_PACK_SCRATCH": str(scratch / "packs" / "media")}
        )
        assert source is not None
        with source.writer(digest) as writer:
            writer.write(data)
            path = writer.commit()
        ProducedScratchState(source.root).offer([digest])
    path.write_bytes(b"changed!")

    result = recover_produced_publications(vault, scratch)
    assert result == {"verifiedPublications": 0, "blockedInspections": 1}
    assert path.read_bytes() == b"changed!"
    assert vault.resolve(digest) is None


def test_storage_routes_are_operator_only_and_reject_paths(tmp_path: Path) -> None:
    async def scenario() -> None:
        library = ServerLibrary(AssetVault(tmp_path / "vault"), LibraryStore(tmp_path / "library"))
        principal = Principal(
            "reader", {"local": frozenset({"assets:read", "assets:write"})}, kind="human"
        )
        client = TestClient(
            TestServer(
                create_app(
                    make_engine,
                    SCHEMAS,
                    library=library,
                    authenticator=StaticBearerAuthenticator({"reader": principal}),
                )
            )
        )
        await client.start_server()
        try:
            headers = {"Authorization": "Bearer reader"}
            assert (await client.get("/api/library/storage")).status == 401
            assert (await client.get("/api/library/storage", headers=headers)).status == 403
            response = await client.post("/api/library/storage/clear", json={}, headers=headers)
            assert response.status == 403
        finally:
            await client.close()

    run_scenario(scenario())


def test_storage_routes_reject_filesystem_targets(tmp_path: Path) -> None:
    async def scenario() -> None:
        library = ServerLibrary(AssetVault(tmp_path / "vault"), LibraryStore(tmp_path / "library"))
        client = TestClient(TestServer(create_app(make_engine, SCHEMAS, library=library)))
        await client.start_server()
        try:
            assert (await client.get("/api/library/storage?path=/")).status == 400
            assert (
                await client.post("/api/library/storage/clear", json={"path": "/"})
            ).status == 400
            assert (
                await client.put("/api/library/storage/policy?path=/", json={"limitBytes": None})
            ).status == 400
        finally:
            await client.close()

    run_scenario(scenario())


def test_storage_clear_without_temporary_vaults_keeps_durable_assets(tmp_path: Path) -> None:
    async def scenario() -> None:
        vault = AssetVault(tmp_path / "vault")
        data = b"saved document"
        digest = digest_bytes(data)
        with vault.writer(digest) as writer:
            writer.write(data)
            writer.commit()
        library = ServerLibrary(vault, LibraryStore(tmp_path / "library"))
        client = TestClient(TestServer(create_app(make_engine, SCHEMAS, library=library)))
        await client.start_server()
        try:
            response = await client.post("/api/library/storage/clear", json={})
            assert response.status == 200
            assert vault.resolve(digest) is not None
        finally:
            await client.close()

    run_scenario(scenario())
