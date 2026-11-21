"""Public pipe contracts for shared project records and helper lifetime."""

from __future__ import annotations

import base64
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
import tomllib
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from threading import Barrier
from typing import cast

import pytest
from dinkster_library import LibraryClient, LibraryError


def test_collection_keeps_dependency_roots_and_collects_main_and_preview_vaults(
    tmp_path: Path,
) -> None:
    from dinkster_assets import AssetVault, digest_bytes
    from dinkster_assets.produced import ProducedAssetInventory

    main, preview = AssetVault(tmp_path / "vault"), AssetVault(tmp_path / "preview")
    retained, child, discarded, previewed = (
        b"kept root",
        b"dependency",
        b"main discard",
        b"preview discard",
    )
    for vault, payload in (
        (main, retained),
        (main, child),
        (main, discarded),
        (preview, previewed),
    ):
        digest = digest_bytes(payload)
        inventory = ProducedAssetInventory(vault.root)
        inventory.pending(digest, len(payload))
        with vault.writer(digest) as writer:
            writer.write(payload)
            path = writer.commit()
        inventory.adopted(digest, len(payload))
        if payload == discarded:
            path.with_suffix(".extra").write_bytes(b"metadata")
    discarded_files = [
        path
        for vault, payload in ((main, discarded), (preview, previewed))
        for path in (vault.root / digest_bytes(payload).split(":", 1)[1][:2]).glob(
            digest_bytes(payload).split(":", 1)[1] + "*"
        )
    ]
    discarded_bytes = sum(path.stat().st_size for path in discarded_files)
    paths = [str(main.root), str(preview.root)]
    with LibraryClient(tmp_path / "library") as helper, LibraryClient(tmp_path / "library") as peer:
        helper.call("references.engine.attach", engine="ab" * 16)
        helper.call("retention.clients.initialize", pristine=True)
        helper.call(
            "references.engine.patch",
            revision=1,
            patch={"complete": True, "digests": [digest_bytes(retained)]},
        )
        helper.call(
            "dependencies.put",
            digest=digest_bytes(retained),
            dependencies=[{"digest": digest_bytes(child)}],
        )
        held = helper.call("references.acquire")
        with pytest.raises(LibraryError) as foreign:
            peer.call("references.collect", lease=held["lease"], vaults=paths, collect=True)
        assert foreign.value.code == "invalid"
        for invalid in (
            {"vaults": [str(main.root), str(main.root)]},
            {"vaults": ["relative"]},
            {"collect": 1},
            {"limitBytes": -1},
            {"limitBytes": True},
        ):
            with pytest.raises(LibraryError) as refused:
                helper.call(
                    "references.collect", **({"lease": held["lease"], "vaults": paths} | invalid)
                )
            assert refused.value.code == "invalid"
        measured = helper.call("references.collect", lease=held["lease"], vaults=paths)
        assert measured == {
            "reclaimedBytes": 0,
            "reclaimedFiles": 0,
            "reclaimableBytes": discarded_bytes,
            "retainedFiles": 2,
            "failedItems": 0,
            "vaultBytes": 46,
            "adoptedBytes": 46,
        }
        assert main.has(digest_bytes(discarded)) and preview.has(digest_bytes(previewed))
        assert (
            helper.call(
                "references.collect", lease=held["lease"], vaults=paths, collect=True, limitBytes=46
            )
            == measured
        )
        assert main.has(digest_bytes(discarded)) and preview.has(digest_bytes(previewed))
        result = helper.call(
            "references.collect", lease=held["lease"], vaults=paths, collect=True, limitBytes=19
        )
        assert result == {
            "reclaimedBytes": discarded_bytes,
            "reclaimedFiles": len(discarded_files),
            "reclaimableBytes": 0,
            "retainedFiles": 2,
            "failedItems": 0,
            "vaultBytes": 19,
            "adoptedBytes": 19,
        }
        assert main.has(digest_bytes(retained)) and main.has(digest_bytes(child))
        assert not main.has(digest_bytes(discarded)) and not preview.has(digest_bytes(previewed))
        assert main.resolve(digest_bytes(discarded)) is None
        assert [entry.digest for entry in ProducedAssetInventory(main.root).entries()] == [
            digest_bytes(retained),
            digest_bytes(child),
        ]
        assert ProducedAssetInventory(preview.root).entries() == ()
        helper.call("references.release", lease=held["lease"])
        with pytest.raises(LibraryError) as released:
            helper.call("references.collect", lease=held["lease"], vaults=paths, collect=True)
        assert released.value.code == "invalid"


@pytest.mark.parametrize("fault", ("pending", "reader", "inventory"))
def test_collection_validates_every_vault_before_deleting(tmp_path: Path, fault: str) -> None:
    from dinkster_assets import AssetVault, digest_bytes
    from dinkster_assets.produced import ProducedAssetInventory
    from dinkster_assets.scratch import hold_produced_vault

    main, preview = AssetVault(tmp_path / "vault"), AssetVault(tmp_path / "preview")
    digest = digest_bytes(b"produced bytes")
    for vault in (main, preview):
        inventory = ProducedAssetInventory(vault.root)
        inventory.pending(digest, 14)
        with vault.writer(digest) as writer:
            writer.write(b"produced bytes")
            writer.commit()
        if vault is main or fault != "pending":
            inventory.adopted(digest, 14)
    lease = hold_produced_vault(preview.root) if fault == "reader" else None
    if fault == "inventory":
        (preview.root / ".produced-assets.sqlite").unlink()
    try:
        with LibraryClient(tmp_path / "library") as helper:
            held = helper.call("references.acquire")
            with pytest.raises(LibraryError) as refused:
                helper.call(
                    "references.collect",
                    lease=held["lease"],
                    vaults=[str(main.root), str(preview.root)],
                    collect=True,
                )
            assert refused.value.code == ("busy" if fault == "reader" else "collection")
            assert main.has(digest) and preview.has(digest)
            assert len(ProducedAssetInventory(main.root).entries()) == 1
    finally:
        if lease is not None:
            lease.close()


@pytest.mark.parametrize("fault", ("content", "sidecar"))
def test_collection_preserves_unverified_adoptions(tmp_path: Path, fault: str) -> None:
    from dinkster_assets import AssetVault, digest_bytes
    from dinkster_assets.produced import ProducedAssetInventory

    vault = AssetVault(tmp_path / "vault")
    payload = b"original content"
    digest = digest_bytes(payload)
    inventory = ProducedAssetInventory(vault.root)
    inventory.pending(digest, len(payload))
    with vault.writer(digest) as writer:
        writer.write(payload)
        path = writer.commit()
    inventory.adopted(digest, len(payload))
    before = inventory.entries()
    if fault == "content":
        path.write_bytes(b"x" * len(payload))
    else:
        path.with_suffix(".unsafe").mkdir()
    with LibraryClient(tmp_path / "library") as helper:
        held = helper.call("references.acquire")
        result = helper.call(
            "references.collect", lease=held["lease"], vaults=[str(vault.root)], collect=True
        )
        assert result["failedItems"] == 1
        assert result["reclaimableBytes"] is None
        assert result["reclaimedBytes"] == result["reclaimedFiles"] == 0
        assert path.read_bytes() == (b"x" * len(payload) if fault == "content" else payload)
        assert inventory.entries() == before


def test_collection_stops_with_helper_without_holding_sqlite_write_lock(tmp_path: Path) -> None:
    from dinkster_assets import AssetVault, digest_bytes
    from dinkster_assets.produced import ProducedAssetInventory
    from dinkster_assets.scratch import claim_inactive_produced_vault

    vault = AssetVault(tmp_path / "vault")
    digest = digest_bytes(b"must survive helper death")
    inventory = ProducedAssetInventory(vault.root)
    inventory.pending(digest, 25)
    with vault.writer(digest) as writer:
        writer.write(b"must survive helper death")
        writer.commit()
    inventory.adopted(digest, 25)
    stored_bytes = sum(
        path.stat().st_size
        for path in (vault.root / digest.split(":", 1)[1][:2]).glob(digest.split(":", 1)[1] + "*")
    )
    marker = tmp_path / "hashing"
    command = [
        sys.executable,
        "-c",
        f"""
from pathlib import Path
import time
from dinkster_library import produced_collection
from dinkster_library.__main__ import main
def paused(path):
    Path({str(marker)!r}).write_text("hashing")
    while True:
        time.sleep(0.01)
produced_collection.digest_file = paused
main()
""",
    ]
    with (
        LibraryClient(tmp_path / "library", command=command) as helper,
        LibraryClient(tmp_path / "library") as peer,
    ):
        held = helper.call("references.acquire")
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(
                helper.call,
                "references.collect",
                lease=held["lease"],
                vaults=[str(vault.root)],
                collect=True,
            )
            try:
                deadline = time.monotonic() + 10
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert marker.exists()
                assert peer.call("projects.register", id="while-hashing")["project"] is not None
            finally:
                os.kill(helper.pid, signal.SIGTERM)
            with pytest.raises(LibraryError):
                pending.result(timeout=10)
        assert vault.has(digest)
        assert len(inventory.entries()) == 1
        with claim_inactive_produced_vault(vault.root) as inactive:
            assert inactive
        new_held = peer.call("references.acquire")
        assert (
            peer.call(
                "references.collect",
                lease=new_held["lease"],
                vaults=[str(vault.root)],
                collect=True,
            )["reclaimedBytes"]
            == stored_bytes
        )
        assert not vault.has(digest)
        assert inventory.entries() == ()


def test_cleanup_policy_is_shared_revisioned_and_not_collection_authority(tmp_path: Path) -> None:
    root = tmp_path / "library"
    with LibraryClient(root) as first, LibraryClient(root) as second:
        assert "cleanup.policy" in cast("list[str]", first.hello["capabilities"])
        initial = first.call("cleanup.policy.get")["policy"]
        assert initial == {"revision": 0, "values": {"limitBytes": 21474836480}}
        second.call(
            "cleanup.policy.patch", revision=0, patch={"future": {"keep": None}, "limitBytes": 91}
        )
        with pytest.raises(LibraryError) as stale:
            first.call("cleanup.policy.patch", revision=0, patch={"limitBytes": 37})
        assert stale.value.code == "stale"
        current = cast("dict[str, object]", first.call("cleanup.policy.get")["policy"])
        for invalid in (True, False, 0, -1, 1.5, "37", 2**53):
            with pytest.raises(LibraryError) as refused:
                first.call(
                    "cleanup.policy.patch",
                    revision=current["revision"],
                    patch={"limitBytes": invalid},
                )
            assert refused.value.code == "invalid"
            assert first.call("cleanup.policy.get")["policy"] == current
        with pytest.raises(LibraryError):
            first.call(
                "cleanup.policy.patch", revision=current["revision"], remove=[["limitBytes"]]
            )
        for limit in (None, 37, 2**53 - 1):
            current = cast(
                "dict[str, object]",
                first.call(
                    "cleanup.policy.patch",
                    revision=current["revision"],
                    patch={"limitBytes": limit},
                )["policy"],
            )
            assert current["values"] == {"limitBytes": limit, "future": {"keep": None}}
        first.call("references.engine.attach", engine="a" * 32)
        with pytest.raises(LibraryError) as incomplete:
            first.call("references.acquire")
        assert incomplete.value.code == "references"
    with LibraryClient(root) as reopened, LibraryClient(tmp_path / "other") as other:
        assert reopened.call("cleanup.policy.get")["policy"] == current
        assert other.call("cleanup.policy.get")["policy"] == initial
        legacy = root / "library.sqlite"
        legacy.write_bytes(b"legacy library remains untouched")
        for operation, arguments in (
            ("cleanup.policy.get", {}),
            (
                "cleanup.policy.patch",
                {"revision": current["revision"], "patch": {"limitBytes": 23}},
            ),
        ):
            with pytest.raises(LibraryError) as migration:
                reopened.call(operation, **arguments)
            assert migration.value.code == "migration"
            assert legacy.read_bytes() == b"legacy library remains untouched"


def test_model_preferences_share_account_not_library_and_preserve_peer_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    from dinkster_assets import AssetRef, MountsError, digest_bytes
    from dinkster_assets.integrity import verification_record
    from dinkster_assets.mounts import ModelFileSelection

    from dinkster.model_settings import ModelLocation, ModelSettings, ModelSettingsConflict

    home = tmp_path / "account"
    monkeypatch.setenv("DINKSTER_HOME", str(home))
    models = tmp_path / "models"
    models.mkdir()
    path = models / "model.bin"
    path.write_bytes(b"verified model")
    with LibraryClient(tmp_path / "one") as first, LibraryClient(tmp_path / "two") as second:
        assert "folders.models" in cast("list[str]", first.hello["capabilities"])
        initial = ModelSettings.load(first)
        changed = ModelSettings(
            (ModelLocation("studio", models, "read", kind="model/lora"),), None, library=first
        )
        changed.save(initial.revision)
        current = ModelSettings.load(second)
        assert current.locations == changed.locations
        raw = cast("dict[str, object]", second.call("folders.models.get")["folders"])
        second.call(
            "folders.models.patch",
            revision=raw["revision"],
            patch={"future": {"nullable": None}, "locations": {"studio": {"future": "keep"}}},
        )
        with pytest.raises(ModelSettingsConflict):
            replace(changed, locations=()).save(current.revision)
        current = ModelSettings.load(first)
        replace(current, locations=(replace(current.locations[0], visibility="opt-in"),)).save(
            current.revision
        )
        raw = cast("dict[str, object]", second.call("folders.models.get")["folders"])
        values = cast("dict[str, object]", raw["values"])
        assert values["future"] == {"nullable": None}
        assert (
            cast("dict[str, dict[str, object]]", values["locations"])["studio"]["future"] == "keep"
        )
        digest = digest_bytes(path.read_bytes())
        verification = verification_record(digest, path.stat())
        assert verification is not None
        selection = ModelFileSelection(
            "studio-lora",
            models,
            "model/lora",
            models,
            "model.bin",
            AssetRef(
                digest,
                "model.bin",
                path.stat().st_size,
                virtual_path="model-files/studio-lora/model.bin",
            ),
            verification,
            ("first.bin",),
        )
        current = ModelSettings.load(first)
        current.remember_file_selection(selection)
        file_record = cast("dict[str, object]", second.call("folders.models.get")["folders"])
        second.call(
            "folders.models.patch",
            revision=file_record["revision"],
            patch={"files": {selection.ref.virtual_path: {"future": {"keep": True}}}},
        )
        current.remember_file_selection(replace(selection, aliases=("second.bin",)))
        file_record = cast("dict[str, object]", second.call("folders.models.get")["folders"])
        files = cast(
            "dict[str, dict[str, object]]",
            cast("dict[str, object]", file_record["values"])["files"],
        )
        assert files[selection.ref.virtual_path]["future"] == {"keep": True}
        assert ModelSettings.load(second).file_selections()[0].aliases == (
            "first.bin",
            "second.bin",
        )
        path.write_bytes(b"changed bytes")
        with pytest.raises(ModelSettingsConflict):
            current.remember_file_selection(selection)
        prior = second.call("folders.models.get")
        with pytest.raises(LibraryError) as stale:
            second.call("folders.models.patch", revision=raw["revision"], patch={"future": False})
        assert stale.value.code == "stale"
        assert second.call("folders.models.get") == prior
        first.close()
        with pytest.raises(MountsError):
            ModelSettings.load(first)
    with LibraryClient(tmp_path / "three") as third:
        assert ModelSettings.load(third).locations[0].visibility == "opt-in"
        assert third.call("folders.models.get") == prior
    assert (home / "model-settings.sqlite").is_file()
    assert not (home / "model-settings.json").exists()
    assert not (home / "model-file-selections.json").exists()
    assert not (tmp_path / "one" / "model-settings.sqlite").exists()
    monkeypatch.setenv("DINKSTER_HOME", str(tmp_path / "other-account"))
    with LibraryClient(tmp_path / "one") as other:
        assert ModelSettings.load(other).locations[0].id == "models"


@pytest.mark.parametrize("legacy", ["model-settings.json", "model-file-selections.json"])
def test_model_preferences_refuse_unmigrated_account_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, legacy: str
) -> None:
    home = tmp_path / "account"
    home.mkdir()
    monkeypatch.setenv("DINKSTER_HOME", str(home))
    original = b"legacy model preferences must not be overwritten"
    (home / legacy).write_bytes(original)
    with LibraryClient(tmp_path / "library") as helper:
        for operation, arguments in (
            ("folders.models.get", {}),
            ("folders.models.patch", {"revision": 0, "patch": {}}),
        ):
            with pytest.raises(LibraryError) as error:
                helper.call(operation, **arguments)
            assert error.value.code == "migration"
    assert (home / legacy).read_bytes() == original
    assert not (home / "model-settings.sqlite").exists()


def test_finished_history_is_shared_immutable_paged_and_does_not_resurrect(tmp_path: Path) -> None:
    def record(identifier: str, finished: float, scope: str = "local") -> dict[str, object]:
        return {
            "jobRef": identifier,
            "runId": identifier,
            "scope": scope,
            "principalId": "local",
            "principalKind": "human",
            "clientId": "editor",
            "jobId": "job-" + identifier,
            "state": "completed",
            "priority": -2,
            "attemptId": 1,
            "submittedAt": 1.0,
            "finishedAt": finished,
            "executed": 2,
            "cached": 1,
            "skipped": 0,
            "nodeReceipts": [],
            "assetDigests": None if identifier == "old" else ["blake3:" + "ab" * 32],
            "futureField": {"keep": True},
        }

    documents = [record("old", 19.0), record("y", 30.0), record("z", 30.0)]
    documents[1]["sourceDocument"] = "blake3:" + "cd" * 32
    with LibraryClient(tmp_path) as first, LibraryClient(tmp_path) as second:
        assert "history" in cast("list[str]", first.hello["capabilities"])
        first.call("projects.register", id="film")
        for patch in (
            {"state": "running"},
            {"assetDigests": ["invalid"]},
            {"attemptId": True},
            {"finishedAt": 10**400},
            {"sourceDocument": "invalid"},
        ):
            with pytest.raises(LibraryError) as invalid:
                first.call(
                    "history.append", scope="local", record=dict(record("invalid", 3.0), **patch)
                )
            assert invalid.value.code == "invalid"
        assert first.call("history.get", scope="local", id="invalid") == {"record": None}
        for document in documents:
            assert first.call("history.append", scope="local", project="film", record=document) == {
                "record": document,
                "project": "film",
                "created": True,
            }
        second.call("history.append", scope="local", record=record("default", 99.0))
        private = dict(record("private", 100.0, "private"), assetDigests=["blake3:" + "ef" * 32])
        second.call("history.append", scope="private", record=private)
        duplicate = dict(documents[-1], finishedAt=200.0, futureField="do not overwrite")
        assert second.call("history.append", scope="local", project="film", record=duplicate) == {
            "record": documents[-1],
            "project": "film",
            "created": False,
        }
        page = second.call("history.list", scope="local", project="film", limit=1)
        assert page["items"] == [documents[-1]]
        assert page["cursor"] is not None
        assert first.call("history.list", scope="local", project="film", after=[30.0, "z"])[
            "items"
        ] == [documents[1], documents[0]]
        for arguments in (
            {"after": [30.0, "z"], "cursor": page["cursor"]},
            {"after": [True, "z"]},
            {"after": [30.0]},
        ):
            with pytest.raises(LibraryError) as invalid:
                first.call("history.list", scope="local", project="film", **arguments)
            assert invalid.value.code == "invalid"
        with pytest.raises(LibraryError) as changed:
            first.call("history.list", scope="private", project="film", cursor=page["cursor"])
        assert changed.value.code == "invalid"
        page = first.call(
            "history.list", scope="local", project="film", cursor=page["cursor"], limit=1
        )
        assert page["items"] == [documents[1]]
        page = second.call(
            "history.list", scope="local", project="film", cursor=page["cursor"], limit=1
        )
        assert page == {"items": [documents[0]], "cursor": None}
        assert first.call("history.get", scope="private", id="z") == {"record": None}
        with pytest.raises(LibraryError):
            second.call(
                "history.append",
                scope="private",
                project="film",
                record=record("z", 9.0, "private"),
            )
        assert second.call("history.delete", scope="private", id="z") == {"deleted": False}
        assert second.call("history.delete", scope="local", id="z") == {"deleted": True}
        assert first.call(
            "history.append", scope="local", project="film", record=documents[-1]
        ) == {
            "record": None,
            "project": "film",
            "created": False,
        }
        assert first.call("history.clear", scope="local", project="film", before=30.0) == {
            "deleted": 1
        }
        assert first.call("history.list", scope="local", project="film") == {
            "items": [documents[1]],
            "cursor": None,
        }
        assert second.call("history.get", scope="private", id="private")["record"] == private
        changes = cast("list[dict[str, object]]", first.call("changes")["items"])
        history_changes = [item for item in changes if item["kind"] == "history"]
        assert len(history_changes) == 7
        with pytest.raises(LibraryError) as reassigned:
            second.call("history.append", scope="local", record=documents[1])
        assert reassigned.value.code == "invalid"
        contenders = [record("concurrent", 40.0), record("concurrent", 50.0)]
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = [
                pool.submit(client.call, "history.append", scope="local", record=document)
                for client, document in zip((first, second), contenders, strict=True)
            ]
            results = [future.result() for future in pending]
        assert [result["created"] for result in results].count(True) == 1
        assert results[0]["record"] == results[1]["record"]
        assert results[0]["record"] in contenders
        assert first.call("history.get", scope="local", id="concurrent") == {
            "record": results[0]["record"],
            "project": "default",
        }
        first.call(
            "history.append", scope="local", record=dict(record("unknown", 60.0), assetDigests=None)
        )
        with pytest.raises(LibraryError) as unknown:
            second.call("references.acquire")
        assert unknown.value.code == "references"
        assert first.call("history.delete", scope="local", id="unknown")["deleted"]
        with pytest.raises(LibraryError):
            first.call("references.acquire", limit=0)
        reserved = first.call("references.acquire", limit=1)
        assert reserved["digests"] == ["blake3:" + "ab" * 32]
        assert reserved["cursor"] == "blake3:" + "ab" * 32
        for client, operation, arguments in (
            (first, "history.clear", {"scope": "local"}),
            (second, "references.acquire", {}),
            (second, "references.release", {"lease": reserved["lease"]}),
        ):
            with pytest.raises(LibraryError):
                client.call(operation, **arguments)
        page = first.call(
            "references.page", lease=reserved["lease"], cursor=reserved["cursor"], limit=1
        )
        assert page["digests"] == ["blake3:" + "cd" * 32]
        assert page["cursor"] == "blake3:" + "cd" * 32
        page = first.call(
            "references.page", lease=reserved["lease"], cursor=page["cursor"], limit=1
        )
        assert page["digests"] == ["blake3:" + "ef" * 32]
        assert page["cursor"] is None
        second.call("projects.register", id="unrelated")
        with closing(sqlite3.connect(tmp_path / "records.sqlite", timeout=0)) as database:
            database.execute("BEGIN IMMEDIATE")
            database.rollback()
        assert second.call("history.get", scope="local", id="y")["record"] == documents[1]
        assert second.call("history.get", scope="local", id="default")["project"] == "default"
        later = dict(record("later", 88.0), assetDigests=["blake3:" + "12" * 32])
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending_write = pool.submit(second.call, "history.append", scope="local", record=later)
            try:
                with pytest.raises(TimeoutError):
                    pending_write.result(timeout=0.1)
                assert first.call("history.get", scope="local", id="later") == {"record": None}
            finally:
                first.call("references.release", lease=reserved["lease"])
            assert pending_write.result(timeout=5) == {
                "record": later,
                "project": "default",
                "created": True,
            }
        with pytest.raises(LibraryError):
            first.call("references.page", lease=reserved["lease"])
        first.call("references.acquire")
        os.kill(first.pid, signal.SIGTERM)
        first.close()
        reserved = second.call("references.acquire")
        second.call("references.release", lease=reserved["lease"])
    with LibraryClient(tmp_path) as reopened:
        assert reopened.call("history.get", scope="local", id="z") == {"record": None}
        assert reopened.call("history.get", scope="local", id="y")["record"] == documents[1]
        assert reopened.call("history.get", scope="local", id="default")["project"] == "default"
        assert reopened.call("history.clear", scope="private", before=100.0) == {"deleted": 0}
        assert reopened.call("history.clear", scope="private") == {"deleted": 1}
    assert not (tmp_path / "library.sqlite").exists()
    assert not (tmp_path / "history.sqlite").exists()


def test_engine_reference_manifests_are_owned_revisioned_and_survive_restart(
    tmp_path: Path,
) -> None:
    first_id, second_id = "ab" * 16, "cd" * 16
    first_digest, second_digest, later_digest = [
        "blake3:" + value * 32 for value in ("12", "34", "56")
    ]
    with (
        LibraryClient(tmp_path) as first,
        LibraryClient(tmp_path) as second,
        LibraryClient(tmp_path) as collector,
    ):
        with pytest.raises(LibraryError):
            first.call("references.engine.attach", engine="../other")
        initial = first.call("references.engine.attach", engine=first_id)["manifest"]
        assert initial == {"engine": first_id, "revision": 1, "digests": [], "complete": False}
        with pytest.raises(LibraryError) as duplicate:
            second.call("references.engine.attach", engine=first_id)
        assert duplicate.value.code == "busy"
        with pytest.raises(LibraryError) as incomplete:
            collector.call("references.acquire")
        assert incomplete.value.code == "references"
        first_manifest = first.call(
            "references.engine.patch",
            revision=1,
            patch={"digests": [first_digest, first_digest], "complete": True},
        )["manifest"]
        assert first_manifest == {
            "engine": first_id,
            "revision": 2,
            "digests": [first_digest],
            "complete": True,
        }
        second.call("references.engine.attach", engine=second_id)
        assert first.call("references.engine.get")["manifest"] == first_manifest
        with pytest.raises(LibraryError):
            collector.call("references.acquire")
        second_manifest = second.call(
            "references.engine.patch",
            revision=1,
            patch={"digests": [second_digest], "complete": True},
        )["manifest"]
        with pytest.raises(LibraryError) as stale:
            first.call("references.engine.patch", revision=1, patch={"digests": []})
        assert stale.value.code == "stale"
        for arguments in (
            {"revision": 2, "engine": second_id, "patch": {"digests": []}},
            {"revision": 2, "patch": {"digests": ["broken"]}},
            {"revision": 2, "patch": {"complete": 1}},
        ):
            with pytest.raises(LibraryError):
                first.call("references.engine.patch", **arguments)
        assert first.call("references.engine.get")["manifest"] == first_manifest
        reserved = collector.call("references.acquire")
        assert reserved["digests"] == [first_digest, second_digest]
        assert first.call("references.engine.get")["manifest"] == first_manifest
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(
                first.call, "references.engine.patch", revision=2, patch={"digests": [later_digest]}
            )
            try:
                with pytest.raises(TimeoutError):
                    pending.result(timeout=0.1)
                assert second.call("references.engine.get")["manifest"] == second_manifest
            finally:
                collector.call("references.release", lease=reserved["lease"])
            assert pending.result(timeout=5)["manifest"] == {
                "engine": first_id,
                "revision": 3,
                "digests": [later_digest],
                "complete": True,
            }
        os.kill(first.pid, signal.SIGTERM)
        first.close()
        with closing(sqlite3.connect(tmp_path / "records.sqlite")) as database:
            database.execute(
                "UPDATE engine_reference_manifests "
                "SET document=json_set(document,'$.futureField','kept') WHERE engine=?",
                (first_id,),
            )
            database.commit()
        reserved = collector.call("references.acquire")
        assert reserved["digests"] == [second_digest, later_digest]
        collector.call("references.release", lease=reserved["lease"])
        with LibraryClient(tmp_path) as restarted:
            carried = restarted.call("references.engine.attach", engine=first_id)["manifest"]
            assert carried == {
                "engine": first_id,
                "revision": 4,
                "digests": [later_digest],
                "complete": False,
                "futureField": "kept",
            }
            assert second.call("references.engine.get")["manifest"] == second_manifest
            with pytest.raises(LibraryError):
                collector.call("references.acquire")
            restarted.call(
                "references.engine.patch", revision=4, patch={"digests": [], "complete": True}
            )
            reserved = collector.call("references.acquire")
            assert reserved["digests"] == [second_digest]
            collector.call("references.release", lease=reserved["lease"])
            assert restarted.call("references.engine.get")["manifest"] == {
                "engine": first_id,
                "revision": 5,
                "digests": [],
                "complete": True,
                "futureField": "kept",
            }
    assert not (tmp_path / "library.sqlite").exists()
    assert not (tmp_path / "history.sqlite").exists()


def test_content_manifests_are_immutable_and_reserve_only_reachable_dependencies(
    tmp_path: Path,
) -> None:
    parent, child, grandchild, output, orphan, unrelated, key = [
        "blake3:" + value * 32 for value in ("12", "34", "56", "78", "9a", "bc", "de")
    ]
    declared = [{"digest": child, "futureField": {"keep": True}}]
    rendered = {"source": parent, "output": output, "futureField": {"keep": True}}
    with LibraryClient(tmp_path) as first, LibraryClient(tmp_path) as second:
        assert {"dependencies", "derivations"} <= set(
            cast("list[str]", first.hello["capabilities"])
        )
        assert first.call("dependencies.get", digest=parent) == {"dependencies": None}
        assert first.call("derivations.get", digest=key) == {"manifest": None}
        assert first.call("dependencies.put", digest=parent, dependencies=declared) == {
            "created": True
        }
        assert second.call("dependencies.put", digest=parent, dependencies=declared) == {
            "created": False
        }
        assert second.call("dependencies.get", digest=parent) == {"dependencies": declared}
        for digest, documents in (
            (child, [{"digest": grandchild}]),
            (orphan, [{"digest": unrelated}]),
        ):
            first.call("dependencies.put", digest=digest, dependencies=documents)
        with ThreadPoolExecutor(max_workers=2) as pool:
            writes = [
                pool.submit(
                    client.call,
                    "dependencies.put",
                    digest=grandchild,
                    dependencies=[{"digest": parent}],
                )
                for client in (first, second)
            ]
            assert sorted(cast(bool, future.result()["created"]) for future in writes) == [
                False,
                True,
            ]
        assert first.call(
            "derivations.put", digest=key, manifest=rendered, digests=[output, parent, output]
        ) == {"created": True}
        assert second.call(
            "derivations.put", digest=key, manifest=rendered, digests=[parent, output]
        ) == {"created": False}
        for operation, arguments in (
            ("dependencies.put", {"digest": parent, "dependencies": []}),
            ("derivations.put", {"digest": key, "manifest": {}, "digests": [parent, output]}),
            (
                "derivations.put",
                {
                    "digest": key,
                    "manifest": dict(rendered, futureField={"keep": 1}),
                    "digests": [parent, output],
                },
            ),
            ("derivations.put", {"digest": key, "manifest": rendered, "digests": [output]}),
        ):
            with pytest.raises(LibraryError) as conflict:
                first.call(operation, **arguments)
            assert conflict.value.code == "conflict"
        for operation, arguments in (
            ("dependencies.put", {"digest": unrelated, "dependencies": [{"digest": "bad"}]}),
            ("dependencies.put", {"digest": unrelated, "dependencies": [None]}),
            ("derivations.put", {"digest": unrelated, "manifest": rendered, "digests": None}),
            ("derivations.put", {"digest": unrelated, "manifest": rendered, "digests": [True]}),
        ):
            with pytest.raises(LibraryError) as invalid:
                first.call(operation, **arguments)
            assert invalid.value.code == "invalid"
        assert first.call("dependencies.get", digest=unrelated) == {"dependencies": None}
        assert first.call("derivations.get", digest=unrelated) == {"manifest": None}
        reserved = first.call("references.acquire")
        assert reserved["digests"] == [parent, child, grandchild, output]
        assert first.call("dependencies.get", digest=parent) == {"dependencies": declared}
        assert first.call("derivations.get", digest=key) == {"manifest": rendered}
        with closing(sqlite3.connect(tmp_path / "records.sqlite", timeout=0)) as database:
            database.execute("BEGIN IMMEDIATE")
            database.rollback()
        with pytest.raises(LibraryError) as held:
            first.call("dependencies.put", digest=output, dependencies=[{"digest": orphan}])
        assert held.value.code == "busy"
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(
                second.call, "dependencies.put", digest=output, dependencies=[{"digest": orphan}]
            )
            try:
                with pytest.raises(TimeoutError):
                    pending.result(timeout=0.1)
                assert first.call("dependencies.get", digest=output) == {"dependencies": None}
            finally:
                first.call("references.release", lease=reserved["lease"])
            assert pending.result(timeout=5) == {"created": True}
        reserved = first.call("references.acquire")
        assert reserved["digests"] == [parent, child, grandchild, output, orphan, unrelated]
        first.call("references.release", lease=reserved["lease"])
    with closing(sqlite3.connect(tmp_path / "records.sqlite")) as database:
        database.execute(
            "UPDATE derivation_manifests SET asset_digests=NULL WHERE digest=?", (key,)
        )
        database.commit()
    with LibraryClient(tmp_path) as reopened:
        assert reopened.call("dependencies.get", digest=parent) == {"dependencies": declared}
        assert reopened.call("derivations.get", digest=key) == {"manifest": rendered}
        with pytest.raises(LibraryError) as unknown:
            reopened.call("references.acquire")
        assert unknown.value.code == "references"
        with pytest.raises(LibraryError):
            reopened.call("derivations.put", digest=key, manifest={}, digests=[])
        with pytest.raises(LibraryError):
            reopened.call("references.acquire")
        assert reopened.call(
            "derivations.put", digest=key, manifest=rendered, digests=[parent, output]
        ) == {"created": False}
        reserved = reopened.call("references.acquire")
        assert reserved["digests"] == [parent, child, grandchild, output, orphan, unrelated]
        reopened.call("references.release", lease=reserved["lease"])
    assert not (tmp_path / "library.sqlite").exists()


def test_saved_records_keep_projects_revisions_unknown_fields_and_reference_barriers(
    tmp_path: Path,
) -> None:
    workflow_type = "application/x-dinkster-workflow+json"
    original, named, private, replacement, dependency = [
        "blake3:" + value * 32 for value in ("12", "34", "56", "78", "9a")
    ]
    facts = {
        "digest": original,
        "packs": [{"pack": "film-pack"}],
        "models": [],
        "machinePaths": False,
    }
    with LibraryClient(tmp_path) as first, LibraryClient(tmp_path) as second:
        first.call("projects.register", id="film")
        with pytest.raises(LibraryError) as invalid_create:
            first.call(
                "records.create",
                scope="local",
                record={"name": "Invalid", "digest": "bad", "mediaType": workflow_type},
            )
        assert invalid_create.value.code == "invalid"
        with pytest.raises(LibraryError) as missing_project:
            first.call(
                "records.create",
                scope="local",
                project="absent",
                record={"name": "Invalid", "digest": original, "mediaType": workflow_type},
            )
        assert missing_project.value.code == "missing"
        created = cast(
            "dict[str, object]",
            first.call(
                "records.create",
                scope="local",
                project="film",
                record={
                    "name": " First film ",
                    "digest": original,
                    "mediaType": workflow_type,
                    "labels": [" draft ", "draft"],
                    "folder": "projects/film",
                },
            )["record"],
        )
        identifier = created["id"]
        assert created["name"] == "First film"
        assert created["labels"] == ["draft"]
        assert created["revision"] == 1
        assert created["project"] == "film"
        asset = cast(
            "dict[str, object]",
            second.call(
                "records.create",
                scope="local",
                project="film",
                record={"name": "Named asset", "digest": named, "mediaType": "text/plain"},
            )["record"],
        )
        hidden = cast(
            "dict[str, object]",
            second.call(
                "records.create",
                scope="private",
                record={"name": "Private", "digest": private, "mediaType": "text/plain"},
            )["record"],
        )
        assert first.call("records.get", scope="private", id=identifier) == {"record": None}
        assert first.call(
            "records.patch", scope="private", id=identifier, revision=1, patch={"name": "no"}
        ) == {"record": None}
        assert first.call("records.delete", scope="private", id=identifier, revision=1) == {
            "deleted": False
        }
        page = first.call("records.list", scope="local", project="film", limit=1)
        assert page["records"] == [asset]
        assert page["cursor"] is not None
        assert first.call("records.list", scope="local", project="film", cursor=page["cursor"])[
            "records"
        ] == [created]
        assert first.call(
            "records.list",
            scope="local",
            project="film",
            q="FIRST",
            label="draft",
            mediaType=workflow_type,
        )["records"] == [created]
        with pytest.raises(LibraryError):
            first.call("records.list", scope="local", project="default", cursor=page["cursor"])
        assert first.call("projects.needs", id="film", includeWorkflows=True)["workflows"] == [
            {key: created[key] for key in ("scope", "id", "digest", "name")}
        ]
        assert first.call("workflows.get", scope="local", id=identifier)["project"] == "film"
        with pytest.raises(LibraryError):
            first.call(
                "workflows.record", scope="local", id=identifier, revision=0, project="default"
            )
        assert (
            first.call("workflows.record", scope="local", id=identifier, revision=0, facts=facts)[
                "facts"
            ]
            == facts
        )
        assert first.call("projects.needs", id="film")["packs"] == [{"pack": "film-pack"}]
        with closing(sqlite3.connect(tmp_path / "records.sqlite")) as database:
            database.execute(
                "UPDATE saved_records SET document=json_set(document,'$.futureField','kept') "
                "WHERE id=?",
                (identifier,),
            )
            database.commit()
        for patch in (
            {"project": "default"},
            {"id": "other"},
            {"labels": [True]},
            {"name": ""},
            {"name": "   "},
            {"labels": ["   "]},
            {"mediaType": "   "},
            {"folder": "../outside"},
            {"digest": "bad"},
        ):
            with pytest.raises(LibraryError):
                first.call("records.patch", scope="local", id=identifier, revision=1, patch=patch)
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = [
                pool.submit(
                    client.call,
                    "records.patch",
                    scope="local",
                    id=identifier,
                    revision=1,
                    patch={"name": name},
                )
                for client, name in ((first, "One"), (second, "Two"))
            ]
            successful: list[dict[str, object]] = []
            failures: list[str] = []
            for future in pending:
                try:
                    successful.append(cast("dict[str, object]", future.result()["record"]))
                except LibraryError as error:
                    failures.append(error.code)
            assert len(successful) == 1
            assert failures == ["stale"]
        updated = successful[0]
        assert updated["revision"] == 2
        assert updated["futureField"] == "kept"
        assert updated["created"] == created["created"]
        assert cast(float, updated["modified"]) > cast(float, created["modified"])
        assert second.call("records.get", scope="local", id=identifier)["record"] == updated
        first.call("dependencies.put", digest=original, dependencies=[{"digest": dependency}])
        reserved = first.call("references.acquire")
        assert reserved["digests"] == [original, named, private, dependency]
        with ThreadPoolExecutor(max_workers=1) as pool:
            write = pool.submit(
                second.call,
                "records.patch",
                scope="local",
                id=identifier,
                revision=2,
                patch={"digest": replacement, "folder": ""},
            )
            try:
                with pytest.raises(TimeoutError):
                    write.result(timeout=0.1)
                assert first.call("records.get", scope="local", id=identifier)["record"] == updated
            finally:
                first.call("references.release", lease=reserved["lease"])
            updated = cast("dict[str, object]", write.result(timeout=5)["record"])
        assert updated["digest"] == replacement
        assert updated["futureField"] == "kept"
        assert "folder" not in updated
        assert first.call("workflows.read", scope="local", id=identifier)["record"] == updated
        assert first.call("workflows.get", scope="local", id=identifier)["facts"] is None
        assert first.call("projects.needs", id="film")["unknown"] == [
            {"scope": "local", "id": identifier}
        ]
        reserved = first.call("references.acquire")
        assert reserved["digests"] == [named, private, replacement]
        first.call("references.release", lease=reserved["lease"])
        with pytest.raises(LibraryError) as stale:
            second.call("records.delete", scope="local", id=identifier, revision=2)
        assert stale.value.code == "stale"
        assert second.call("records.delete", scope="local", id=identifier, revision=3) == {
            "deleted": True
        }
        assert first.call("records.get", scope="local", id=identifier) == {"record": None}
        assert first.call("projects.needs", id="film", includeWorkflows=True)["workflows"] == []
        assert first.call("records.list", scope="local", project="film")["records"] == [asset]
        reserved = first.call("references.acquire")
        assert reserved["digests"] == [named, private]
        first.call("references.release", lease=reserved["lease"])
    with LibraryClient(tmp_path) as reopened:
        assert reopened.call("records.get", scope="local", id=identifier) == {"record": None}
        assert reopened.call("records.get", scope="private", id=hidden["id"])["record"] == hidden
    assert not (tmp_path / "library.sqlite").exists()


def test_catalog_ingest_is_immutable_scoped_read_only_and_reserves_references(
    tmp_path: Path,
) -> None:
    older, newer, private, arriving = ["blake3:" + value * 32 for value in ("12", "34", "56", "78")]

    def observation(identifier: str, digest: str, created: float, name: str) -> dict[str, object]:
        return {
            "id": identifier * 32,
            "asset": {
                "digest": digest,
                "name": name,
                "size": 13,
                "mediaType": "image/png",
                "virtualPath": "images/" + name,
                "futureAsset": {"keep": True},
            },
            "originalName": name,
            "kind": "image/raster",
            "origin": {"kind": "upload"},
            "project": "film",
            "created": created,
            "futureField": True,
        }

    first_name = observation("a", older, 10.0, "First.png")
    alias = observation("b", older, 30.0, "Alias.png")
    latest = observation("c", newer, 30.0, "Newest.png")
    downloaded = dict(
        alias, origin={"kind": "download", "url": "https://example.test/picture?discard=1#part"}
    )
    cleaned_alias = dict(
        downloaded, origin={"kind": "download", "url": "https://example.test/picture"}
    )
    with LibraryClient(tmp_path) as first, LibraryClient(tmp_path) as second:
        assert "assets.record" in cast("list[str]", first.hello["capabilities"])
        first.call("projects.register", id="film")
        with ThreadPoolExecutor(max_workers=2) as pool:
            writes = [
                pool.submit(client.call, "assets.record", scope="local", observation=first_name)
                for client in (first, second)
            ]
            assert sorted(cast(bool, write.result()["created"]) for write in writes) == [
                False,
                True,
            ]
        assert first.call("assets.record", scope="local", observation=downloaded) == {
            "created": True,
            "observation": cleaned_alias,
        }
        first.call("assets.record", scope="local", observation=latest)
        for scope, changed in (
            ("private", first_name),
            ("local", dict(first_name, futureField=1)),
            ("local", dict(first_name, originalName="Changed.png")),
        ):
            with pytest.raises(LibraryError) as conflict:
                second.call("assets.record", scope=scope, observation=changed)
            assert conflict.value.code == "conflict"
        for invalid in (
            dict(first_name, id="bad"),
            dict(first_name, created=True),
            dict(first_name, created=-1),
            dict(first_name, created=10**400),
            dict(first_name, originalName="../outside"),
            dict(first_name, kind="Image"),
            dict(first_name, origin={"kind": "download", "url": "file:///not-allowed"}),
            dict(first_name, origin={"kind": "upload", "url": "https://example.test"}),
            dict(first_name, asset=dict(cast(dict[str, object], first_name["asset"]), size=True)),
            dict(first_name, asset=dict(cast(dict[str, object], first_name["asset"]), size=2**63)),
            dict(
                first_name,
                asset=dict(cast(dict[str, object], first_name["asset"]), virtualPath="../outside"),
            ),
        ):
            with pytest.raises(LibraryError) as error:
                second.call("assets.record", scope="local", observation=invalid)
            assert error.value.code == "invalid"
        with pytest.raises(LibraryError) as missing:
            second.call(
                "assets.record", scope="local", observation=dict(first_name, project="absent")
            )
        assert missing.value.code == "missing"
        second.call(
            "assets.record",
            scope="private",
            observation=observation("d", private, 5.0, "Private.png"),
        )
        second.call(
            "assets.record",
            scope="local",
            observation=dict(first_name, id="e" * 32, project="default"),
        )
        with closing(sqlite3.connect(tmp_path / "records.sqlite")) as database:
            before = list(database.iterdump())
        page = first.call("assets.list", scope="local", project="film", limit=1)
        assert cast(list[dict[str, object]], page["assets"])[0]["observations"] == [latest]
        assert page["cursor"] is not None
        rest = first.call("assets.list", scope="local", project="film", cursor=page["cursor"])
        assert rest["cursor"] is None
        group = cast(list[dict[str, object]], rest["assets"])[0]
        assert group["observations"] == [first_name, cleaned_alias]
        assert group["aliases"] == ["First.png", "Alias.png"]
        assert group["asset"] == cleaned_alias["asset"]
        filtered = first.call(
            "assets.list",
            scope="local",
            project="film",
            q="FIRST",
            origin="upload",
            since=10,
            until=10,
        )
        assert cast(list[dict[str, object]], filtered["assets"])[0]["observations"] == [
            first_name,
            cleaned_alias,
        ]
        assert (
            len(
                cast(
                    list[object],
                    first.call("assets.list", scope="private", project="film")["assets"],
                )
            )
            == 1
        )
        assert len(cast(list[object], first.call("assets.list", scope="local")["assets"])) == 1
        with pytest.raises(LibraryError):
            first.call("assets.list", scope="private", project="film", cursor=page["cursor"])
        with closing(sqlite3.connect(tmp_path / "records.sqlite")) as database:
            assert list(database.iterdump()) == before
        held = first.call("references.acquire")
        assert held["digests"] == [older, newer, private]
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(
                second.call,
                "assets.record",
                scope="private",
                observation=observation("f", arriving, 1.0, "Arriving.png"),
            )
            try:
                with pytest.raises(TimeoutError):
                    pending.result(timeout=0.1)
            finally:
                first.call("references.release", lease=held["lease"])
            assert pending.result(timeout=5)["created"] is True
        held = first.call("references.acquire")
        assert held["digests"] == [older, newer, private, arriving]
        first.call("references.release", lease=held["lease"])
        assert first.call("records.list", scope="local")["records"] == []
    with LibraryClient(tmp_path) as reopened:
        assert cast(
            list[dict[str, object]],
            reopened.call("assets.list", scope="local", project="film", q="First")["assets"],
        )[0]["observations"] == [first_name, cleaned_alias]
    assert not (tmp_path / "library.sqlite").exists()


def test_media_grants_keep_exact_scope_immutable_facts_and_reference_barriers(
    tmp_path: Path,
) -> None:
    image, latent, later = ["blake3:" + value * 32 for value in ("12", "34", "56")]
    grant = {
        "digest": image,
        "kind": "media/image",
        "mediaType": "image/png",
        "extension": "png",
        "byteSize": 73,
        "futureField": {"keep": True},
    }
    stored = dict(grant, scope="local")
    latent_grant = {
        "digest": latent,
        "kind": "data/latent",
        "mediaType": "application/x-comfy-latent",
        "extension": "latent",
        "byteSize": 109,
    }
    with LibraryClient(tmp_path) as first, LibraryClient(tmp_path) as second:
        assert "grants" in cast("list[str]", first.hello["capabilities"])
        with ThreadPoolExecutor(max_workers=2) as pool:
            writes = [
                pool.submit(client.call, "grants.put", scope="local", grant=grant)
                for client in (first, second)
            ]
            assert sorted(cast(bool, write.result()["created"]) for write in writes) == [
                False,
                True,
            ]
        assert first.call("grants.get", scope="local", digest=image, kind="media/image") == {
            "grant": stored
        }
        assert first.call("grants.get", scope="private", digest=image, kind="media/image") == {
            "grant": None
        }
        assert first.call("grants.get", scope="local", digest=image, kind="media/audio") == {
            "grant": None
        }
        for changes in (
            {"byteSize": 74},
            {"mediaType": "image/jpeg", "extension": "jpg"},
            {"futureField": {"keep": 1}},
        ):
            with pytest.raises(LibraryError) as conflict:
                first.call("grants.put", scope="local", grant=dict(grant, **changes))
            assert conflict.value.code == "conflict"
        for changes in (
            {"digest": "bad"},
            {"kind": "Image"},
            {"scope": "private"},
            {"mediaType": "image/jpeg"},
            {"byteSize": True},
            {"byteSize": 0},
            {"byteSize": 2**63},
        ):
            with pytest.raises(LibraryError) as invalid:
                first.call("grants.put", scope="local", grant=dict(grant, **changes))
            assert invalid.value.code == "invalid"
        assert second.call("grants.put", scope="private", grant=latent_grant)["created"] is True
        assert first.call("assets.list", scope="local")["assets"] == []
        first.call("dependencies.put", digest=latent, dependencies=[{"digest": later}])
        held = first.call("references.acquire")
        assert held["digests"] == [image, latent, later]
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(second.call, "grants.put", scope="private", grant=grant)
            try:
                with pytest.raises(TimeoutError):
                    pending.result(timeout=0.1)
                assert (
                    first.call("grants.get", scope="local", digest=image, kind="media/image")[
                        "grant"
                    ]
                    == stored
                )
            finally:
                first.call("references.release", lease=held["lease"])
            assert pending.result(timeout=5)["created"] is True
    with LibraryClient(tmp_path) as reopened:
        assert reopened.call("grants.get", scope="local", digest=image, kind="media/image") == {
            "grant": stored
        }
        assert reopened.call("grants.get", scope="private", digest=latent, kind="data/latent") == {
            "grant": dict(latent_grant, scope="private")
        }
    assert not (tmp_path / "library.sqlite").exists()


def test_result_publications_keep_engine_scope_immutable_retries_and_reference_roots(
    tmp_path: Path,
) -> None:
    first_id, second_id, token = "12" * 16, "34" * 16, "56" * 16
    source, other, derived, child, later, document = [
        "blake3:" + value * 32 for value in ("12", "34", "56", "78", "9a", "bc")
    ]
    offer: dict[str, object] = {
        "token": token,
        "jobRef": "run-first",
        "attemptId": 2,
        "nodeId": "node-first",
        "outputId": "image",
        "publication": 3,
        "digests": [source],
        "complete": True,
        "futureField": {"keep": True},
    }
    rendition = {"documentDigest": document, "digests": [derived], "futureField": {"keep": True}}
    with (
        LibraryClient(tmp_path) as first,
        LibraryClient(tmp_path) as second,
        LibraryClient(tmp_path) as collector,
    ):
        assert {"results.offers", "results.renditions"} <= set(
            cast("list[str]", first.hello["capabilities"])
        )
        with pytest.raises(LibraryError) as unattached:
            first.call("results.offers.put", scope="local", offer=offer)
        assert unattached.value.code == "invalid"
        for client, identifier in ((first, first_id), (second, second_id)):
            client.call("references.engine.attach", engine=identifier)
            client.call("references.engine.patch", revision=1, patch={"complete": True})
        stored = dict(offer, scope="local", engine=first_id)
        assert first.call("results.offers.put", scope="local", offer=offer) == {
            "offer": stored,
            "created": True,
        }
        assert first.call(
            "results.offers.put", scope="local", offer=dict(offer, digests=[source, source])
        ) == {
            "offer": stored,
            "created": False,
        }
        assert second.call("results.offers.get", scope="local", token=token) == {"offer": None}
        assert first.call("results.offers.get", scope="private", token=token) == {"offer": None}
        for fields in ({"digests": [other]}, {"futureField": {"keep": 1}}):
            with pytest.raises(LibraryError) as conflict:
                first.call("results.offers.put", scope="local", offer=dict(offer, **fields))
            assert conflict.value.code == "conflict"
        with pytest.raises(LibraryError) as reassigned:
            first.call("results.offers.put", scope="private", offer=offer)
        assert reassigned.value.code == "conflict"
        for fields in (
            {"engine": second_id},
            {"scope": "private"},
            {"token": "invalid"},
            {"attemptId": True},
            {"publication": 0},
            {"digests": ["bad"]},
            {"complete": 1},
            {"artifact": {}},
            {"outputId": " "},
        ):
            with pytest.raises(LibraryError) as invalid:
                first.call("results.offers.put", scope="local", offer=dict(offer, **fields))
            assert invalid.value.code == "invalid"
        assert first.call("results.offers.get", scope="local", token=token) == {"offer": stored}
        assert second.call(
            "results.renditions.put",
            scope="local",
            token=token,
            cacheKey="view",
            rendition=rendition,
        ) == {"rendition": None, "created": False}
        second_offer = dict(offer, jobRef="run-second", digests=[other])
        del second_offer["outputId"]
        second_offer["artifact"] = {
            "nodeId": "node-first",
            "digest": other,
            "index": 0,
            "futureField": True,
        }
        second.call("results.offers.put", scope="private", offer=second_offer)
        for scope in ("local", "private"):
            assert second.call(
                "results.renditions.get", scope=scope, token=token, cacheKey="view"
            ) == {"rendition": None}
        arguments = {"scope": "local", "token": token, "cacheKey": "view"}
        assert first.call("results.renditions.put", **arguments, rendition=rendition) == {
            "rendition": rendition,
            "created": True,
        }
        assert first.call("results.renditions.put", **arguments, rendition=rendition) == {
            "rendition": rendition,
            "created": False,
        }
        assert second.call("results.renditions.get", **arguments) == {"rendition": None}
        assert first.call("results.renditions.get", **dict(arguments, scope="private")) == {
            "rendition": None
        }
        for fields in ({"digests": [later]}, {"futureField": {"keep": 1}}):
            with pytest.raises(LibraryError) as conflict:
                first.call(
                    "results.renditions.put", **arguments, rendition=dict(rendition, **fields)
                )
            assert conflict.value.code == "conflict"
        first.call("dependencies.put", digest=derived, dependencies=[{"digest": child}])
        reserved = collector.call("references.acquire")
        assert reserved["digests"] == [source, other, derived, child]
        with closing(sqlite3.connect(tmp_path / "records.sqlite", timeout=0)) as db:
            db.execute("BEGIN IMMEDIATE")
            db.rollback()
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = [
                pool.submit(
                    first.call,
                    "results.renditions.put",
                    scope="local",
                    token=token,
                    cacheKey="next",
                    rendition=dict(rendition, digests=[later]),
                ),
                pool.submit(
                    second.call,
                    "results.offers.put",
                    scope="private",
                    offer=dict(second_offer, token="78" * 16, digests=[later]),
                ),
            ]
            try:
                for write in pending:
                    with pytest.raises(TimeoutError):
                        write.result(timeout=0.1)
                assert collector.call("projects.get", id="default")["project"] is not None
            finally:
                collector.call("references.release", lease=reserved["lease"])
            assert all(write.result(timeout=5)["created"] is True for write in pending)
        os.kill(first.pid, signal.SIGTERM)
        first.close()
        with LibraryClient(tmp_path) as restarted:
            manifest = cast(
                "dict[str, object]",
                restarted.call("references.engine.attach", engine=first_id)["manifest"],
            )
            assert restarted.call("results.offers.get", scope="local", token=token) == {
                "offer": stored
            }
            assert restarted.call("results.renditions.get", **arguments) == {"rendition": rendition}
            assert second.call("results.offers.get", scope="private", token=token)["offer"] == dict(
                second_offer, scope="private", engine=second_id
            )
            restarted.call(
                "references.engine.patch", revision=manifest["revision"], patch={"complete": True}
            )
            reserved = collector.call("references.acquire")
            assert reserved["digests"] == [source, other, derived, child, later]
            collector.call("references.release", lease=reserved["lease"])
            incomplete = dict(offer, token="ab" * 16, complete=False)
            restarted.call("results.offers.put", scope="local", offer=incomplete)
            with pytest.raises(LibraryError) as blocked:
                collector.call("references.acquire")
            assert blocked.value.code == "references"
    assert not (tmp_path / "library.sqlite").exists()


@pytest.mark.parametrize("state", ["completed", "failed", "cancelled"])
def test_terminal_results_seal_exact_engine_scoped_publications(tmp_path: Path, state: str) -> None:
    first_id, second_id = "12" * 16, "34" * 16
    tokens = [value * 16 for value in ("12", "34", "56", "78")]
    digest = "blake3:" + "9a" * 32
    output: dict[str, object] = {
        "token": tokens[0],
        "jobRef": "run",
        "attemptId": 2,
        "nodeId": "node",
        "outputId": "image",
        "publication": 3,
        "digests": [digest],
        "complete": True,
        "futureField": {"kept": True},
    }
    artifact = dict(output, token=tokens[1], artifact={"index": 0, "futureField": True})
    del artifact["outputId"]
    identity = {"workflow": "saved", "futureIdentity": {"kept": True}}
    arguments: dict[str, object] = {
        "scope": "private",
        "jobRef": "run",
        "attemptId": 2,
        "tokens": tokens[:2],
        "state": state,
        "identity": identity,
        "artifactsComplete": True,
    }
    with LibraryClient(tmp_path) as first, LibraryClient(tmp_path) as second:
        assert "results.terminal" in cast("list[str]", first.hello["capabilities"])
        for helper, engine in ((first, first_id), (second, second_id)):
            helper.call("references.engine.attach", engine=engine)
            helper.call("references.engine.patch", revision=1, patch={"complete": True})
        first.call("results.offers.put", scope="private", offer=output)
        first.call("results.offers.put", scope="private", offer=artifact)
        first.call("results.offers.put", scope="private", offer=dict(output, token=tokens[2]))
        malformed = (
            {"tokens": [tokens[0], tokens[2]]},
            {"tokens": [tokens[0], tokens[0]]},
            {"tokens": [tokens[3]]},
            {"tokens": ["not-a-token"]},
            {"tokens": "not-a-list"},
            {"attemptId": 3},
            {"jobRef": "another-run"},
            {"attemptId": True},
            {"state": "running"},
            {"artifactsComplete": False},
            {"artifactsComplete": 1},
            {"identity": []},
            {"engine": second_id},
            {"scope": "other"},
        )
        for changed in malformed:
            with pytest.raises(LibraryError):
                first.call("results.terminal.put", **dict(arguments, **changed))
            assert first.call("results.terminal.get", scope="private", jobRef="run") == {
                "result": None
            }
        expected = {
            "engine": first_id,
            "scope": "private",
            "jobRef": "run",
            "attemptId": 2,
            "state": state,
            "identity": identity,
            "resultOffers": {"node": {"image": dict(output, engine=first_id, scope="private")}},
            "artifacts": [dict(artifact, engine=first_id, scope="private")],
            "complete": state == "completed",
        }
        assert first.call("results.terminal.put", **arguments) == {
            "result": expected,
            "created": True,
        }
        assert first.call("results.terminal.put", **arguments) == {
            "result": expected,
            "created": False,
        }
        for changed in (
            {"identity": {"workflow": "different"}},
            {"identity": {"workflow": "saved", "futureIdentity": {"kept": 1}}},
            {"tokens": [tokens[0]]},
        ):
            with pytest.raises(LibraryError) as sealed:
                first.call("results.terminal.put", **dict(arguments, **changed))
            assert sealed.value.code == "conflict"
        assert second.call("results.terminal.get", scope="private", jobRef="run") == {
            "result": None
        }
        assert first.call("results.terminal.get", scope="other", jobRef="run") == {"result": None}
        # Publishing a later offer cannot change an already sealed terminal result.
        first.call(
            "results.offers.put",
            scope="private",
            offer=dict(output, token=tokens[3], publication=4),
        )
        assert (
            first.call("results.terminal.get", scope="private", jobRef="run")["result"] == expected
        )
        with pytest.raises(LibraryError):
            second.call("results.terminal.put", **arguments)
        second.call("results.offers.put", scope="private", offer=dict(output, digests=[]))
        other_arguments = dict(arguments, tokens=[tokens[0]], artifactsComplete=False)
        other = cast(
            "dict[str, object]", second.call("results.terminal.put", **other_arguments)["result"]
        )
        assert other["engine"] == second_id and other["complete"] is False
        assert other["artifacts"] is None
        held = first.call("references.acquire")
        assert held["digests"] == [digest]
        with closing(sqlite3.connect(tmp_path / "records.sqlite", timeout=0)) as db:
            db.execute("BEGIN IMMEDIATE")
            db.rollback()
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(second.call, "results.terminal.put", **other_arguments)
            try:
                with pytest.raises(TimeoutError):
                    pending.result(timeout=0.1)
                assert (
                    first.call("results.terminal.get", scope="private", jobRef="run")["result"]
                    == expected
                )
            finally:
                first.call("references.release", lease=held["lease"])
            assert pending.result(timeout=5)["created"] is False
        ordered_tokens = [value * 16 for value in ("a0", "a1", "a2", "a3")]
        for token, index in zip(ordered_tokens, (1, True, 0, 1), strict=True):
            first.call(
                "results.offers.put",
                scope="private",
                offer=dict(artifact, token=token, jobRef="ordered", artifact={"index": index}),
            )
        ordered_args = dict(arguments, jobRef="ordered")
        for selection in ([ordered_tokens[0]], [ordered_tokens[1]], ordered_tokens[2:][::-1]):
            with pytest.raises(LibraryError) as ambiguous:
                first.call("results.terminal.put", **dict(ordered_args, tokens=selection))
            assert ambiguous.value.code == "invalid"
        ordered = cast(
            "dict[str, object]",
            first.call("results.terminal.put", **dict(ordered_args, tokens=ordered_tokens[2:]))[
                "result"
            ],
        )
        assert ordered["complete"] is (state == "completed")
        assert [
            item["token"] for item in cast("list[dict[str, object]]", ordered["artifacts"])
        ] == ordered_tokens[2:]
        first.call(
            "results.offers.put",
            scope="private",
            offer=dict(output, token="b0" * 16, jobRef="incomplete", complete=False),
        )
        incomplete = cast(
            "dict[str, object]",
            first.call(
                "results.terminal.put",
                **dict(arguments, jobRef="incomplete", tokens=["b0" * 16]),
            )["result"],
        )
        assert incomplete["complete"] is False
    with LibraryClient(tmp_path) as restarted:
        restarted.call("references.engine.attach", engine=first_id)
        assert (
            restarted.call("results.terminal.get", scope="private", jobRef="run")["result"]
            == expected
        )
    assert not (tmp_path / "library.sqlite").exists()


def test_project_folder_preferences_are_shared_patched_and_preserve_extensions(
    tmp_path: Path,
) -> None:
    from dinkster_assets import MountsError

    from dinkster.project_folders import HiddenModelPreference, ProjectFolders

    available = ("input", "output")
    identity = "blake3:" + "ab" * 32
    with LibraryClient(tmp_path) as first, LibraryClient(tmp_path) as second:
        assert "folders.projects" in cast("list[str]", first.hello["capabilities"])
        assert first.call("folders.projects.get") == {"folders": {"revision": 0, "values": {}}}
        start = first.call("changes")["cursor"]
        left = ProjectFolders(first)
        right = ProjectFolders(second)
        left.initialize(available)
        assert right.ordered("alpha", available) == available
        right.update("alpha", ["output", "input"], available=available, output="output")
        assert left.ordered("alpha", available) == ("output", "input")
        assert left.output("alpha", None) == "output"
        left.update("beta", ["input"], available=available, output=None)
        raw = cast("dict[str, object]", second.call("folders.projects.get")["folders"])
        values = cast("dict[str, object]", raw["values"])
        assert values["outputs"] == {"alpha": "output", "beta": None}
        hidden = {
            "name": "old",
            "kind": "model/lora",
            "identities": [identity],
            "extension": {"nullable": None, "keep": [True, 1]},
        }
        duplicate = {**hidden, "extension": {"keep": "second occurrence"}}
        second.call(
            "folders.projects.patch",
            revision=raw["revision"],
            patch={"future": {"keep": True}, "hiddenModels": {"alpha": [hidden, duplicate]}},
        )
        assert left.hidden_models("alpha") == (identity,)
        other = HiddenModelPreference("other", "model/vae", ("blake3:" + "cd" * 32,))
        left.set_model_hidden("alpha", other, True)
        assert right.hidden_models("alpha") == (identity, *other.identities)
        raw = cast("dict[str, object]", first.call("folders.projects.get")["folders"])
        values = cast("dict[str, object]", raw["values"])
        assert values["future"] == {"keep": True}
        assert values["hiddenModels"] == {"alpha": [hidden, duplicate, other.to_wire()]}
        before = first.call("changes")
        with pytest.raises(MountsError):
            right.update("alpha", ["input"], available=available, output="output")
        with pytest.raises(MountsError):
            right.update("alpha", ["input"], available=available, output=None, required=("output",))
        with pytest.raises(MountsError):
            right.update("alpha", ["input"], available=available, output=None, groups=(available,))
        assert first.call("changes") == before
        held = first.call("references.acquire")
        try:
            right.register("output", project="gamma", shared=False)
        finally:
            first.call("references.release", lease=held["lease"])
        assert left.output("alpha", "input") == "input"
        assert left.output_projects("output") == ()
        assert left.ordered("gamma", available) == ("output", "input")
        assert left.ordered("beta", available) == ("input",)
        raw = cast("dict[str, object]", first.call("folders.projects.get")["folders"])
        before = first.call("changes")
        for arguments, code in (
            ({"revision": False, "patch": {"future": False}}, "stale"),
            (
                {
                    "revision": raw["revision"],
                    "patch": {"future": {"lost": True}},
                    "remove": [["future", "keep", "bad"]],
                },
                "invalid",
            ),
        ):
            with pytest.raises(LibraryError) as error:
                second.call("folders.projects.patch", **arguments)
            assert error.value.code == code
            assert second.call("folders.projects.get")["folders"] == raw
            assert first.call("changes") == before
        with ThreadPoolExecutor(max_workers=2) as pool:
            writes = [
                pool.submit(
                    client.call,
                    "folders.projects.patch",
                    revision=raw["revision"],
                    patch={"race": value},
                )
                for client, value in ((first, "first"), (second, "second"))
            ]
            successes = 0
            for write in writes:
                try:
                    write.result()
                    successes += 1
                except LibraryError as error:
                    assert error.code == "stale"
            assert successes == 1
        expected = second.call("folders.projects.get")
        events = cast("list[dict[str, object]]", first.call("changes", cursor=start)["items"])
        folder_events = [event for event in events if event["kind"] == "folders"]
        assert len(folder_events) == 7
        assert [event["revision"] for event in folder_events] == list(range(1, 8))
        assert all(
            json.loads(cast(str, event["id"])) == ["folders", "projects"] for event in folder_events
        )
        assert first.call("settings.get", pack="projects")["settings"] == {
            "pack": "projects",
            "revision": 0,
            "values": {},
        }
    with LibraryClient(tmp_path) as reopened:
        assert reopened.call("folders.projects.get") == expected
        folders = ProjectFolders(reopened)
        assert folders.hidden_models("alpha") == (identity, *other.identities)
        assert folders.output("beta", "output") is None
        folders.initialize(available)
        assert folders.ordered("alpha", available) == ("input",)
        folders.set_model_hidden(
            "alpha", HiddenModelPreference("old", "model/lora", (identity,)), False
        )
        assert folders.hidden_models("alpha") == other.identities
        raw = cast("dict[str, object]", reopened.call("folders.projects.get")["folders"])
        values = cast("dict[str, object]", raw["values"])
        assert values["hiddenModels"] == {"alpha": [other.to_wire()]}
    with LibraryClient(tmp_path / "other-library") as separate:
        folders = ProjectFolders(separate)
        assert folders.ordered("alpha", available) == ()
        assert folders.output("alpha", "output") == "output"
        assert folders.hidden_models("alpha") == ()
    assert not (tmp_path / "project-folders.json").exists()
    assert not (tmp_path / "library.sqlite").exists()


def test_concurrent_folder_initialization_converges_but_user_edits_remain_strict(
    tmp_path: Path,
) -> None:
    from dinkster_assets import MountsError

    from dinkster.project_folders import ProjectFolders

    class ConcurrentClient(LibraryClient):
        gate: Barrier | None = None

        def call(self, operation: str, **arguments: object) -> dict[str, object]:
            if operation == "folders.projects.patch" and self.gate is not None:
                gate, self.gate = self.gate, None
                gate.wait(timeout=10)
            return super().call(operation, **arguments)

    available = ("input", "output", "extra-left", "extra-right")
    with ConcurrentClient(tmp_path) as first, ConcurrentClient(tmp_path) as second:
        left, right = ProjectFolders(first), ProjectFolders(second)
        first.gate = second.gate = Barrier(2)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(store.initialize, ("input", "output")) for store in (left, right)
            ]
            for future in futures:
                future.result()
        left.update("alpha", ["input"], available=available, output=None)
        first.gate = second.gate = Barrier(2)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(store.initialize, available, shared_defaults=(extra,))
                for store, extra in ((left, "extra-left"), (right, "extra-right"))
            ]
            for future in futures:
                future.result()
        assert set(left.ordered("alpha", available)) == {"input", "extra-left", "extra-right"}
        assert left.output("alpha", "output") is None
        assert set(right.ordered("other", available)) == set(available)
        first.gate = second.gate = Barrier(2)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(store.update, project, ["input"], available=available, output=None)
                for store, project in ((left, "left"), (right, "right"))
            ]
            accepted = refused = 0
            for future in futures:
                try:
                    future.result()
                    accepted += 1
                except MountsError as error:
                    assert isinstance(error.__cause__, LibraryError)
                    assert error.__cause__.code == "stale"
                    refused += 1
            assert (accepted, refused) == (1, 1)
        events = cast("list[dict[str, object]]", first.call("changes")["items"])
        assert [item["revision"] for item in events if item["kind"] == "folders"] == [1, 2, 3, 4, 5]


@pytest.mark.parametrize(
    "kind,legacy",
    [
        ("projects", "library.sqlite"),
        ("projects", "project-folders.json"),
        ("grants", "library.sqlite"),
        ("grants", "mounts.toml"),
    ],
)
def test_folder_helper_refuses_legacy_without_touching_it(
    tmp_path: Path, kind: str, legacy: str
) -> None:
    from dinkster_assets import MountConfiguration, MountsError

    from dinkster.project_folders import ProjectFolders

    original = b"legacy folders must not be overwritten"
    with LibraryClient(tmp_path) as helper:
        path = tmp_path / legacy
        path.write_bytes(original)
        for operation, arguments in (
            (f"folders.{kind}.get", {}),
            (f"folders.{kind}.patch", {"revision": 0, "patch": {}}),
        ):
            with pytest.raises(LibraryError) as error:
                helper.call(operation, **arguments)
            assert error.value.code == "migration"
        with pytest.raises(MountsError):
            if kind == "projects":
                ProjectFolders(helper)
            else:
                MountConfiguration.load(helper)
        assert path.read_bytes() == original


def test_filesystem_grant_configuration_preserves_extensions_and_refuses_stale_edits(
    tmp_path: Path,
) -> None:
    from dinkster_assets import MountConfiguration, MountDef, MountsError, parse_mounts

    first_mount = MountDef("input", tmp_path / "missing", priority=-3, kind="model/lora")
    output = MountDef("output", tmp_path / "output", "readwrite", priority=9)
    with LibraryClient(tmp_path) as first, LibraryClient(tmp_path) as second:
        assert "folders.grants" in cast("list[str]", first.hello["capabilities"])
        empty = MountConfiguration.load(first)
        assert (empty.mounts, empty.output_mount, empty.revision) == ((), None, 0)
        saved = empty.save((first_mount, output), output_mount="output")
        assert (saved.mounts, saved.output_mount, saved.revision) == (
            (first_mount, output),
            "output",
            1,
        )
        assert MountConfiguration.load(second) == saved
        second.call(
            "folders.grants.patch",
            revision=1,
            patch={
                "future": {"keep": True},
                "mounts": {"input": {"future": [1, False]}, "output": {"future": "gone"}},
                "settings": {"future": {"nullable": None}},
            },
        )
        baseline = first.call("changes")
        changed = MountDef("input", first_mount.path, "readwrite", priority=0)
        with pytest.raises(MountsError, match="changed"):
            saved.save((changed,))
        assert first.call("changes") == baseline
        current = MountConfiguration.load(first)
        for mounts, selected in (
            ((first_mount,), "input"),
            ((first_mount,), "missing"),
            ((first_mount, first_mount), None),
            ((first_mount, MountDef("alias", first_mount.path)), None),
        ):
            with pytest.raises(MountsError):
                current.save(mounts, output_mount=selected)
            assert first.call("changes") == baseline
        held = first.call("references.acquire")
        try:
            current = MountConfiguration.load(second).save((changed,))
        finally:
            first.call("references.release", lease=held["lease"])
        assert (current.mounts, current.output_mount, current.revision) == ((changed,), None, 3)
        raw = cast("dict[str, object]", first.call("folders.grants.get")["folders"])
        assert raw["values"] == {
            "future": {"keep": True},
            "mounts": {
                "input": {
                    "path": str(first_mount.path),
                    "mode": "readwrite",
                    "priority": 0,
                    "kind": "",
                    "future": [1, False],
                }
            },
            "settings": {"output-mount": None, "future": {"nullable": None}},
        }
        assert first.call("folders.projects.get") == {"folders": {"revision": 0, "values": {}}}
        assert first.call("settings.get", environment="grants")["settings"] == {
            "environment": "grants",
            "revision": 0,
            "values": {},
        }
        with pytest.raises(MountsError, match="unknown keys"):
            parse_mounts('[mounts.input]\npath="folder"\nfuture=true\n')
        latest = MountConfiguration.load(first)
        with ThreadPoolExecutor(max_workers=2) as pool:
            writes = [
                pool.submit(config.save, (MountDef("winner", tmp_path / name),))
                for config, name in ((latest, "first"), (MountConfiguration.load(second), "second"))
            ]
            winners: list[MountConfiguration] = []
            for write in writes:
                try:
                    winners.append(write.result())
                except MountsError as error:
                    assert "changed" in str(error)
            assert len(winners) == 1
        expected = MountConfiguration.load(first)
        assert expected == winners[0]
        events = cast("list[dict[str, object]]", first.call("changes")["items"])
        grants = [
            event
            for event in events
            if event["kind"] == "folders"
            and json.loads(cast(str, event["id"])) == ["folders", "grants"]
        ]
        assert [event["revision"] for event in grants] == [1, 2, 3, 4]
    with LibraryClient(tmp_path) as reopened:
        assert MountConfiguration.load(reopened) == expected
        raw = cast("dict[str, object]", reopened.call("folders.grants.get")["folders"])
        values = cast("dict[str, object]", raw["values"])
        assert "input" not in cast("dict[str, object]", values["mounts"])
        assert values["future"] == {"keep": True}
        assert values["settings"] == {"output-mount": None, "future": {"nullable": None}}
    assert not first_mount.path.exists()
    assert not output.path.exists()
    assert not (tmp_path / "mounts.toml").exists()


def test_folder_transactions_publish_both_records_or_neither(tmp_path: Path) -> None:
    with LibraryClient(tmp_path) as first, LibraryClient(tmp_path) as second:
        assert "folders.transactions" in cast("list[str]", first.hello["capabilities"])
        assert first.call("folders.get") == {
            "grants": {"revision": 0, "values": {}},
            "projects": {"revision": 0, "values": {}},
        }
        first.call(
            "folders.patch",
            grants={"revision": 0, "patch": {"future": True, "mounts": {"input": {"path": "a"}}}},
            projects={"revision": 0, "patch": {"future": {"keep": 1}, "shared": ["input"]}},
        )
        second.call("folders.projects.patch", revision=1, patch={"outputs": {"film": None}})
        baseline, events = first.call("folders.get"), first.call("changes")
        for changes in (
            {"grants": {"revision": 1, "patch": {"future": False}}, "projects": {"revision": 1}},
            {"grants": {"revision": 0}, "projects": {"revision": 2, "patch": {"future": False}}},
            {"grants": {"revision": 1}, "projects": {"revision": 2, "remove": [["shared", "x"]]}},
            {"grants": {"revision": 1}, "projects": {"revision": 2, "patch": []}},
            {"grants": {"revision": 1}, "projects": {"revision": 2, "pack": "not-allowed"}},
            {"grants": {"revision": 1}, "projects": None},
            {"grants": {"revision": 1}},
            {"grants": {"revision": 1}, "projects": {"revision": 2}, "extra": {}},
        ):
            with pytest.raises(LibraryError) as refused:
                first.call("folders.patch", **changes)
            assert refused.value.code in ("stale", "invalid")
            assert second.call("folders.get") == baseline
            assert second.call("changes") == events
        with pytest.raises(LibraryError):
            first.call("folders.get", projects={})
        held = first.call("references.acquire")
        try:
            updated = second.call(
                "folders.patch",
                grants={
                    "revision": 1,
                    "patch": {"mounts": {"output": {"path": "b", "mode": "readwrite"}}},
                    "remove": [["mounts", "input"]],
                },
                projects={"revision": 2, "patch": {"shared": ["output"]}},
            )
        finally:
            first.call("references.release", lease=held["lease"])
        assert updated == {
            "grants": {
                "revision": 2,
                "values": {
                    "future": True,
                    "mounts": {"output": {"path": "b", "mode": "readwrite"}},
                },
            },
            "projects": {
                "revision": 3,
                "values": {"future": {"keep": 1}, "shared": ["output"], "outputs": {"film": None}},
            },
        }
        for name in ("grants", "projects"):
            assert first.call(f"folders.{name}.get")["folders"] == updated[name]
        assert first.call("settings.get", environment="folders")["settings"] == {
            "environment": "folders",
            "revision": 0,
            "values": {},
        }
        baseline_events = cast("list[dict[str, object]]", first.call("changes")["items"])
        assert [
            (json.loads(cast(str, item["id"])), item["revision"])
            for item in baseline_events
            if item["kind"] == "folders"
        ] == [
            (["folders", "grants"], 1),
            (["folders", "projects"], 1),
            (["folders", "projects"], 2),
            (["folders", "grants"], 2),
            (["folders", "projects"], 3),
        ]
        for legacy in ("library.sqlite", "mounts.toml", "project-folders.json"):
            path = tmp_path / legacy
            original = b"legacy folder records"
            path.write_bytes(original)
            try:
                for operation, arguments in (
                    ("folders.get", {}),
                    ("folders.patch", {"grants": {"revision": 2}, "projects": {"revision": 3}}),
                ):
                    with pytest.raises(LibraryError) as refused:
                        first.call(operation, **arguments)
                    assert refused.value.code == "migration"
                assert path.read_bytes() == original
            finally:
                path.unlink()
            assert second.call("folders.get") == updated
            assert second.call("changes")["items"] == baseline_events

        def publish() -> None:
            for sequence in range(1, 31):
                first.call(
                    "folders.patch",
                    grants={"revision": sequence + 1, "patch": {"pair": sequence}},
                    projects={"revision": sequence + 2, "patch": {"pair": sequence}},
                )

        def observe() -> None:
            for _ in range(60):
                state = second.call("folders.get")
                grants = cast("dict[str, object]", state["grants"])
                projects = cast("dict[str, object]", state["projects"])
                assert cast("dict[str, object]", grants["values"]).get("pair") == cast(
                    "dict[str, object]", projects["values"]
                ).get("pair")
                assert cast(int, projects["revision"]) == cast(int, grants["revision"]) + 1

        with ThreadPoolExecutor(max_workers=2) as pool:
            writer, reader = pool.submit(publish), pool.submit(observe)
            writer.result()
            reader.result()
        expected = second.call("folders.get")
        grants = cast("dict[str, object]", expected["grants"])
        projects = cast("dict[str, object]", expected["projects"])
        assert (grants["revision"], projects["revision"]) == (32, 33)
        assert cast("dict[str, object]", grants["values"])["pair"] == 30
        assert cast("dict[str, object]", projects["values"])["pair"] == 30
    with LibraryClient(tmp_path) as reopened:
        assert reopened.call("folders.get") == expected


def test_settings_preserve_unknown_fields_and_reject_concurrent_stale_writes(
    tmp_path: Path,
) -> None:
    key = {"environment": "stable-environment"}
    initial = {
        "logging": {"level": "info", "overrides": {"engine": "warning"}, "future": [1, False]},
        "features": {"future": {"enabled": True}},
        "future": {"nullable": None, "revision": "user value"},
    }
    with LibraryClient(tmp_path) as first, LibraryClient(tmp_path) as second:
        assert "settings" in cast("list[str]", first.hello["capabilities"])
        baseline = first.call("changes")
        assert first.call("settings.get", **key) == {
            "settings": {**key, "revision": 0, "values": {}}
        }
        created = first.call("settings.patch", **key, revision=0, patch=initial)
        assert created == {"settings": {**key, "revision": 1, "values": initial}}
        assert second.call("settings.get", **key) == created
        assert second.call("settings.get", environment="other") == {
            "settings": {"environment": "other", "revision": 0, "values": {}}
        }
        assert first.call("settings.get", pack="stable-environment") == {
            "settings": {"pack": "stable-environment", "revision": 0, "values": {}}
        }
        for arguments in (
            {"environment": "stable-environment", "pack": "pack"},
            {},
            {"environment": ""},
            {"pack": None},
        ):
            with pytest.raises(LibraryError) as invalid:
                first.call("settings.get", **arguments)
            assert invalid.value.code == "invalid"
        for arguments in (
            {"revision": True, "patch": {"logging": {"level": "debug"}}},
            {"revision": 0, "patch": {"logging": {"level": "debug"}}},
        ):
            with pytest.raises(LibraryError) as stale:
                second.call("settings.patch", **key, **arguments)
            assert stale.value.code == "stale"
        for arguments in (
            {"patch": {"logging": {"level": "debug"}}, "remove": [["logging", "level", "bad"]]},
            {"remove": [[]]},
            {"remove": [[False]]},
            {"remove": "logging"},
            {"patch": []},
            {"values": {}},
        ):
            with pytest.raises(LibraryError) as invalid:
                first.call("settings.patch", **key, revision=1, **arguments)
            assert invalid.value.code == "invalid"
            assert second.call("settings.get", **key) == created
        held = first.call("references.acquire")
        try:
            modified = second.call(
                "settings.patch",
                **key,
                revision=1,
                patch={"logging": {"level": "debug"}, "future": {"new": 0}},
                remove=[["logging", "overrides", "engine"]],
            )
        finally:
            first.call("references.release", lease=held["lease"])
        expected = {
            "logging": {"level": "debug", "overrides": {}, "future": [1, False]},
            "features": {"future": {"enabled": True}},
            "future": {"nullable": None, "revision": "user value", "new": 0},
        }
        assert modified == {"settings": {**key, "revision": 2, "values": expected}}
        pack = second.call(
            "settings.patch",
            pack="pack-one",
            revision=0,
            patch={"flag": False, "future": {"keep": "yes"}},
        )
        assert pack == {
            "settings": {
                "pack": "pack-one",
                "revision": 1,
                "values": {"flag": False, "future": {"keep": "yes"}},
            }
        }
        pack = first.call("settings.patch", pack="pack-one", revision=1, patch={"flag": True})
        assert cast("dict[str, object]", pack["settings"])["values"] == {
            "flag": True,
            "future": {"keep": "yes"},
        }
        with ThreadPoolExecutor(max_workers=2) as pool:
            writes = {
                pool.submit(
                    client.call, "settings.patch", **key, revision=2, patch={"race": value}
                ): value
                for client, value in ((first, "first"), (second, "second"))
            }
            winners: list[str] = []
            for future, value in writes.items():
                try:
                    result = future.result()
                    assert result == {
                        "settings": {**key, "revision": 3, "values": {**expected, "race": value}}
                    }
                    winners.append(value)
                except LibraryError as stale:
                    assert stale.code == "stale"
        assert len(winners) == 1
        expected["race"] = winners[0]
        changes = first.call("changes", cursor=baseline["cursor"])
        events = [
            item
            for item in cast("list[dict[str, object]]", changes["items"])
            if item["kind"] == "settings"
        ]
        assert [(json.loads(cast(str, item["id"])), item["revision"]) for item in events] == [
            (["environment", "stable-environment"], 1),
            (["environment", "stable-environment"], 2),
            (["pack", "pack-one"], 1),
            (["pack", "pack-one"], 2),
            (["environment", "stable-environment"], 3),
        ]
    with LibraryClient(tmp_path) as reopened:
        assert reopened.call("settings.get", **key) == {
            "settings": {**key, "revision": 3, "values": expected}
        }
        assert reopened.call("settings.get", pack="pack-one") == pack
        removed = reopened.call("settings.patch", **key, revision=3, remove=[["logging"], ["race"]])
        expected.pop("logging")
        expected.pop("race")
        assert removed == {"settings": {**key, "revision": 4, "values": expected}}
    assert not (tmp_path / "settings.json").exists()
    assert not (tmp_path / "pack-settings").exists()
    assert not (tmp_path / "library.sqlite").exists()


def test_resolver_snapshots_publish_leads_atomically_with_revisioned_chunk_reads(
    tmp_path: Path,
) -> None:
    op = "sources.resolvers."
    identifier = "a" * 32
    digest, other = "blake3:" + "15" * 32, "blake3:" + "26" * 32

    def snapshot(target: str, source: str = "https://example.com/index") -> dict[str, object]:
        return {
            "subscription": {
                "id": identifier,
                "source": source,
                "sourceType": "url",
                "completeSnapshot": True,
                "checkedAt": 2.0,
                "refreshedAt": 1.0,
                "index": {
                    "dinksterResolver": 1,
                    "entries": [
                        {"digest": target, "name": "model", "urls": ["https://example.com/model"]}
                    ],
                },
                "futureSubscription": {"keep": True},
            },
            "records": [
                {"digest": target, "sources": ["https://example.com/model"], "futureLead": True}
            ],
            "futureEnvelope": {"keep": True},
        }

    def upload(client: LibraryClient, expected: int, envelope: dict[str, object]) -> str:
        token = cast(str, client.call(op + "begin", id=identifier, revision=expected)["token"])
        payload = json.dumps(envelope).encode()
        for offset in range(0, len(payload), 1024 * 1024):
            chunk = payload[offset : offset + 1024 * 1024]
            assert client.call(
                op + "append", token=token, offset=offset, data=base64.b64encode(chunk).decode()
            )["offset"] == offset + len(chunk)
        return token

    with LibraryClient(tmp_path) as first, LibraryClient(tmp_path) as second:
        assert "sources.resolvers" in cast("list[str]", first.hello["capabilities"])
        initial = snapshot(digest)
        for invalid_json in (
            json.dumps(initial).replace(
                '{"subscription":', '{"subscription": {}, "subscription":', 1
            ),
            json.dumps(initial).replace('"name": "model"', '"name": "wrong", "name": "model"'),
        ):
            invalid_token = first.call(op + "begin", id=identifier, revision=0)["token"]
            first.call(
                op + "append",
                token=invalid_token,
                offset=0,
                data=base64.b64encode(invalid_json.encode()).decode(),
            )
            feed = second.call("changes", after=0)
            with pytest.raises(LibraryError) as invalid:
                first.call(op + "commit", token=invalid_token)
            assert invalid.value.code == "invalid"
            assert second.call("changes", after=0) == feed
            assert second.call(op + "get", id=identifier) == {"subscription": None}
            assert second.call("sources.provenance.get", digest=digest) == {"record": None}
            first.call(op + "cancel", token=invalid_token)
        token = upload(first, 0, initial)
        assert second.call(op + "get", id=identifier) == {"subscription": None}
        assert second.call("sources.provenance.get", digest=digest) == {"record": None}
        with pytest.raises(LibraryError) as foreign:
            second.call(op + "commit", token=token)
        assert foreign.value.code == "stale"
        result = first.call(op + "commit", token=token)
        assert result["revision"] == 1
        lead = cast(
            "dict[str, object]", second.call("sources.provenance.get", digest=digest)["record"]
        )
        assert lead["futureLead"] is True
        raw = second.call(op + "read", id=identifier, revision=1)
        assert raw["complete"] is True
        assert json.loads(base64.b64decode(cast(str, raw["data"]))) == initial
        assert len(cast("list[object]", first.call(op + "list")["items"])) == 1
        refreshed = snapshot(other)
        refreshed.pop("futureEnvelope")
        cast("dict[str, object]", refreshed["subscription"]).pop("futureSubscription")
        left = upload(first, 1, refreshed)
        right = upload(second, 1, snapshot(digest))
        assert first.call(op + "commit", token=left)["revision"] == 2
        feed = first.call("changes", after=0)
        with pytest.raises(LibraryError) as stale:
            second.call(op + "commit", token=right)
        assert stale.value.code == "stale"
        second.call(op + "cancel", token=right)
        assert first.call("changes", after=0) == feed
        assert first.call("sources.provenance.get", digest=digest) == {"record": None}
        assert first.call("sources.provenance.get", digest=other)["record"] is not None
        with pytest.raises(LibraryError) as old_read:
            second.call(op + "read", id=identifier, revision=1)
        assert old_read.value.code == "stale"
        refreshed_raw = first.call(op + "read", id=identifier, revision=2)
        preserved = json.loads(base64.b64decode(cast(str, refreshed_raw["data"])))
        assert preserved["futureEnvelope"] == {"keep": True}
        assert preserved["subscription"]["futureSubscription"] == {"keep": True}
        duplicate = snapshot(digest)
        cast("dict[str, object]", duplicate["subscription"])["id"] = "b" * 32
        duplicate_token = second.call(op + "begin", id="b" * 32, revision=0)["token"]
        second.call(
            op + "append",
            token=duplicate_token,
            offset=0,
            data=base64.b64encode(json.dumps(duplicate).encode()).decode(),
        )
        with pytest.raises(LibraryError) as duplicate_source:
            second.call(op + "commit", token=duplicate_token)
        assert duplicate_source.value.code == "conflict"
        second.call(op + "cancel", token=duplicate_token)
        assert first.call("sources.provenance.get", digest=digest) == {"record": None}
        assert first.call("sources.provenance.get", digest=other)["record"] is not None
        assert first.call("changes", after=0) == feed
        conflicting = upload(second, 2, snapshot(digest, "https://example.com/changed"))
        with pytest.raises(LibraryError) as conflict:
            second.call(op + "commit", token=conflicting)
        assert conflict.value.code == "conflict"
        second.call(op + "cancel", token=conflicting)
        assert first.call("changes", after=0) == feed
        for action in ("replace", "remove"):
            with pytest.raises(LibraryError) as managed:
                second.call(
                    "sources.provenance." + action,
                    source="resolver-index:" + identifier,
                    **({"records": []} if action == "replace" else {}),
                )
            assert managed.value.code == "invalid"
        held = first.call("references.acquire")
        try:
            assert held["digests"] == []
            assert second.call(op + "remove", id=identifier, revision=2) == {"revision": 3}
            assert second.call("sources.provenance.get", digest=other) == {"record": None}
        finally:
            first.call("references.release", lease=held["lease"])
        header = cast("dict[str, object]", first.call(op + "get", id=identifier)["subscription"])
        assert header["revision"] == 3 and header["deleted"] is True
        assert first.call(op + "list")["items"] == []
        missing = upload(first, 0, initial)
        with pytest.raises(LibraryError) as tombstone:
            first.call(op + "commit", token=missing)
        assert tombstone.value.code == "stale"
        first.call(op + "cancel", token=missing)
        initial["padding"] = "x" * (17 * 1024 * 1024)
        large = upload(first, 3, initial)
        assert first.call(op + "commit", token=large)["revision"] == 4
        chunks = bytearray()
        while True:
            page = second.call(op + "read", id=identifier, revision=4, offset=len(chunks))
            chunks.extend(base64.b64decode(cast(str, page["data"])))
            assert page["offset"] == len(chunks)
            if page["complete"]:
                break
        assert json.loads(chunks) == initial
        lost = upload(second, 4, snapshot(other))
    with LibraryClient(tmp_path) as reopened:
        with pytest.raises(LibraryError) as exited:
            reopened.call(op + "commit", token=lost)
        assert exited.value.code == "stale"
        header = cast("dict[str, object]", reopened.call(op + "get", id=identifier)["subscription"])
        assert header["revision"] == 4 and header["bytes"] == len(chunks)
        assert reopened.call("sources.provenance.get", digest=other) == {"record": None}
        assert reopened.call("sources.provenance.get", digest=digest)["record"] is not None


def test_resolver_envelope_capacity_is_not_the_bounded_index_capacity(tmp_path: Path) -> None:
    op = "sources.resolvers."
    identifier = "c" * 32
    source = "https://example.com/" + "s" * 4000
    digests = ["blake3:" + f"{index:064x}" for index in range(17_000)]
    index = {
        "dinksterResolver": 1,
        "entries": [
            {"digest": digest, "name": "model", "urls": ["https://example.com/model"]}
            for digest in digests
        ],
    }
    assert len(json.dumps(index).encode()) < 16 * 1024 * 1024
    envelope = {
        "subscription": {"id": identifier, "source": source, "sourceType": "url", "index": index},
        "records": [
            {
                "digest": digest,
                "sources": ["https://example.com/model"],
                "metadata": {"name": "model", "resolverIndex": source},
            }
            for digest in digests
        ],
    }
    payload = json.dumps(envelope).encode()
    assert len(payload) > 64 * 1024 * 1024
    with LibraryClient(tmp_path) as helper:
        token = helper.call(op + "begin", id=identifier, revision=0)["token"]
        for offset in range(0, len(payload), 1024 * 1024):
            helper.call(
                op + "append",
                token=token,
                offset=offset,
                data=base64.b64encode(payload[offset : offset + 1024 * 1024]).decode(),
            )
        result = helper.call(op + "commit", token=token)
        assert result["revision"] == 1
        assert result["bytes"] == len(payload)
        tail = helper.call(op + "read", id=identifier, revision=1, offset=64 * 1024 * 1024)
        assert (
            base64.b64decode(cast(str, tail["data"]))
            == payload[64 * 1024 * 1024 : 65 * 1024 * 1024]
        )
        state = helper.call(op + "state")
        items = cast("list[dict[str, object]]", state["items"])
        assert items == [{"id": identifier, "revision": 1, "snapshot": envelope}]
        assert helper.call(op + "patch", updates=items) == {"revisions": {identifier: 2}}
    with LibraryClient(tmp_path) as reopened:
        for digest in (digests[0], digests[-1]):
            record = cast(
                "dict[str, object]",
                reopened.call("sources.provenance.get", digest=digest)["record"],
            )
            assert record["metadata"] == {"name": "model", "resolverIndex": source}
        with pytest.raises(LibraryError) as invalid_offset:
            reopened.call(op + "read", id=identifier, revision=1, offset=2**63)
        assert invalid_offset.value.code == "invalid"


def test_resolver_subscriptions_do_not_share_one_index_entry_budget(tmp_path: Path) -> None:
    op = "sources.resolvers."
    count = 50_001
    with LibraryClient(tmp_path) as writer, LibraryClient(tmp_path) as reader:
        for number, identifier in enumerate(("a" * 32, "b" * 32)):
            digests = ["blake3:" + f"{number * count + index:064x}" for index in range(count)]
            source = f"https://example.com/index-{number}"
            envelope = {
                "subscription": {
                    "id": identifier,
                    "source": source,
                    "sourceType": "url",
                    "index": {
                        "dinksterResolver": 1,
                        "entries": [
                            {
                                "digest": digest,
                                "name": "model",
                                "urls": ["https://example.com/model"],
                            }
                            for digest in digests
                        ],
                    },
                },
                "records": [{"digest": digest, "sources": [source]} for digest in digests],
            }
            subscription = cast("dict[str, object]", envelope["subscription"])
            assert len(json.dumps(subscription["index"]).encode()) < 16 * 1024 * 1024
            payload = json.dumps(envelope).encode()
            token = writer.call(op + "begin", id=identifier, revision=0)["token"]
            for offset in range(0, len(payload), 1024 * 1024):
                writer.call(
                    op + "append",
                    token=token,
                    offset=offset,
                    data=base64.b64encode(payload[offset : offset + 1024 * 1024]).decode(),
                )
            assert writer.call(op + "commit", token=token)["revision"] == 1
            for digest in (digests[0], digests[-1]):
                record = cast(
                    "dict[str, object]",
                    reader.call("sources.provenance.get", digest=digest)["record"],
                )
                assert record["sources"] == [source]
            page = reader.call(
                "sources.provenance.list", source="resolver-index:" + identifier, limit=1
            )
            assert cast("list[dict[str, object]]", page["items"])[0]["digest"] == digests[0]
            assert page["cursor"] is not None
        assert len(cast("list[object]", reader.call(op + "list")["items"])) == 2
        writer.call(op + "remove", id="a" * 32, revision=1)
        assert reader.call("sources.provenance.get", digest="blake3:" + "0" * 64) == {
            "record": None
        }
        surviving = "blake3:" + f"{count:064x}"
        assert reader.call("sources.provenance.get", digest=surviving)["record"] is not None
    with LibraryClient(tmp_path) as reopened:
        assert reopened.call("sources.provenance.get", digest=surviving)["record"] is not None


def test_resolver_bootstrap_persists_user_unsubscribe_and_guards_provider_identity(
    tmp_path: Path,
) -> None:
    op = "sources.resolvers."
    identifier = "a" * 32
    source = "https://example.com/official"
    digest = "blake3:" + "37" * 32
    configuration = {
        "source": source,
        "providerId": "official",
        "subscriptionId": identifier,
        "future": {"keep": True, "count": 1},
    }

    def upload(
        client: LibraryClient, revision: int, provider: str, target: str = identifier
    ) -> str:
        envelope = {
            "subscription": {
                "id": target,
                "source": source,
                "sourceType": "url",
                "index": {"dinksterResolver": 1, "name": provider, "entries": [{"digest": digest}]},
            },
            "records": [{"digest": digest, "sources": [source]}],
        }
        token = cast(str, client.call(op + "begin", id=target, revision=revision)["token"])
        client.call(
            op + "append",
            token=token,
            offset=0,
            data=base64.b64encode(json.dumps(envelope).encode()).decode(),
        )
        return token

    with LibraryClient(tmp_path) as first, LibraryClient(tmp_path) as second:
        assert first.call(op + "bootstrap.get") == {"configuration": None}
        mismatched = upload(first, 0, "impostor")
        before = second.call("changes", after=0)
        with pytest.raises(LibraryError) as identity:
            first.call(op + "commit", token=mismatched, officialBootstrap=configuration)
        assert identity.value.code == "conflict"
        assert second.call(op + "bootstrap.get") == {"configuration": None}
        assert second.call(op + "get", id=identifier) == {"subscription": None}
        assert second.call("sources.provenance.get", digest=digest) == {"record": None}
        assert second.call("changes", after=0) == before
        first.call(op + "cancel", token=mismatched)
        occupied = upload(second, 0, "official", "b" * 32)
        second.call(op + "commit", token=occupied)
        before = first.call("changes", after=0)
        with pytest.raises(LibraryError) as wrong_existing:
            first.call(
                op + "bootstrap.put",
                configuration={
                    **configuration,
                    "subscriptionId": "b" * 32,
                    "providerId": "impostor",
                },
            )
        assert wrong_existing.value.code == "conflict"
        assert first.call(op + "bootstrap.get") == {"configuration": None}
        assert first.call("changes", after=0) == before
        token = upload(first, 0, "official")
        before = second.call("changes", after=0)
        with pytest.raises(LibraryError) as duplicate_source:
            first.call(op + "commit", token=token, officialBootstrap=configuration)
        assert duplicate_source.value.code == "conflict"
        assert second.call(op + "bootstrap.get") == {"configuration": None}
        assert second.call(op + "get", id=identifier) == {"subscription": None}
        assert second.call(
            "sources.provenance.get", digest=digest, source="resolver-index:" + identifier
        ) == {"record": None}
        assert second.call("changes", after=0) == before
        second.call(op + "remove", id="b" * 32, revision=1)
        assert (
            first.call(op + "commit", token=token, officialBootstrap=configuration)["revision"] == 1
        )
        assert second.call(op + "bootstrap.get") == {"configuration": configuration}
        before = second.call("changes", after=0)
        older = {key: value for key, value in configuration.items() if key != "future"}
        assert second.call(op + "bootstrap.put", configuration=older) == {
            "configuration": configuration,
            "created": False,
        }
        assert second.call("changes", after=0) == before
        for extension in ({"keep": 1, "count": 1}, {"keep": True, "count": 1.0}):
            with pytest.raises(LibraryError) as changed_type:
                second.call(op + "bootstrap.put", configuration={**older, "future": extension})
            assert changed_type.value.code == "conflict"
            assert second.call(op + "bootstrap.get") == {"configuration": configuration}
            assert second.call("changes", after=0) == before
        assert second.call(
            op + "bootstrap.put", configuration={**older, "future": {"count": 1, "keep": True}}
        ) == {"configuration": configuration, "created": False}
        with pytest.raises(LibraryError) as changed:
            second.call(op + "bootstrap.put", configuration={**older, "providerId": "different"})
        assert changed.value.code == "conflict"
        refresh = upload(second, 1, "impostor")
        with pytest.raises(LibraryError) as provider:
            second.call(op + "commit", token=refresh)
        assert provider.value.code == "conflict"
        assert first.call("changes", after=0) == before
        second.call(op + "cancel", token=refresh)
        lease = first.call("references.acquire")["lease"]
        try:
            second.call(op + "remove", id=identifier, revision=1)
            assert second.call(op + "bootstrap.get") == {"configuration": configuration}
            assert second.call(op + "list")["items"] == []
        finally:
            first.call("references.release", lease=lease)
    with LibraryClient(tmp_path) as reopened:
        assert reopened.call(op + "bootstrap.get") == {"configuration": configuration}
        assert reopened.call(op + "list")["items"] == []
        assert reopened.call("sources.provenance.get", digest=digest) == {"record": None}
    with (
        LibraryClient(tmp_path / "unsubscribed-import") as imported,
        LibraryClient(tmp_path / "unsubscribed-import") as holder,
    ):
        lease = holder.call("references.acquire")["lease"]
        try:
            # A legacy bootstrap may outlive its subscription, and must not recreate it.
            assert imported.call(op + "bootstrap.put", configuration=configuration) == {
                "configuration": configuration,
                "created": True,
            }
            assert imported.call(op + "list")["items"] == []
        finally:
            holder.call("references.release", lease=lease)


@pytest.mark.parametrize("filename", ["library.sqlite", "provenance.json", "resolver-indexes.json"])
def test_resolvers_refuse_unmigrated_state(tmp_path: Path, filename: str) -> None:
    (tmp_path / filename).write_bytes(b"legacy resolver state")
    with LibraryClient(tmp_path) as helper:
        for action in (
            "begin",
            "append",
            "commit",
            "cancel",
            "get",
            "list",
            "read",
            "remove",
            "bootstrap.get",
            "bootstrap.put",
            "state",
            "patch",
        ):
            with pytest.raises(LibraryError) as failure:
                helper.call("sources.resolvers." + action)
            assert failure.value.code == "migration"
    assert (tmp_path / filename).read_bytes() == b"legacy resolver state"


def test_provenance_merges_live_layers_without_losing_other_helpers_or_unknown_fields(
    tmp_path: Path,
) -> None:
    digest = "blake3:" + "31" * 32
    other = "blake3:" + "42" * 32
    op = "sources.provenance."
    with LibraryClient(tmp_path) as first, LibraryClient(tmp_path) as second:
        assert "sources.provenance" in cast("list[str]", first.hello["capabilities"])
        with ThreadPoolExecutor(max_workers=2) as pool:
            writes = [
                pool.submit(
                    client.call,
                    op + "add",
                    record={
                        "digest": digest,
                        "sources": [url],
                        "metadata": {url: True},
                        "future-" + url: {"keep": True},
                    },
                )
                for client, url in ((first, "a"), (second, "b"))
            ]
            for future in writes:
                future.result()
        base = cast("dict[str, object]", first.call(op + "get", digest=digest)["record"])
        assert set(cast("list[str]", base["sources"])) == {"a", "b"}
        assert base["metadata"] == {"a": True, "b": True}
        assert base["future-a"] == base["future-b"] == {"keep": True}
        # Insertion order, not the lexical source name, determines scalar precedence.
        first.call(
            op + "replace",
            source="z",
            records=[
                {
                    "digest": digest,
                    "sources": ["b", "z"],
                    "license": "first",
                    "note": "z",
                    "metadata": {"shared": "z"},
                    "future-z": 1,
                },
                {"digest": other, "sources": ["other"]},
            ],
        )
        second.call(
            op + "replace",
            source="a",
            records=[
                {
                    "digest": digest,
                    "sources": ["a-layer"],
                    "license": "last",
                    "note": "",
                    "metadata": {"shared": "a"},
                    "future-a-layer": True,
                },
            ],
        )
        merged = cast("dict[str, object]", first.call(op + "get", digest=digest)["record"])
        assert cast("list[str]", merged["sources"])[-2:] == ["z", "a-layer"]
        assert merged["license"] == "last" and merged["note"] == "z"
        assert merged["metadata"] == {"a": True, "b": True, "shared": "a"}
        assert merged["future-z"] == 1 and merged["future-a-layer"] is True
        assert first.call(op + "get", digest=digest, source="")["record"] == base
        first.call(
            op + "replace",
            source="z",
            records=[
                {"digest": digest, "sources": ["replacement"], "license": "replaced"},
                {"digest": digest, "sources": ["duplicate"], "note": "combined"},
            ],
        )
        assert second.call(op + "get", digest=other) == {"record": None}
        current = cast("dict[str, object]", first.call(op + "get", digest=digest)["record"])
        assert current["license"] == "last" and current["note"] == "combined"
        assert current["sources"] == cast("list[str]", base["sources"]) + [
            "replacement",
            "duplicate",
            "a-layer",
        ]
        assert current["future-z"] == 1
        first.call(op + "add", record={"digest": other, "sources": ["independent"]})
        page = first.call(op + "list", limit=1)
        assert cast("list[dict[str, object]]", page["items"])[0]["digest"] == digest
        assert second.call(op + "list", cursor=page["cursor"], limit=1)["items"] == [
            {"digest": other, "sources": ["independent"], "license": "", "note": "", "metadata": {}}
        ]
        layers = first.call(op + "layers", limit=1)
        assert layers["items"] == ["a"]
        assert second.call(op + "layers", cursor=layers["cursor"], limit=1)["items"] == ["z"]
        for call, arguments in (
            ("list", {"source": "z", "cursor": page["cursor"]}),
            ("list", {"cursor": layers["cursor"]}),
            ("list", {"limit": 101}),
            ("remove", {"source": ""}),
            ("replace", {"source": "bad", "records": [{"digest": other}, {"digest": "invalid"}]}),
        ):
            with pytest.raises(LibraryError) as invalid:
                first.call(op + call, **arguments)
            assert invalid.value.code == "invalid"
        assert first.call(op + "layers")["items"] == ["a", "z"]
        held = first.call("references.acquire")
        assert held["digests"] == []
        try:
            assert second.call(op + "remove", source="a") == {"removed": True}
            assert second.call(op + "remove", source="missing") == {"removed": False}
            restored = cast("dict[str, object]", second.call(op + "get", digest=digest)["record"])
            assert restored["license"] == "replaced"
            second.call(op + "replace", source="z", records=[])
            assert second.call(op + "get", digest=digest)["record"] == base
        finally:
            first.call("references.release", lease=held["lease"])
        changes = cast("list[dict[str, object]]", first.call("changes", after=0)["items"])
        assert any(change["kind"] == "provenance" for change in changes)
    with LibraryClient(tmp_path) as reopened:
        assert reopened.call(op + "get", digest=digest)["record"] == base
        assert reopened.call(op + "layers")["items"] == []


def test_provenance_consumer_shares_current_helper_records_and_keeps_extensions(
    tmp_path: Path,
) -> None:
    from dinkster_assets import AssetError, ProvenanceRecord, ProvenanceStore

    digest = "blake3:" + "57" * 32
    with LibraryClient(tmp_path) as first, LibraryClient(tmp_path) as second:
        writer, reader = ProvenanceStore(first), ProvenanceStore(second)
        assert reader.get(digest) is None
        first.call("sources.provenance.add", record={"digest": digest, "future": {"keep": True}})
        with ThreadPoolExecutor() as pool:
            pending = [
                pool.submit(store.add, ProvenanceRecord(digest, sources=(url,)))
                for store, url in (
                    (writer, "https://example.com/first"),
                    (reader, "https://example.com/second"),
                )
            ]
            for task in pending:
                task.result()
        assert set(reader.sources(digest)) == {
            "https://example.com/first",
            "https://example.com/second",
        }
        first.call(
            "sources.provenance.replace",
            source="feed",
            records=[{"digest": digest, "license": "old", "future": {"layer": True}}],
        )
        writer.replace_source("feed", (ProvenanceRecord(digest, license="new"),))
        assert reader.get(digest) == ProvenanceRecord(
            digest, sources=reader.sources(digest), license="new"
        )
        raw = cast(
            "dict[str, object]",
            second.call("sources.provenance.get", digest=digest, source="feed")["record"],
        )
        assert raw["future"] == {"layer": True}
        for index in range(101):
            writer.replace_source(
                f"source-{index:03}", (ProvenanceRecord("blake3:" + f"{index:064x}"),)
            )
        assert set(reader.source_names()) == {"feed"} | {f"source-{i:03}" for i in range(101)}
        assert len(reader.records()) == 102
        assert reader.remove_source("feed")
        assert not writer.remove_source("feed")
        assert second.call("sources.provenance.get", digest=digest)["record"] == {
            "digest": digest,
            "sources": list(reader.sources(digest)),
            "license": "",
            "note": "",
            "metadata": {},
            "future": {"keep": True},
        }
        with pytest.raises(AssetError):
            writer.replace_source("resolver-index:managed", (ProvenanceRecord(digest),))
        assert first.call("hello")["apiMin"] == 1
    with LibraryClient(tmp_path) as reopened:
        assert len(ProvenanceStore(reopened).records()) == 102
    assert not (tmp_path / "provenance.json").exists()


@pytest.mark.parametrize("failure", ["disconnected", "migration"])
def test_provenance_leads_remain_advisory_when_metadata_is_unavailable(
    tmp_path: Path, failure: str
) -> None:
    from dinkster_assets import AssetError, ProvenanceRecord, ProvenanceStore

    digest = "blake3:" + "58" * 32
    with LibraryClient(tmp_path) as helper:
        store = ProvenanceStore(helper)
        store.add(ProvenanceRecord(digest, sources=("https://example.com/model",)))
        assert store.sources(digest) == ("https://example.com/model",)
        if failure == "disconnected":
            helper.close()
        else:
            (tmp_path / "provenance.json").write_bytes(b"preserve legacy metadata")
        assert store.sources(digest) == ()
        with pytest.raises(AssetError):
            store.get(digest)
        with pytest.raises(AssetError):
            store.records()
        with pytest.raises(AssetError):
            store.sources("invalid digest")
    if failure == "migration":
        assert (tmp_path / "provenance.json").read_bytes() == b"preserve legacy metadata"


def test_resolver_consumer_shares_atomic_refreshes_sources_and_saved_choices(
    tmp_path: Path,
) -> None:
    from unittest.mock import patch

    from dinkster_assets import (
        ProvenanceStore,
        ResolverSubscriptionError,
        ResolverSubscriptionStore,
    )

    digest, other = "blake3:" + "61" * 32, "blake3:" + "62" * 32

    def document(name: str, value: str) -> dict[str, object]:
        return {
            "dinksterResolver": 1,
            "name": name,
            "entries": [
                {
                    "digest": value,
                    "name": name + ".bin",
                    "size": 7,
                    "urls": ["https://example.com/default"],
                    "regions": {"eu": ["https://example.com/eu"]},
                }
            ],
        }

    first_path, second_path = tmp_path / "first.json", tmp_path / "second.json"
    first_path.write_text(json.dumps(document("first", digest)), "utf-8")
    second_path.write_text(json.dumps(document("second", other)), "utf-8")
    root = tmp_path / "library"
    with LibraryClient(root) as first, LibraryClient(root) as second:
        provenance = ProvenanceStore(second)
        writer = ResolverSubscriptionStore(
            first, ProvenanceStore(first), region="eu", clock=lambda: 20.0
        )
        reader = ResolverSubscriptionStore(second, provenance, region="eu")
        one = writer.subscribe(str(first_path))
        two = writer.subscribe(str(second_path))
        assert [item.id for item in reader.subscriptions()] == [one.id, two.id]
        assert second.call("sources.resolvers.state", revisions={one.id: 1, two.id: 1}) == {
            "items": [{"id": one.id, "revision": 1}, {"id": two.id, "revision": 1}],
            "officialBootstrap": None,
        }
        with pytest.raises(LibraryError) as invalid_revision:
            second.call("sources.resolvers.state", revisions={one.id: True})
        assert invalid_revision.value.code == "invalid"
        assert reader.suggest("dir/first.bin")[0].digest == digest
        assert reader.public_sources(digest)[0].listed_urls == (
            "https://example.com/eu",
            "https://example.com/default",
        )
        assert provenance.sources(digest) == (
            "https://example.com/eu",
            "https://example.com/default",
        )
        state = first.call("sources.resolvers.state")
        items = cast("list[dict[str, object]]", state["items"])
        before_changes = first.call("changes")
        changed = json.loads(json.dumps(items))
        cast("dict[str, object]", changed[0]["snapshot"])["futureEnvelope"] = {"keep": True}
        first_document = cast("dict[str, object]", changed[0]["snapshot"]["subscription"])
        first_document["futureSubscription"] = {"keep": True}
        raw_index = cast("dict[str, object]", first_document["index"])
        raw_index["futureIndex"] = {"keep": True}
        raw_entry = cast("list[dict[str, object]]", raw_index["entries"])[0]
        raw_entry["futureEntry"] = {"keep": True}
        raw_entry["license"] = "known old license"
        changed[1]["revision"] += 1
        with pytest.raises(LibraryError) as stale:
            first.call("sources.resolvers.patch", updates=changed)
        assert stale.value.code == "stale"
        assert second.call("sources.resolvers.state") == state
        assert second.call("changes") == before_changes
        with pytest.raises(LibraryError) as stale_after_remove:
            first.call(
                "sources.resolvers.patch",
                updates=[{**changed[0], "revision": 2}],
                remove=[{"id": two.id, "revision": 1}],
            )
        assert stale_after_remove.value.code == "stale"
        assert second.call("sources.resolvers.state") == state
        assert second.call("changes") == before_changes
        first.call("sources.resolvers.patch", updates=[changed[0]])
        first_path.write_text("{invalid", "utf-8")
        statuses = writer.refresh()
        assert len(statuses) == 2 and "invalid resolver index JSON" in str(statuses[0]["error"])
        stored = cast("list[dict[str, object]]", second.call("sources.resolvers.state")["items"])[0]
        envelope = cast("dict[str, object]", stored["snapshot"])
        subscription = cast("dict[str, object]", envelope["subscription"])
        assert envelope["futureEnvelope"] == {"keep": True}
        assert subscription["futureSubscription"] == {"keep": True}
        assert cast("dict[str, object]", subscription["index"])["futureIndex"] == {"keep": True}
        assert provenance.sources(digest) == (
            "https://example.com/eu",
            "https://example.com/default",
        )
        first_path.write_text(json.dumps(document("retained", digest)), "utf-8")
        writer.refresh(one.id)
        stored = cast("list[dict[str, object]]", second.call("sources.resolvers.state")["items"])[0]
        current_snapshot = cast("dict[str, object]", stored["snapshot"])
        current_subscription = cast("dict[str, object]", current_snapshot["subscription"])
        current_index = cast("dict[str, object]", current_subscription["index"])
        current_entry = cast("list[dict[str, object]]", current_index["entries"])[0]
        assert current_index["futureIndex"] == {"keep": True}
        assert current_entry["futureEntry"] == {"keep": True}
        assert "license" not in current_entry
        first_path.write_text(json.dumps(document("replacement", other)), "utf-8")
        writer.refresh(one.id)
        assert reader.suggest("first.bin") == ()
        assert reader.suggest("replacement.bin")[0].digest == other
        replaced = cast("list[dict[str, object]]", second.call("sources.resolvers.state")["items"])[
            0
        ]
        replaced_snapshot = cast("dict[str, object]", replaced["snapshot"])
        replaced_subscription = cast("dict[str, object]", replaced_snapshot["subscription"])
        replaced_index = cast("dict[str, object]", replaced_subscription["index"])
        assert "futureEntry" not in cast("list[dict[str, object]]", replaced_index["entries"])[0]
        assert provenance.sources(digest) == ()
        assert reader.unsubscribe(one.id)
        assert [item.id for item in writer.subscriptions()] == [two.id]
        assert first.call("sources.resolvers.get", id=one.id)["subscription"] is not None
        with patch(
            "dinkster_assets.resolver_subscription._fetch_url",
            return_value=(json.dumps(document("official", digest)).encode(), "tag"),
        ):
            official = writer.bootstrap_official("https://example.com/index", "official")
        assert official is not None
        assert reader.unsubscribe(official.id)
        assert writer.bootstrap_official("https://example.com/index", "official") is None
        with pytest.raises(ResolverSubscriptionError, match="saved choice preserved"):
            writer.bootstrap_official("https://example.com/changed", "official")
        assert first.call("hello")["apiMin"] == 1
    with LibraryClient(root) as reopened:
        store = ResolverSubscriptionStore(reopened, ProvenanceStore(reopened))
        assert [item.id for item in store.subscriptions()] == [two.id]
        assert store.bootstrap_official("https://example.com/index", "official") is None
    assert not (root / "resolver-indexes.json").exists()
    assert not (root / "provenance.json").exists()


@pytest.mark.parametrize("failure", ["disconnected", "migration", "unreadable"])
def test_resolver_advisory_reads_drop_unavailable_shared_state(
    tmp_path: Path, failure: str
) -> None:
    from dinkster_assets import (
        ProvenanceStore,
        ResolverSubscriptionError,
        ResolverSubscriptionStore,
    )

    digest = "blake3:" + "64" * 32
    source = tmp_path / "index.json"
    source.write_text(
        json.dumps(
            {
                "dinksterResolver": 1,
                "entries": [
                    {
                        "digest": digest,
                        "name": "model.bin",
                        "size": 7,
                        "kind": "community/future-kind",
                        "urls": ["https://example.com/model"],
                    }
                ],
            }
        ),
        "utf-8",
    )
    root = tmp_path / "library"
    with LibraryClient(root) as helper, LibraryClient(root) as other:
        store = ResolverSubscriptionStore(helper, ProvenanceStore(helper))
        subscription = store.subscribe(str(source))
        assert store.suggest("model.bin")[0].digest == digest
        assert store.public_sources(digest)[0].source_id == subscription.id
        legacy = root / "resolver-indexes.json"
        peer_state: tuple[dict[str, object], dict[str, object]] | None = None
        if failure == "unreadable":
            item = cast("list[dict[str, object]]", other.call("sources.resolvers.state")["items"])[
                0
            ]
            snapshot = cast("dict[str, object]", item["snapshot"])
            raw_subscription = cast("dict[str, object]", snapshot["subscription"])
            index = cast("dict[str, object]", raw_subscription["index"])
            entry = cast("list[dict[str, object]]", index["entries"])[0]
            entry["kind"] = "not-a-namespaced-kind"
            other.call("sources.resolvers.patch", updates=[item])
            peer_state = (other.call("sources.resolvers.state"), other.call("changes"))
        else:
            assert ResolverSubscriptionStore(other, ProvenanceStore(other)).unsubscribe(
                subscription.id
            )
            if failure == "disconnected":
                helper.close()
            else:
                legacy.write_bytes(b"preserve legacy subscriptions")
        assert store.public_sources(digest) == ()
        assert store.suggest("model.bin") == ()
        for operation in (
            store.subscriptions,
            lambda: store.unsubscribe(subscription.id),
            lambda: store.subscribe(str(source)),
            store.refresh,
            lambda: store.bootstrap_official("https://example.com/index", "official"),
        ):
            with pytest.raises(ResolverSubscriptionError):
                operation()
        if peer_state is not None:
            assert other.call("sources.resolvers.state") == peer_state[0]
            assert other.call("changes") == peer_state[1]
            other.call("sources.resolvers.remove", id=subscription.id, revision=2)
        elif failure == "migration":
            assert legacy.read_bytes() == b"preserve legacy subscriptions"
            legacy.unlink()
        if failure != "disconnected":
            assert store.subscriptions() == ()
            assert store.suggest("model.bin") == ()
            assert store.public_sources(digest) == ()


def test_resolver_consumer_refuses_a_refresh_after_shared_unsubscribe(tmp_path: Path) -> None:
    from unittest.mock import patch

    from dinkster_assets import (
        ProvenanceStore,
        ResolverSubscriptionError,
        ResolverSubscriptionStore,
    )
    from dinkster_assets import resolver_subscription as transport

    digest = "blake3:" + "63" * 32
    source = tmp_path / "index.json"
    source.write_text(
        json.dumps(
            {
                "dinksterResolver": 1,
                "entries": [
                    {
                        "digest": digest,
                        "name": "model.bin",
                        "urls": ["https://example.com/model"],
                    }
                ],
            }
        ),
        "utf-8",
    )
    with (
        LibraryClient(tmp_path / "library") as first,
        LibraryClient(tmp_path / "library") as second,
    ):
        writer = ResolverSubscriptionStore(first, ProvenanceStore(first))
        other = ResolverSubscriptionStore(second, ProvenanceStore(second))
        subscription = writer.subscribe(str(source))
        read_local = transport._read_local

        def unsubscribe_during_fetch(path: str) -> bytes:
            assert other.unsubscribe(subscription.id)
            return read_local(path)

        with patch.object(transport, "_read_local", side_effect=unsubscribe_during_fetch):
            with pytest.raises(ResolverSubscriptionError, match="changed during refresh"):
                writer.refresh(subscription.id)
        assert writer.subscriptions() == ()
        assert ProvenanceStore(second).sources(digest) == ()
        tombstone = cast(
            "dict[str, object]",
            second.call("sources.resolvers.get", id=subscription.id)["subscription"],
        )
        assert tombstone["deleted"] is True


def test_provenance_preserves_large_records_merges_and_layer_withdrawals(tmp_path: Path) -> None:
    digest = "blake3:" + "55" * 32
    op = "sources.provenance."
    license_text = "L" * (17 * 1024 * 1024)
    expected: dict[str, object] = {
        "digest": digest,
        "sources": [],
        "license": license_text,
        "note": "",
        "metadata": {},
        "future": {"keep": True},
    }
    with LibraryClient(tmp_path) as helper, LibraryClient(tmp_path) as reader:
        assert "transport.chunks" in cast("list[str]", helper.hello["capabilities"])
        assert helper.call(op + "add", record=expected) == {"record": expected}
        assert reader.call(op + "get", digest=digest) == {"record": expected}
        helper.call(op + "replace", source="mask", records=[{"digest": digest, "license": "short"}])
        helper.call(
            op + "replace", source="notes", records=[{"digest": digest, "note": "N" * 70_000}]
        )
        assert (
            cast("dict[str, object]", reader.call(op + "get", digest=digest)["record"])["license"]
            == "short"
        )
        assert helper.call(op + "remove", source="mask") == {"removed": True}
        expected["note"] = "N" * 70_000
        assert reader.call(op + "get", digest=digest) == {"record": expected}
        assert reader.call(op + "list", limit=1) == {"items": [expected], "cursor": None}
        assert reader.call(op + "get", digest=digest, source="")["record"] == {
            **expected,
            "note": "",
        }
        helper.call(op + "replace", source="mask", records=[{"digest": digest, "license": "short"}])
        helper.call(op + "replace", source="mask", records=[])
        assert reader.call(op + "get", digest=digest) == {"record": expected}
        helper.call(op + "remove", source="notes")
        expected["note"] = ""
        assert reader.call(op + "get", digest=digest) == {"record": expected}
    with LibraryClient(tmp_path) as reopened:
        assert reopened.call(op + "get", digest=digest) == {"record": expected}


@pytest.mark.parametrize("filename", ["library.sqlite", "provenance.json", "resolver-indexes.json"])
def test_provenance_refuses_legacy_sources_without_hiding_them(
    tmp_path: Path, filename: str
) -> None:
    original = b"legacy source bytes"
    (tmp_path / filename).write_bytes(original)
    with LibraryClient(tmp_path) as helper:
        for action in ("add", "replace", "remove", "get", "list", "layers"):
            with pytest.raises(LibraryError) as failure:
                helper.call("sources.provenance." + action)
            assert failure.value.code == "migration"
    assert (tmp_path / filename).read_bytes() == original


def test_acquisition_receipts_are_shared_immutable_and_paged_without_retaining_bytes(
    tmp_path: Path,
) -> None:
    digest, other_digest = "blake3:" + "ab" * 32, "blake3:" + "cd" * 32
    receipt = {
        "version": 1,
        "receiptId": "12" * 16,
        "digest": digest,
        "sizeBytes": 0,
        "sourceType": "declarative-resolver",
        "sourceId": "index-one",
        "sourceRevision": "v1",
        "listedUrl": "https://example.invalid/listed",
        "finalUrl": "https://mirror.invalid/bytes",
        "fetchedAt": 12.0,
        "futureField": {"preserve": True},
    }
    with LibraryClient(tmp_path) as first, LibraryClient(tmp_path) as second:
        assert "sources.receipts" in cast("list[str]", first.hello["capabilities"])
        assert first.call("sources.receipts.get", receiptId="12" * 16) == {"receipt": None}
        assert first.call("sources.receipts.list")["items"] == []
        reserved = first.call("references.acquire")
        with ThreadPoolExecutor(max_workers=2) as executor:
            writes = [
                executor.submit(helper.call, "sources.receipts.put", receipt=receipt)
                for helper in (first, second)
            ]
            assert sorted(cast("bool", write.result()["created"]) for write in writes) == [
                False,
                True,
            ]
        first.call("references.release", lease=reserved["lease"])
        assert second.call("sources.receipts.get", receiptId="12" * 16) == {"receipt": receipt}
        changes = first.call("changes")
        for position, change in enumerate(
            (
                {"futureField": {"preserve": 1}},
                {"fetchedAt": 13.0},
                {"digest": other_digest},
                {"version": True},
                {"receiptId": "bad"},
                {"sizeBytes": True},
                {"sizeBytes": -1},
                {"sourceType": "unknown"},
                {"sourceId": " untrimmed"},
                {"sourceRevision": "bad\n"},
                {"listedUrl": "http://example.invalid/bytes"},
                {"finalUrl": "https://example.invalid/bytes?private=value"},
                {"finalUrl": "https://example.invalid/bytes#fragment"},
                {"finalUrl": "https://example.invalid:invalid/bytes"},
                {"fetchedAt": True},
                {"fetchedAt": -1.0},
                {"fetchedAt": 10**400},
                {"futureField": "x" * (128 * 1024)},
            )
        ):
            candidate = {**receipt, **({"receiptId": "56" * 16} if position >= 3 else {}), **change}
            with pytest.raises(LibraryError) as invalid:
                first.call("sources.receipts.put", receipt=candidate)
            assert invalid.value.code == ("conflict" if position < 3 else "invalid")
        assert first.call("sources.receipts.get", receiptId="12" * 16) == {"receipt": receipt}
        assert first.call("changes") == changes
        later = {**receipt, "receiptId": "34" * 16}
        unrelated = {**receipt, "receiptId": "23" * 16, "digest": other_digest}
        for document in (later, unrelated):
            second.call("sources.receipts.put", receipt=document)
        page = first.call("sources.receipts.list", digest=digest, limit=1)
        assert page["items"] == [receipt]
        assert page["cursor"] is not None
        next_page = second.call(
            "sources.receipts.list", digest=digest, limit=1, cursor=page["cursor"]
        )
        assert next_page == {"items": [later], "cursor": None}
        with pytest.raises(LibraryError) as wrong_digest:
            first.call("sources.receipts.list", digest=other_digest, cursor=page["cursor"])
        assert wrong_digest.value.code == "invalid"
        assert first.call("sources.receipts.list", digest=other_digest) == {
            "items": [unrelated],
            "cursor": None,
        }
        assert first.call("sources.receipts.list")["items"] == [receipt, unrelated, later]
        for limit in (True, 0, 101):
            with pytest.raises(LibraryError):
                first.call("sources.receipts.list", limit=limit)
        with pytest.raises(LibraryError):
            first.call("sources.receipts.get", receiptId="12" * 16, scope="private")
        reserved = first.call("references.acquire")
        assert reserved["digests"] == []
        first.call("references.release", lease=reserved["lease"])
        events = [
            item
            for item in cast("list[dict[str, object]]", first.call("changes")["items"])
            if item["kind"] == "acquisition-receipt"
        ]
        assert [(item["id"], item["revision"]) for item in events] == [
            ("12" * 16, 1),
            ("34" * 16, 1),
            ("23" * 16, 1),
        ]
    with LibraryClient(tmp_path) as reopened:
        assert reopened.call("sources.receipts.get", receiptId="12" * 16) == {"receipt": receipt}
        assert reopened.call("sources.receipts.list", digest=digest)["items"] == [receipt, later]


@pytest.mark.parametrize(
    "filename",
    [
        "provenance.json",
        "resolver-indexes.json",
        "public-acquisition-receipts.json",
    ],
)
def test_source_receipts_refuse_to_hide_legacy_sources(tmp_path: Path, filename: str) -> None:
    path = tmp_path / filename
    path.write_text('{"existing": true}\n')
    original = path.read_bytes()
    with LibraryClient(tmp_path) as helper:
        for operation, arguments in (
            ("sources.receipts.put", {"receipt": {}}),
            ("sources.receipts.get", {"receiptId": "12" * 16}),
            ("sources.receipts.list", {}),
        ):
            with pytest.raises(LibraryError) as refusal:
                helper.call(operation, **arguments)
            assert refusal.value.code == "migration"
    assert path.read_bytes() == original


def test_permissions_and_revocations_survive_other_helpers_and_restarts(tmp_path: Path) -> None:
    principal = "owner"
    with LibraryClient(tmp_path) as first, LibraryClient(tmp_path) as second:
        assert {"permissions", "delegations"} <= set(cast("list[str]", first.hello["capabilities"]))
        assert first.call("permissions.get", principalId=principal) == {
            "permissions": {"principalId": principal, "revision": 0, "values": {}}
        }
        first.call(
            "permissions.patch",
            principalId=principal,
            revision=0,
            patch={"execute": False, "future-category": True},
        )
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    helper.call,
                    "permissions.patch",
                    principalId=principal,
                    revision=1,
                    patch={category: False},
                )
                for helper, category in [(first, "edit"), (second, "read")]
            ]
            winners = 0
            for future in futures:
                try:
                    future.result()
                    winners += 1
                except LibraryError as error:
                    assert error.code == "stale"
            assert winners == 1
        current = first.call("permissions.get", principalId=principal)
        values = cast("dict[str, object]", current["permissions"])["values"]
        assert cast("dict[str, object]", values)["execute"] is False
        assert cast("dict[str, object]", values)["future-category"] is True
        before = first.call("changes")
        for arguments in (
            {"revision": True, "patch": {"execute": True}},
            {"revision": 2, "patch": {"execute": 1}},
            {"revision": 2, "patch": {"execute": True, "": False}},
        ):
            with pytest.raises(LibraryError):
                first.call("permissions.patch", principalId=principal, **arguments)
        assert first.call("permissions.get", principalId=principal) == current
        assert first.call("changes") == before
        reserved = first.call("references.acquire")
        second.call("permissions.patch", principalId="other", revision=0, patch={"execute": True})
        delegation = {
            "id": "delegation-a",
            "principalId": principal,
            "agentPrincipalId": "agent:a",
            "credentialHash": "ab" * 32,
            "scope": "local",
            "displayName": "Example agent",
            "sessionId": None,
            "createdAt": 12.0,
            "expiresAt": 30.0,
            "futureField": {"preserve": True},
        }
        assert second.call("delegations.create", record=delegation) == {"created": True}
        first.call("references.release", lease=reserved["lease"])
        assert first.call("delegations.create", record=delegation) == {"created": False}
        public = {key: value for key, value in delegation.items() if key != "credentialHash"}
        assert first.call("delegations.lookup", credentialHash="ab" * 32) == {
            "delegation": {**public, "revokedAt": None}
        }
        assert first.call("delegations.get", id="delegation-a") == {
            "delegation": {**public, "revokedAt": None}
        }
        assert first.call("delegations.lookup", credentialHash="cd" * 32) == {"delegation": None}
        before = first.call("changes")
        for change in (
            {"id": "other-id"},
            {"agentPrincipalId": "other-agent"},
            {"credentialHash": "cd" * 32},
            {"revokedAt": None},
            {"token": "not-a-credential"},
            {"credentialHash": "invalid"},
            {"id": "x" * 513, "agentPrincipalId": "agent:c", "credentialHash": "ef" * 32},
            {"createdAt": True},
            {"expiresAt": 11.0},
        ):
            with pytest.raises(LibraryError):
                first.call("delegations.create", record={**delegation, **change})
        assert first.call("changes") == before
        assert first.call("delegations.revoke", principalId="stranger", id="delegation-a") == {
            "revoked": False
        }
        first.call(
            "delegations.create",
            record={
                **delegation,
                "id": "delegation-b",
                "agentPrincipalId": "agent:b",
                "credentialHash": "cd" * 32,
            },
        )
        page = first.call("delegations.list", principalId=principal, limit=1)
        assert page["items"] == [public]
        assert page["cursor"] is not None
        with pytest.raises(LibraryError) as wrong_owner:
            first.call("delegations.list", principalId="stranger", cursor=page["cursor"])
        assert wrong_owner.value.code == "invalid"
        assert (
            cast(
                "list[dict[str, object]]",
                first.call(
                    "delegations.list",
                    principalId=principal,
                    cursor=page["cursor"],
                    limit=1,
                )["items"],
            )[0]["id"]
            == "delegation-b"
        )
        assert first.call("delegations.revoke", principalId=principal, id="delegation-a") == {
            "revoked": True
        }
        revoked = second.call("delegations.get", id="delegation-a")
        assert type(cast("dict[str, object]", revoked["delegation"])["revokedAt"]) is float
        assert first.call("delegations.create", record=delegation) == {"created": False}
        before = first.call("changes")
        assert second.call("delegations.revoke", principalId=principal, id="delegation-a") == {
            "revoked": False
        }
        assert first.call("changes") == before
        assert first.call("delegations.get", id="delegation-a") == revoked
        assert second.call("delegations.lookup", credentialHash="ab" * 32) == revoked
        assert [
            item["id"]
            for item in cast(
                "list[dict[str, object]]",
                second.call(
                    "delegations.list",
                    principalId=principal,
                )["items"],
            )
        ] == ["delegation-b"]
    with LibraryClient(tmp_path) as restarted:
        assert restarted.call("permissions.get", principalId=principal) == current
        assert restarted.call("delegations.lookup", credentialHash="ab" * 32) == revoked
        assert restarted.call("delegations.create", record=delegation) == {"created": False}
    legacy = tmp_path / "principals.sqlite"
    legacy.write_bytes(b"existing permissions and revocations")
    with LibraryClient(tmp_path) as blocked:
        for operation, arguments in (
            ("permissions.get", {"principalId": principal}),
            ("permissions.patch", {"principalId": principal, "revision": 2, "patch": {}}),
            ("delegations.get", {"id": "delegation-a"}),
            ("delegations.lookup", {"credentialHash": "ab" * 32}),
            ("delegations.list", {"principalId": principal}),
            ("delegations.create", {"record": delegation}),
            ("delegations.revoke", {"principalId": principal, "id": "delegation-a"}),
        ):
            with pytest.raises(LibraryError) as refusal:
                blocked.call(operation, **arguments)
            assert refusal.value.code == "migration"
    assert legacy.read_bytes() == b"existing permissions and revocations"


def test_settings_refuse_growth_before_committing_an_unreadable_record(tmp_path: Path) -> None:
    identity = {"environment": 'record"' + "\\" * 100}
    values = {"first": "a" * (600 * 1024), "second": ""}
    result = {"settings": {**identity, "revision": 2, "values": values}}
    overhead = len(json.dumps({"result": result}).encode("utf-8")) + 1
    values["second"] = "b" * (1024 * 1024 - overhead)
    assert len(json.dumps({"result": result}).encode("utf-8")) + 1 == 1024 * 1024
    with LibraryClient(tmp_path) as helper:
        helper.call("settings.patch", **identity, revision=0, patch={"first": values["first"]})
        assert (
            helper.call(
                "settings.patch", **identity, revision=1, patch={"second": values["second"]}
            )
            == result
        )
        before = helper.call("changes")
        with pytest.raises(LibraryError) as oversized:
            helper.call(
                "settings.patch", **identity, revision=2, patch={"second": values["second"] + "b"}
            )
        assert oversized.value.code == "invalid"
        assert helper.call("settings.get", **identity) == result
        assert helper.call("changes") == before
    with LibraryClient(tmp_path) as restarted:
        assert restarted.call("settings.get", **identity) == result
        assert restarted.call("settings.patch", **identity, revision=2, remove=[["second"]]) == {
            "settings": {**identity, "revision": 3, "values": {"first": values["first"]}}
        }


@pytest.mark.parametrize("filename", ["settings.json", "pack-settings/example.json"])
def test_settings_refuse_unmigrated_files_without_hiding_them(
    tmp_path: Path, filename: str
) -> None:
    legacy = tmp_path / filename
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text('{"existing-preference": true}\n')
    before = legacy.read_bytes()
    with LibraryClient(tmp_path) as helper:
        for operation in ("settings.get", "settings.patch"):
            arguments: dict[str, object] = {"environment": "existing"}
            if operation == "settings.patch":
                arguments.update(revision=0, patch={})
            with pytest.raises(LibraryError) as blocked:
                helper.call(operation, **arguments)
            assert blocked.value.code == "migration"
    assert legacy.read_bytes() == before


def test_result_claim_transfers_roots_atomically_without_metadata_revival(tmp_path: Path) -> None:
    first_id, second_id, token = "12" * 16, "34" * 16, "56" * 16
    source, other, derived, child = ["blake3:" + value * 32 for value in ("12", "34", "56", "78")]
    offer = {
        "token": token,
        "jobRef": "run",
        "attemptId": 1,
        "nodeId": "node",
        "outputId": "image",
        "publication": 1,
        "digests": [source],
        "complete": True,
    }
    key = {"scope": "private", "ownerKind": "shown-result", "ownerId": "first"}
    with (
        LibraryClient(tmp_path) as first,
        LibraryClient(tmp_path) as second,
        LibraryClient(tmp_path) as collector,
    ):
        assert "results.claim" in cast("list[str]", first.hello["capabilities"])
        for helper, engine in ((first, first_id), (second, second_id)):
            helper.call("references.engine.attach", engine=engine)
            helper.call("references.engine.patch", revision=1, patch={"complete": True})
            helper.call("retention.clients.initialize", pristine=True)
        client = first.call("retention.clients.open", principalId="principal")
        client_key = {"principalId": "principal", "generation": client["generation"]}
        binding = dict(client_key, holderClasses=[1])
        first.call(
            "retention.clients.acknowledge",
            **client_key,
            revision=client["revision"],
            holders=client["holders"],
        )
        first.call("results.offers.put", scope="private", offer=offer)
        second.call("results.offers.put", scope="private", offer=dict(offer, digests=[other]))
        rendition = {"documentDigest": source, "digests": [derived]}
        first.call(
            "results.renditions.put",
            scope="private",
            token=token,
            cacheKey="view",
            rendition=rendition,
        )
        first.call("dependencies.put", digest=derived, dependencies=[{"digest": child}])
        assert first.call(
            "results.claim", **dict(key, scope="other"), token=token, client=binding
        ) == {"owner": None}
        with pytest.raises(LibraryError) as wrong_client:
            first.call(
                "results.claim", **key, token=token, client=dict(binding, generation="00" * 16)
            )
        assert wrong_client.value.code == "stale"
        assert first.call("retention.clients.get", **client_key)["revision"] == client["revision"]
        with pytest.raises(LibraryError) as wrong_kind:
            first.call(
                "results.claim", **dict(key, ownerKind="document"), token=token, client=binding
            )
        assert wrong_kind.value.code == "invalid"
        held = collector.call("references.acquire")
        assert held["digests"] == [source, other, derived, child]
        with closing(sqlite3.connect(tmp_path / "records.sqlite", timeout=0)) as db:
            db.execute("BEGIN IMMEDIATE")
            db.rollback()
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(first.call, "results.claim", **key, token=token, client=binding)
            try:
                with pytest.raises(TimeoutError):
                    pending.result(timeout=0.1)
                assert collector.call("projects.get", id="default")["project"] is not None
            finally:
                collector.call("references.release", lease=held["lease"])
            owner = pending.result(timeout=5)["owner"]
        assert owner == dict(
            key, engine=first_id, revision=1, digests=[source], complete=True, sourceToken=token
        )
        with pytest.raises(LibraryError) as duplicate:
            first.call("results.claim", **key, token=token, client=binding)
        assert duplicate.value.code == "stale"
        second_key = dict(key, ownerKind="preview", ownerId="second")
        second_owner = first.call("results.claim", **second_key, token=token, client=binding)[
            "owner"
        ]
        assert isinstance(second_owner, dict) and second_owner["sourceToken"] == token
        first.call("retention.owners.release", **key, revision=1, client=binding)
        with pytest.raises(LibraryError) as tombstone:
            first.call("results.claim", **key, token=token, client=binding)
        assert tombstone.value.code == "stale"
        status = first.call("retention.clients.get", **client_key)
        first.call(
            "retention.clients.acknowledge",
            **client_key,
            revision=status["revision"],
            holders=status["holders"],
        )
        held = collector.call("references.acquire")
        assert held["digests"] == [source, other, derived, child]
        collector.call("references.release", lease=held["lease"])
        edited = cast(
            "dict[str, object]",
            first.call(
                "retention.owners.put",
                **second_key,
                revision=1,
                client=binding,
                patch={"digests": [source]},
            )["owner"],
        )
        assert "sourceToken" not in edited
        status = first.call("retention.clients.get", **client_key)
        first.call(
            "retention.clients.acknowledge",
            **client_key,
            revision=status["revision"],
            holders=status["holders"],
        )
        held = collector.call("references.acquire")
        assert held["digests"] == [source, other]
        collector.call("references.release", lease=held["lease"])
        assert first.call(
            "results.claim", **dict(key, ownerId="late"), token=token, client=binding
        ) == {"owner": None}
        first.call("retention.owners.release", **second_key, revision=2, client=binding)
        status = first.call("retention.clients.get", **client_key)
        first.call(
            "retention.clients.acknowledge",
            **client_key,
            revision=status["revision"],
            holders=status["holders"],
        )
        assert first.call(
            "results.renditions.put",
            scope="private",
            token=token,
            cacheKey="late",
            rendition=rendition,
        ) == {"rendition": None, "created": False}
        assert first.call("results.offers.put", scope="private", offer=offer)["created"] is False
        held = collector.call("references.acquire")
        assert held["digests"] == [other]
        collector.call("references.release", lease=held["lease"])
    with LibraryClient(tmp_path) as restarted:
        manifest = cast(
            "dict[str, object]",
            restarted.call("references.engine.attach", engine=first_id)["manifest"],
        )
        old_client = restarted.call("retention.clients.get", **client_key)
        rotated = restarted.call(
            "retention.clients.open", **client_key, revision=old_client["revision"]
        )
        client_key = dict(client_key, generation=rotated["generation"])
        binding = dict(client_key, holderClasses=[1])
        restarted.call(
            "retention.clients.acknowledge",
            **client_key,
            revision=rotated["revision"],
            holders=rotated["holders"],
        )
        restarted.call(
            "references.engine.patch", revision=manifest["revision"], patch={"complete": True}
        )
        assert restarted.call(
            "results.claim", **dict(key, ownerId="after-restart"), token=token, client=binding
        ) == {"owner": None}
        held = restarted.call("references.acquire")
        assert held["digests"] == [other]
        restarted.call("references.release", lease=held["lease"])
        incomplete_token = "ab" * 16
        restarted.call(
            "results.offers.put",
            scope="private",
            offer=dict(offer, token=incomplete_token, complete=False),
        )
        incomplete_key = dict(key, ownerId="incomplete")
        incomplete = cast(
            "dict[str, object]",
            restarted.call(
                "results.claim", **incomplete_key, token=incomplete_token, client=binding
            )["owner"],
        )
        assert incomplete["complete"] is False
        with pytest.raises(LibraryError) as blocked:
            restarted.call("references.acquire")
        assert blocked.value.code == "references"
        restarted.call("retention.owners.release", **incomplete_key, revision=1, client=binding)
        status = restarted.call("retention.clients.get", **client_key)
        restarted.call(
            "retention.clients.acknowledge",
            **client_key,
            revision=status["revision"],
            holders=status["holders"],
        )
        held = restarted.call("references.acquire")
        assert held["digests"] == [other]
        restarted.call("references.release", lease=held["lease"])
    assert not (tmp_path / "library.sqlite").exists()


def test_released_result_revival_verifies_bytes_and_keeps_vault_owned_through_claim(
    tmp_path: Path,
) -> None:
    from dinkster_assets.identity import digest_bytes
    from dinkster_assets.produced import ProducedAssetInventory
    from dinkster_assets.records import verified_result_source
    from dinkster_assets.scratch import claim_inactive_produced_vault
    from dinkster_assets.vault import AssetVault

    root = tmp_path / "library"
    vault = AssetVault(tmp_path / "vault")
    ProducedAssetInventory.initialize_vault(vault.root, existing_empty=True)
    inventory = ProducedAssetInventory(vault.root)
    payloads = [b"source", b"rendition", b"dependency", b"mounted grandchild"]
    digests = [digest_bytes(value) for value in payloads]
    paths: list[Path] = []
    for digest, payload in zip(digests[:3], payloads[:3], strict=True):
        inventory.pending(digest, len(payload))
        with vault.writer(digest) as writer:
            writer.write(payload)
            paths.append(writer.commit())
        inventory.adopted(digest, len(payload))
    mounted = tmp_path / "mounted.bin"
    mounted.write_bytes(payloads[3])
    paths.append(mounted)

    class Resolver:
        def resolve(self, digest: str) -> Path | None:
            return mounted if digest == digests[3] else None

    engine, token = "12" * 16, "34" * 16
    key = {"scope": "private", "ownerKind": "preview", "ownerId": "first"}
    with LibraryClient(root) as helper, LibraryClient(root) as other:
        assert "results.sources" in cast("list[str]", helper.hello["capabilities"])
        helper.call("references.engine.attach", engine=engine)
        other.call("references.engine.attach", engine="56" * 16)
        helper.call("references.engine.patch", revision=1, patch={"complete": True})
        other.call("references.engine.patch", revision=1, patch={"complete": True})
        for client in (helper, other):
            client.call("retention.clients.initialize", pristine=True)
        opened = helper.call("retention.clients.open", principalId="reader")
        client_key = {"principalId": "reader", "generation": opened["generation"]}
        binding = dict(client_key, holderClasses=[1])

        def acknowledge() -> None:
            status = helper.call("retention.clients.get", **client_key)
            helper.call(
                "retention.clients.acknowledge",
                **client_key,
                revision=status["revision"],
                holders=status["holders"],
            )

        acknowledge()
        helper.call(
            "results.offers.put",
            scope="private",
            offer={
                "token": token,
                "jobRef": "run",
                "attemptId": 1,
                "nodeId": "node",
                "outputId": "image",
                "publication": 1,
                "digests": [digests[0]],
                "complete": True,
            },
        )
        rendition = {"documentDigest": digests[1], "digests": [digests[1]]}
        helper.call(
            "results.renditions.put",
            scope="private",
            token=token,
            cacheKey="view",
            rendition=rendition,
        )
        helper.call("dependencies.put", digest=digests[1], dependencies=[{"digest": digests[2]}])
        helper.call("dependencies.put", digest=digests[2], dependencies=[{"digest": digests[3]}])
        assert other.call("results.sources.acquire", scope="private", token=token) == {
            "source": None
        }
        other.call("results.sources.release")
        with verified_result_source(helper, "other", token, vault=vault) as proof:
            assert proof is None
        with verified_result_source(helper, "private", token, vault=vault) as proof:
            assert proof is not None and proof == {}
            helper.call("results.claim", **key, token=token, client=binding, **proof)
        helper.call("retention.owners.release", **key, revision=1, client=binding)
        acknowledge()
        assert helper.call(
            "results.claim", **dict(key, ownerId="metadata"), token=token, client=binding
        ) == {"owner": None}
        for path, payload in zip(paths, payloads, strict=True):
            path.with_name(path.name + ".verified.json").unlink(missing_ok=True)
            path.unlink()
            with verified_result_source(
                helper, "private", token, vault=vault, resolver=Resolver()
            ) as proof:
                assert proof is None
            path.write_bytes(payload + b"wrong")
            with verified_result_source(
                helper, "private", token, vault=vault, resolver=Resolver()
            ) as proof:
                assert proof is None
            path.write_bytes(payload)
        with verified_result_source(
            helper, "private", token, vault=vault, resolver=Resolver()
        ) as proof:
            assert proof is not None and proof == {"verifiedDigests": sorted(digests)}
            with closing(sqlite3.connect(root / "records.sqlite", timeout=0)) as db:
                db.execute("BEGIN IMMEDIATE")
                db.rollback()
            with claim_inactive_produced_vault(vault.root) as inactive:
                assert inactive is False
            with pytest.raises(LibraryError) as reserved:
                other.call("references.acquire")
            assert reserved.value.code == "busy"
            new_digest = digest_bytes(b"late dependency")
            helper.call(
                "dependencies.put", digest=digests[3], dependencies=[{"digest": new_digest}]
            )
            assert helper.call(
                "results.claim",
                **dict(key, ownerId="changed"),
                token=token,
                client=binding,
                **proof,
            ) == {"owner": None}
        with pytest.raises(LibraryError) as expired:
            helper.call(
                "results.claim",
                **dict(key, ownerId="expired"),
                token=token,
                client=binding,
                **proof,
            )
        assert expired.value.code == "invalid"
        with verified_result_source(
            helper, "private", token, vault=vault, resolver=Resolver()
        ) as proof:
            assert proof is None
        inventory.pending(new_digest, len(b"late dependency"))
        with vault.writer(new_digest) as writer:
            writer.write(b"late dependency")
            writer.commit()
        inventory.adopted(new_digest, len(b"late dependency"))
        with verified_result_source(
            helper, "private", token, vault=vault, resolver=Resolver()
        ) as proof:
            assert proof is not None and proof == {
                "verifiedDigests": sorted([*digests, new_digest])
            }
            result = helper.call(
                "results.claim",
                **dict(key, ownerId="revived"),
                token=token,
                client=binding,
                **proof,
            )
            owner = cast("dict[str, object]", result["owner"])
            assert owner["digests"] == [digests[0]] and owner["sourceToken"] == token
        acknowledge()
        held = helper.call("references.acquire")
        assert held["digests"] == sorted([*digests, new_digest])
        helper.call("references.release", lease=held["lease"])
        helper.call(
            "retention.owners.release", **dict(key, ownerId="revived"), revision=1, client=binding
        )
        acknowledge()
        unbound_payload = b"new rendition after release"
        unbound_digest = digest_bytes(unbound_payload)
        inventory.pending(unbound_digest, len(unbound_payload))
        with vault.writer(unbound_digest) as writer:
            writer.write(unbound_payload)
            writer.commit()
        inventory.adopted(unbound_digest, len(unbound_payload))
        new_rendition = {"documentDigest": unbound_digest, "digests": [unbound_digest]}
        with verified_result_source(
            helper, "private", token, vault=vault, resolver=Resolver()
        ) as proof:
            assert proof is not None
            with pytest.raises(LibraryError) as unbound:
                helper.call(
                    "results.renditions.put",
                    scope="private",
                    token=token,
                    cacheKey="revived",
                    rendition=new_rendition,
                    **proof,
                )
            assert unbound.value.code == "invalid"
            assert helper.call(
                "results.renditions.get", scope="private", token=token, cacheKey="revived"
            ) == {"rendition": None}
            assert helper.call(
                "results.renditions.put",
                scope="private",
                token=token,
                cacheKey="revived",
                rendition=new_rendition,
            ) == {"rendition": None, "created": False}
            result = helper.call(
                "results.claim",
                **dict(key, ownerId="rendition-reader"),
                token=token,
                client=binding,
                **proof,
            )
            assert result["owner"] is not None
            assert (
                helper.call(
                    "results.renditions.put",
                    scope="private",
                    token=token,
                    cacheKey="revived",
                    rendition=new_rendition,
                )["created"]
                is True
            )
        acknowledge()
        held = helper.call("references.acquire")
        assert held["digests"] == sorted([*digests, new_digest, unbound_digest])
        helper.call("references.release", lease=held["lease"])
        helper.call(
            "retention.owners.release",
            **dict(key, ownerId="rendition-reader"),
            revision=1,
            client=binding,
        )
        acknowledge()
        held = helper.call("references.acquire")
        assert held["digests"] == []
        helper.call("references.release", lease=held["lease"])
        with claim_inactive_produced_vault(vault.root) as inactive:
            assert inactive is True
        with pytest.raises(LibraryError) as invalid:
            other.call("results.sources.acquire", scope="private", token="bad")
        assert invalid.value.code == "invalid"
        held = helper.call("references.acquire")
        helper.call("references.release", lease=held["lease"])
        other.call("results.sources.acquire", scope="private", token=token)
        os.kill(other.pid, signal.SIGTERM)
        other.close()
        held = helper.call("references.acquire")
        assert held["digests"] == []
        helper.call("references.release", lease=held["lease"])
    assert not (root / "library.sqlite").exists()


def test_retention_census_keeps_engine_generations_exact_holders_and_revision_roots(
    tmp_path: Path,
) -> None:
    first_id, second_id = "12" * 16, "34" * 16
    first_digest, second_digest, child = ["blake3:" + value * 32 for value in ("12", "34", "56")]
    owner_key = {"scope": "private", "ownerKind": "document", "ownerId": "same-owner"}
    with (
        LibraryClient(tmp_path) as first,
        LibraryClient(tmp_path) as second,
        LibraryClient(tmp_path) as collector,
    ):
        assert "retention" in cast("list[str]", first.hello["capabilities"])
        clients: list[dict[str, object]] = []
        for client, engine in ((first, first_id), (second, second_id)):
            client.call("references.engine.attach", engine=engine)
            client.call("references.engine.patch", revision=1, patch={"complete": True})
            assert client.call("retention.clients.initialize", pristine=True) == {
                "closed": True,
                "legacy": False,
            }
            clients.append(
                client.call("retention.clients.open", principalId="principal", revision=0)
            )
        first_client, second_client = clients
        first_key = {"principalId": "principal", "generation": first_client["generation"]}
        second_key = {"principalId": "principal", "generation": second_client["generation"]}
        first_binding = dict(first_key, holderClasses=[1, 8])
        second_binding = dict(second_key, holderClasses=[3])
        with pytest.raises(LibraryError) as foreign:
            first.call("retention.clients.get", **second_key)
        assert foreign.value.code == "stale"
        with pytest.raises(LibraryError) as pending:
            collector.call("references.acquire")
        assert pending.value.code == "references"
        first_owner = first.call(
            "retention.owners.put",
            **owner_key,
            revision=0,
            patch={"digests": [first_digest, first_digest], "complete": True},
            client=first_binding,
        )["owner"]
        assert first_owner == dict(
            owner_key, engine=first_id, revision=1, digests=[first_digest], complete=True
        )
        assert second.call("retention.owners.get", **owner_key) == {"owner": None}
        assert first.call("retention.owners.get", **dict(owner_key, scope="other")) == {
            "owner": None
        }
        for classes in ([True], [0], [9], [1, 1], []):
            with pytest.raises(LibraryError) as invalid:
                first.call(
                    "retention.owners.put",
                    **owner_key,
                    revision=1,
                    patch={"complete": True},
                    client=dict(first_binding, holderClasses=classes),
                )
            assert invalid.value.code == "invalid"
        before_invalid = first.call("retention.clients.get", **first_key)
        for patch in ({"complete": 1}, {"digests": ["invalid"]}, {"engine": second_id}):
            with pytest.raises(LibraryError) as invalid:
                first.call(
                    "retention.owners.put",
                    **owner_key,
                    revision=1,
                    patch=patch,
                    client=first_binding,
                )
            assert invalid.value.code == "invalid"
        assert first.call("retention.clients.get", **first_key) == before_invalid
        assert first.call("retention.owners.get", **owner_key) == {"owner": first_owner}
        second_owner = second.call(
            "retention.owners.put",
            **owner_key,
            revision=0,
            patch={"digests": [second_digest], "complete": True},
            client=second_binding,
        )["owner"]
        for client, key, owner, classes in (
            (first, first_key, first_owner, ("1", "8")),
            (second, second_key, second_owner, ("3",)),
        ):
            status = client.call("retention.clients.get", **key)
            expected = {str(n): [owner] if str(n) in classes else [] for n in range(1, 9)}
            assert status["holders"] == expected
            assert status["complete"] is False
            bad_holders = json.loads(
                json.dumps(expected).replace('"complete": true', '"complete": 1')
            )
            with pytest.raises(LibraryError) as changed:
                client.call(
                    "retention.clients.acknowledge",
                    **key,
                    revision=status["revision"],
                    holders=bad_holders,
                )
            assert changed.value.code == "stale"
            client.call(
                "retention.clients.acknowledge",
                **key,
                revision=status["revision"],
                holders=expected,
            )
        first.call("dependencies.put", digest=first_digest, dependencies=[{"digest": child}])
        held = collector.call("references.acquire")
        assert held["digests"] == [first_digest, second_digest, child]
        with closing(sqlite3.connect(tmp_path / "records.sqlite", timeout=0)) as db:
            db.execute("BEGIN IMMEDIATE")
            db.rollback()
        with ThreadPoolExecutor(max_workers=1) as pool:
            write = pool.submit(
                first.call,
                "retention.owners.release",
                **owner_key,
                revision=1,
                client=first_binding,
            )
            try:
                with pytest.raises(TimeoutError):
                    write.result(timeout=0.1)
                assert second.call("retention.clients.get", **second_key)["complete"] is True
            finally:
                collector.call("references.release", lease=held["lease"])
            released = write.result(timeout=5)["owner"]
        assert released == dict(owner_key, engine=first_id, revision=2, digests=[], complete=True)
        for client, revision in (
            (None, 2),
            (first_binding, 1),
            (dict(first_binding, holderClasses=[1]), 2),
        ):
            with pytest.raises(LibraryError) as stale:
                first.call(
                    "retention.owners.put",
                    **owner_key,
                    revision=revision,
                    patch={"digests": []},
                    client=client,
                )
            assert stale.value.code == "stale"
        status = first.call("retention.clients.get", **first_key)
        first.call(
            "retention.clients.acknowledge",
            **first_key,
            revision=status["revision"],
            holders=status["holders"],
        )
        assert first.call("retention.clients.admit", **first_key) == {}
        admitted = first.call("retention.clients.get", **first_key)
        assert admitted["pendingRequests"] == 1 and admitted["complete"] is False
        with pytest.raises(LibraryError):
            first.call(
                "retention.clients.acknowledge",
                **first_key,
                revision=admitted["revision"],
                holders=admitted["holders"],
            )
        with pytest.raises(LibraryError):
            first.call("retention.clients.open", **first_key, revision=admitted["revision"])
        first.call("retention.clients.settle", **first_key)
        settled = first.call("retention.clients.get", **first_key)
        first.call("retention.clients.settle", **first_key)
        assert first.call("retention.clients.get", **first_key) == settled
        first.call(
            "retention.clients.acknowledge",
            **first_key,
            revision=settled["revision"],
            holders=settled["holders"],
        )
        first.call("retention.clients.admit", **first_key, stream=True)
        stream = first.call("retention.clients.get", **first_key)
        assert stream["activeStreams"] == 1
        with pytest.raises(LibraryError):
            collector.call("references.acquire")
        first.call("retention.clients.settle", **first_key, stream=True)
        settled = first.call("retention.clients.get", **first_key)
        assert settled["activeStreams"] == 0 and settled["revision"] != stream["revision"]
        first.call("retention.clients.settle", **first_key, stream=True)
        assert first.call("retention.clients.get", **first_key) == settled
        first.call(
            "retention.clients.acknowledge",
            **first_key,
            revision=settled["revision"],
            holders=settled["holders"],
        )
        held = collector.call("references.acquire")
        assert held["digests"] == [second_digest]
        collector.call("references.release", lease=held["lease"])
        # This models a record written by newer library code before an older patch.
        with closing(sqlite3.connect(tmp_path / "records.sqlite")) as db:
            db.execute(
                "UPDATE retained_owners SET document=json_set(document,'$.futureField','kept') "
                "WHERE engine=?",
                (first_id,),
            )
            db.execute(
                "UPDATE retention_clients SET document="
                "json_set(document,'$.futureClientField','kept') WHERE engine=?",
                (first_id,),
            )
            db.commit()
        first.call(
            "retention.owners.put",
            **owner_key,
            revision=2,
            client=first_binding,
            patch={"digests": [first_digest]},
        )
        second_status = second.call("retention.clients.get", **second_key)
        os.kill(first.pid, signal.SIGTERM)
        first.close()
        with LibraryClient(tmp_path) as restarted:
            manifest = cast(
                "dict[str, object]",
                restarted.call("references.engine.attach", engine=first_id)["manifest"],
            )
            status = restarted.call("retention.clients.get", **first_key)
            assert status["connected"] is False and status["complete"] is False
            assert status["futureClientField"] == "kept"
            assert status["pendingRequests"] == status["activeStreams"] == 0
            assert second.call("retention.clients.get", **second_key) == second_status
            assert second.call("retention.owners.get", **owner_key)["owner"] == second_owner
            retained = cast(
                "dict[str, object]", restarted.call("retention.owners.get", **owner_key)["owner"]
            )
            assert retained["digests"] == [first_digest] and retained["futureField"] == "kept"
            with pytest.raises(LibraryError):
                restarted.call(
                    "retention.clients.acknowledge",
                    **first_key,
                    revision=status["revision"],
                    holders=status["holders"],
                )
            rotated = restarted.call(
                "retention.clients.open", **first_key, revision=status["revision"]
            )
            assert rotated["clientId"] == first_client["clientId"]
            assert rotated["futureClientField"] == "kept"
            assert rotated["generation"] != first_key["generation"]
            with pytest.raises(LibraryError):
                restarted.call("retention.clients.get", **first_key)
            new_key = dict(first_key, generation=rotated["generation"])
            new_binding = dict(new_key, holderClasses=[1, 8])
            retained = cast(
                "dict[str, object]",
                restarted.call(
                    "retention.owners.put",
                    **owner_key,
                    revision=3,
                    client=new_binding,
                    patch={"complete": False},
                )["owner"],
            )
            assert retained["futureField"] == "kept"
            status = restarted.call("retention.clients.get", **new_key)
            with pytest.raises(LibraryError) as incomplete:
                restarted.call(
                    "retention.clients.acknowledge",
                    **new_key,
                    revision=status["revision"],
                    holders=status["holders"],
                )
            assert incomplete.value.code == "invalid"
            restarted.call(
                "retention.owners.put",
                **owner_key,
                revision=4,
                client=new_binding,
                patch={"complete": True},
            )
            status = restarted.call("retention.clients.get", **new_key)
            restarted.call(
                "retention.clients.acknowledge",
                **new_key,
                revision=status["revision"],
                holders=status["holders"],
            )
            restarted.call(
                "references.engine.patch", revision=manifest["revision"], patch={"complete": True}
            )
            held = collector.call("references.acquire")
            assert held["digests"] == [first_digest, second_digest, child]
            collector.call("references.release", lease=held["lease"])
            restarted.call("retention.clients.disconnect", **new_key)
            disconnected = restarted.call("retention.clients.get", **new_key)
            assert disconnected["connected"] is False
            retained = cast(
                "dict[str, object]", restarted.call("retention.owners.get", **owner_key)["owner"]
            )
            assert retained["digests"] == [first_digest]
            with pytest.raises(LibraryError):
                collector.call("references.acquire")
    assert not (tmp_path / "library.sqlite").exists()


@pytest.mark.parametrize("unknown", ["nonpristine", "admission", "owner"])
def test_retention_unknown_universe_cannot_be_reinitialized_as_empty(
    tmp_path: Path, unknown: str
) -> None:
    engine = "12" * 16
    digest = "blake3:" + "34" * 32
    with LibraryClient(tmp_path) as helper:
        helper.call("references.engine.attach", engine=engine)
        helper.call("references.engine.patch", revision=1, patch={"complete": True})
        helper.call("retention.clients.initialize", pristine=unknown != "nonpristine")
        if unknown == "admission":
            helper.call("retention.clients.admit", principalId="legacy")
        if unknown == "owner":
            helper.call(
                "retention.owners.put",
                scope="private",
                ownerKind="preview",
                ownerId="old-tab",
                revision=0,
                patch={"digests": [digest], "complete": True},
            )
        universe = helper.call("retention.clients.initialize", pristine=True)
        assert universe["legacy"] is True
        with pytest.raises(LibraryError) as blocked:
            helper.call("references.acquire")
        assert blocked.value.code == "references"
        if unknown == "owner":
            retained = cast(
                "dict[str, object]",
                helper.call(
                    "retention.owners.get", scope="private", ownerKind="preview", ownerId="old-tab"
                )["owner"],
            )
            assert retained["digests"] == [digest]
    with LibraryClient(tmp_path) as reopened:
        reopened.call("references.engine.attach", engine=engine)
        assert reopened.call("retention.clients.initialize", pristine=True)["legacy"] is True


@pytest.mark.parametrize("filename", ["history.sqlite", "library.sqlite"])
def test_helper_records_refuse_to_hide_unmigrated_data(tmp_path: Path, filename: str) -> None:
    legacy = tmp_path / filename
    with closing(sqlite3.connect(legacy)) as database:
        database.execute("CREATE TABLE history (run_id TEXT PRIMARY KEY)")
        database.execute("INSERT INTO history VALUES ('existing-run')")
        database.commit()
    before = legacy.read_bytes()
    with LibraryClient(tmp_path) as helper:
        if filename == "history.sqlite":
            operations = [
                (operation, {"scope": "local", "id": "existing-run"})
                for operation in ("history.list", "history.get", "history.delete", "history.clear")
            ]
        else:
            helper.call("references.engine.attach", engine="12" * 16)
            operations = [
                ("sources.receipts.put", {"receipt": {}}),
                ("sources.receipts.get", {"receiptId": "12" * 16}),
                ("sources.receipts.list", {}),
                ("permissions.get", {"principalId": "local"}),
                ("permissions.patch", {"principalId": "local", "revision": 0, "patch": {}}),
                ("delegations.get", {"id": "existing"}),
                ("delegations.lookup", {"credentialHash": "ab" * 32}),
                ("delegations.list", {"principalId": "local"}),
                ("delegations.create", {"record": {}}),
                ("delegations.revoke", {"principalId": "local", "id": "existing"}),
                ("settings.get", {"environment": "existing"}),
                ("settings.patch", {"environment": "existing", "revision": 0, "patch": {}}),
                ("retention.clients.initialize", {"pristine": True}),
                ("retention.clients.open", {"principalId": "local"}),
                ("retention.clients.get", {"principalId": "local", "generation": "12" * 16}),
                ("retention.owners.put", {"scope": "local"}),
                ("retention.owners.get", {"scope": "local"}),
                ("retention.owners.release", {"scope": "local"}),
                ("results.claim", {"scope": "local", "token": "12" * 16}),
                ("results.sources.acquire", {"scope": "local", "token": "12" * 16}),
                ("results.terminal.put", {"scope": "local", "jobRef": "run"}),
                ("results.terminal.get", {"scope": "local", "jobRef": "run"}),
                ("results.offers.put", {"scope": "local", "offer": {}}),
                ("results.offers.get", {"scope": "local", "token": "12" * 16}),
                ("results.renditions.put", {"scope": "local", "token": "12" * 16}),
                ("results.renditions.get", {"scope": "local", "token": "12" * 16}),
                ("grants.put", {"scope": "local", "grant": {}}),
                (
                    "grants.get",
                    {"scope": "local", "digest": "blake3:" + "ab" * 32, "kind": "media/image"},
                ),
                ("assets.record", {"scope": "local", "observation": {}}),
                ("records.create", {"scope": "local", "record": {}}),
                ("records.get", {"scope": "local", "id": "existing-record"}),
                (
                    "records.patch",
                    {"scope": "local", "id": "existing-record", "revision": 1, "patch": {}},
                ),
                ("records.delete", {"scope": "local", "id": "existing-record", "revision": 1}),
                ("dependencies.get", {"digest": "blake3:" + "ab" * 32}),
                ("dependencies.put", {"digest": "blake3:" + "ab" * 32, "dependencies": []}),
                ("derivations.get", {"digest": "blake3:" + "ab" * 32}),
                (
                    "derivations.put",
                    {"digest": "blake3:" + "ab" * 32, "manifest": {}, "digests": []},
                ),
            ]
        operations.append(("references.acquire", {}))
        for operation, arguments in operations:
            with pytest.raises(LibraryError) as refused:
                helper.call(operation, **arguments)
            assert refused.value.code == "migration"
        assert helper.call("projects.list")["items"]
    assert legacy.read_bytes() == before


def test_helper_describes_api_without_opening_a_library(tmp_path: Path) -> None:
    library = tmp_path / "not-created"
    result = subprocess.run(
        [sys.executable, "-m", "dinkster_library", "--describe", "--library", str(library)],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    description = json.loads(result.stdout)
    assert (description["apiMin"], description["apiMax"]) == (1, 1)
    assert {"index.scan", "index.progress", "index.scan.selected"} <= set(
        description["capabilities"]
    )
    assert not library.exists()
    with LibraryClient(library) as helper:
        assert helper.hello["capabilities"] == description["capabilities"]


def test_record_browse_keeps_scope_ownership_filters_and_query_bound_pages(tmp_path: Path) -> None:
    from dinkster_assets import LibraryStore, digest_bytes

    library = tmp_path / "library"
    store = LibraryStore(library / "library.sqlite")
    workflow = "application/x-dinkster-workflow+json"
    try:
        first = store.create(
            "local", "First film", digest_bytes(b"first"), workflow, labels=["film"]
        )
        second = store.create(
            "local", "Second film", digest_bytes(b"second"), workflow, labels=["film"]
        )
        legacy = store.create("local", "Legacy", digest_bytes(b"legacy"), workflow)
        generic = store.create("local", "Film note", digest_bytes(b"note"), "text/plain")
        store.create("remote", "Private film", digest_bytes(b"remote"), workflow)
    finally:
        store.close()
    before = (library / "library.sqlite").read_bytes()
    with LibraryClient(library) as helper:
        helper.call("projects.register", id="film")
        helper.call("workflows.record", scope="local", id=first.id, revision=0, project="film")
        helper.call("workflows.record", scope="local", id=second.id, revision=0, project="film")
        needs = helper.call("projects.needs", id="film", includeWorkflows=True)
        assert sorted(
            cast("list[dict[str, str]]", needs["workflows"]), key=lambda item: item["id"]
        ) == sorted(
            [
                {"scope": "local", "id": item.id, "digest": item.digest, "name": item.name}
                for item in (first, second)
            ],
            key=lambda item: item["id"],
        )
        assert "workflows" not in helper.call("projects.needs", id="film")
        with pytest.raises(LibraryError) as invalid:
            helper.call("projects.needs", id="film", includeWorkflows="true")
        assert invalid.value.code == "invalid"
        page = helper.call("records.list", scope="local", project="film", limit=1)
        page_records = cast("list[dict[str, object]]", page["records"])
        assert [row["id"] for row in page_records] == [second.id]
        assert page_records == [second.to_wire() | {"project": "film"}]
        assert page["cursor"] is not None
        last = helper.call(
            "records.list", scope="local", project="film", cursor=page["cursor"], limit=2
        )
        assert [row["id"] for row in cast("list[dict[str, object]]", last["records"])] == [first.id]
        assert last["cursor"] is None
        defaults = cast(
            "list[dict[str, object]]",
            helper.call("records.list", scope="local", project="default")["records"],
        )
        assert [row["id"] for row in defaults] == [legacy.id]
        assert defaults[0]["project"] == "default"
        assert (
            helper.call("records.list", scope="local", project="film", mediaType="text/plain")[
                "records"
            ]
            == []
        )
        selected = helper.call(
            "records.list", scope="local", q="FIRST", label="film", mediaType=workflow
        )
        selected_records = cast("list[dict[str, object]]", selected["records"])
        assert [row["id"] for row in selected_records] == [first.id]
        assert selected_records[0]["labels"] == ["film"]
        all_rows = cast(
            "list[dict[str, object]]", helper.call("records.list", scope="local")["records"]
        )
        assert [row["id"] for row in all_rows] == [generic.id, legacy.id, second.id, first.id]
        assert "project" not in all_rows[0]
        for arguments in (
            {"scope": "remote", "project": "film", "cursor": page["cursor"]},
            {"scope": "local", "project": "default", "cursor": page["cursor"]},
            {"scope": "local", "project": "film", "q": "first", "cursor": page["cursor"]},
            {"scope": "local", "cursor": "bad"},
            {"scope": "local", "limit": True},
            {"scope": "local", "limit": 0},
            {"scope": "local", "limit": 201},
            {"scope": "local", "q": 1},
            {"scope": "local", "unknown": "ignored"},
        ):
            with pytest.raises(LibraryError) as invalid:
                helper.call("records.list", **arguments)
            assert invalid.value.code == "invalid"
        helper.call("projects.delete", id="film", revision=1)
        with pytest.raises(LibraryError) as missing:
            helper.call("records.list", scope="local", project="film")
        assert missing.value.code == "missing"
    assert (library / "library.sqlite").read_bytes() == before
    with LibraryClient(tmp_path / "empty") as helper:
        assert helper.call("records.list", scope="local") == {"records": [], "cursor": None}
    assert not (tmp_path / "empty" / "library.sqlite").exists()


def test_asset_browse_groups_aliases_and_binds_scope_project_and_filters(tmp_path: Path) -> None:
    from dinkster_assets import AssetRef, LibraryStore, digest_bytes

    library = tmp_path / "library"
    store = LibraryStore(library / "library.sqlite")
    first = AssetRef(digest_bytes(b"first"), "stored.bin", 5, "image/png", "output/stored.bin")
    second = AssetRef(digest_bytes(b"second"), "second.bin", 6, "image/png")
    try:
        original = store.record_asset(
            "local",
            first,
            original_name="first.png",
            kind="media/image",
            origin={"kind": "upload"},
            project="film",
            created=10,
        )
        alias = store.record_asset(
            "local",
            first,
            original_name="alias.png",
            kind="media/image",
            origin={"kind": "produced", "job": "scene", "node": "save"},
            project="film",
            created=30,
        )
        store.record_asset(
            "local",
            second,
            original_name="second.png",
            kind="media/image",
            origin={"kind": "upload"},
            project="film",
            created=20,
        )
        store.record_asset(
            "remote",
            second,
            original_name="private.png",
            kind="media/image",
            origin={"kind": "upload"},
            project="film",
            created=90,
        )
        store.record_asset(
            "local",
            second,
            original_name="other.png",
            kind="media/image",
            origin={"kind": "upload"},
            project="default",
            created=100,
        )
    finally:
        store.close()
    before = (library / "library.sqlite").read_bytes()
    with LibraryClient(library) as helper:
        helper.call("projects.register", id="film")
        page = helper.call("assets.list", scope="local", project="film", limit=1)
        assets = cast("list[dict[str, object]]", page["assets"])
        assert assets == [
            {
                "asset": first.to_wire(),
                "aliases": ["first.png", "stored.bin", "alias.png"],
                "observations": [original.to_wire(), alias.to_wire()],
            }
        ]
        last = helper.call(
            "assets.list", scope="local", project="film", limit=1, cursor=page["cursor"]
        )
        assert [item["asset"] for item in cast("list[dict[str, object]]", last["assets"])] == [
            second.to_wire()
        ]
        assert last["cursor"] is None
        assert (
            helper.call(
                "assets.list",
                scope="local",
                project="film",
                q="FIRST",
                kind="media/image",
                origin="upload",
                since=10,
                until=10,
            )["assets"]
            == assets
        )
        assert (
            helper.call("assets.list", scope="local", project="film", since=11, until=19)["assets"]
            == []
        )
        assert (
            helper.call("assets.list", scope="local", project="film", kind="model/custom")["assets"]
            == []
        )
        for changed in (
            {"scope": "remote"},
            {"project": "default"},
            {"q": "first"},
            {"kind": "media/image"},
            {"origin": "upload"},
            {"since": 0},
            {"until": 200},
        ):
            with pytest.raises(LibraryError) as invalid:
                helper.call(
                    "assets.list",
                    **({"scope": "local", "project": "film", "cursor": page["cursor"]} | changed),
                )
            assert invalid.value.code == "invalid"
        for invalid_query in (
            {"kind": "image"},
            {"since": "10"},
            {"since": True},
            {"since": 20, "until": 10},
            {"limit": 0},
            {"limit": 201},
            {"q": []},
            {"cursor": "bad"},
            {"source": "unrecorded"},
        ):
            with pytest.raises(LibraryError) as invalid:
                helper.call("assets.list", scope="local", **invalid_query)
            assert invalid.value.code == "invalid"
        helper.call("projects.delete", id="film", revision=1)
        with pytest.raises(LibraryError) as missing:
            helper.call("assets.list", scope="local", project="film")
        assert missing.value.code == "missing"
        assert len(cast("list[object]", helper.call("assets.list", scope="local")["assets"])) == 1
    assert (library / "library.sqlite").read_bytes() == before
    with LibraryClient(tmp_path / "empty-assets") as helper:
        assert helper.call("assets.list", scope="local") == {"assets": [], "cursor": None}
    assert not (tmp_path / "empty-assets" / "library.sqlite").exists()


def test_project_registration_and_revision_conflicts(tmp_path: Path) -> None:
    with LibraryClient(tmp_path) as first, LibraryClient(tmp_path) as second:
        with ThreadPoolExecutor(2) as pool:
            registered = list(
                pool.map(
                    lambda client: client.call("projects.register", id="film", name="Film"),
                    (first, second),
                )
            )
        assert registered[0] == registered[1]
        assert first.call("projects.register", id="film", name="Ignored") == registered[0]
        renamed = second.call("projects.patch", id="film", revision=1, patch={"name": "New"})
        assert cast("dict[str, object]", renamed["project"])["revision"] == 2
        assert first.call("projects.get", id="film") == renamed
        with pytest.raises(LibraryError) as stale:
            first.call("projects.patch", id="film", revision=1, patch={"name": "Lost"})
        assert stale.value.code == "stale"
        with pytest.raises(LibraryError) as protected:
            first.call("projects.delete", id="default", revision=1)
        assert protected.value.code == "invalid"
        assert first.call("projects.list", limit=1)["cursor"] == "default"
        assert (
            len(
                cast(
                    "list[object]", first.call("projects.list", cursor="default", limit=1)["items"]
                )
            )
            == 1
        )
    with LibraryClient(tmp_path) as reopened:
        assert reopened.call("projects.get", id="film") == renamed
    assert not (tmp_path / "library.sqlite").exists()


def test_project_identity_fields_filters_and_remembered_deletion(tmp_path: Path) -> None:
    with LibraryClient(tmp_path / "first") as first, LibraryClient(tmp_path / "second") as second:
        default = cast("dict[str, object]", first.call("projects.get", id="default")["project"])
        other = cast("dict[str, object]", second.call("projects.get", id="default")["project"])
        assert str(uuid.UUID(cast(str, default["uid"]))) == default["uid"]
        assert default["uid"] != other["uid"]
        for identifier in ("a", "0", "a" * 64):
            assert first.call("projects.register", id=identifier)["project"] is not None
        for identifier in ("", "a" * 65, "Film", "-film", "film_name", "film/name"):
            with pytest.raises(LibraryError) as invalid:
                first.call("projects.register", id=identifier)
            assert invalid.value.code == "invalid"
        registered = cast(
            "dict[str, object]", first.call("projects.register", id="film", name="Film")["project"]
        )
        patched = cast(
            "dict[str, object]",
            first.call(
                "projects.patch",
                id="film",
                revision=1,
                patch={
                    "packs": {"extra": ["grain", "color"]},
                    "environment": {"id": "editing", "lastSnapshot": "12", "lastRunAt": "now"},
                    "accountProject": {"issuer": "https://example.test", "id": "shared-film"},
                },
            )["project"],
        )
        assert patched["uid"] == registered["uid"]
        assert patched["created"] == registered["created"]
        assert patched["revision"] == 2
        assert patched["packs"] == {"extra": ["grain", "color"]}
        assert patched["accountProject"] == {"issuer": "https://example.test", "id": "shared-film"}
        assert first.call("projects.list", environment="editing")["items"] == [patched]
        assert first.call("projects.list", environment="other")["items"] == []
        updated = cast(
            "dict[str, object]",
            first.call(
                "projects.patch",
                id="film",
                revision=2,
                patch={"environment": {"lastSnapshot": "13"}},
            )["project"],
        )
        assert updated["environment"] == {"id": "editing", "lastSnapshot": "13", "lastRunAt": "now"}
        for patch in ({"uid": other["uid"]}, {"packs": {"extra": [3]}}, {"environment": {"id": 2}}):
            with pytest.raises(LibraryError) as invalid:
                first.call("projects.patch", id="film", revision=3, patch=patch)
            assert invalid.value.code == "invalid"
        assert first.call("projects.get", id="film")["project"] == updated
        deleted = cast(
            "dict[str, object]", first.call("projects.delete", id="film", revision=3)["project"]
        )
        assert deleted["deleted"] is True and deleted["revision"] == 4
        assert first.call("projects.list", environment="editing")["items"] == []
        assert first.call("projects.list", environment="editing", includeDeleted=True)["items"] == [
            deleted
        ]
        assert first.call("projects.register", id="film")["project"] == deleted
        changes = cast("list[dict[str, object]]", first.call("changes")["items"])
        film_changes = [change["revision"] for change in changes if change["id"] == "film"]
        assert film_changes == [1, 2, 3, 4]
    with LibraryClient(tmp_path / "first") as reopened:
        assert reopened.call("projects.get", id="film")["project"] == deleted


def test_project_environment_selection_does_not_copy_another_environments_run(
    tmp_path: Path,
) -> None:
    with LibraryClient(tmp_path) as helper:
        assert "projects.environment-selection" in cast("list[str]", helper.hello["capabilities"])
        helper.call("projects.register", id="film")
        recorded = cast(
            "dict[str, object]",
            helper.call(
                "projects.patch",
                id="film",
                revision=1,
                patch={"environment": {"id": "editing", "lastSnapshot": "12", "lastRunAt": "now"}},
            )["project"],
        )
        selected = cast(
            "dict[str, object]",
            helper.call(
                "projects.patch",
                id="film",
                revision=2,
                patch={"environment": {"id": "fork"}},
            )["project"],
        )
        assert selected["environment"] == {"id": "fork", "lastSnapshot": None, "lastRunAt": None}
        assert helper.call("projects.list", environment="fork")["items"] == [selected]
        assert helper.call("projects.list", environment="editing")["items"] == []
        for environment in (
            {"lastSnapshot": "3"},
            {"lastRunAt": "later"},
            {"id": "other", "lastSnapshot": "3"},
            {"id": "", "lastSnapshot": None, "lastRunAt": None},
        ):
            with pytest.raises(LibraryError) as invalid:
                helper.call(
                    "projects.patch", id="film", revision=3, patch={"environment": environment}
                )
            assert invalid.value.code == "invalid"
            assert helper.call("projects.get", id="film")["project"] == selected
        with pytest.raises(LibraryError) as stale:
            helper.call(
                "projects.patch", id="film", revision=2, patch={"environment": {"id": "editing"}}
            )
        assert stale.value.code == "stale"
        settled = cast(
            "dict[str, object]",
            helper.call(
                "projects.patch",
                id="film",
                revision=3,
                patch={"environment": {"lastSnapshot": "3", "lastRunAt": "later"}},
            )["project"],
        )
        assert settled["environment"] == {"id": "fork", "lastSnapshot": "3", "lastRunAt": "later"}
        same = cast(
            "dict[str, object]",
            helper.call(
                "projects.patch", id="film", revision=4, patch={"environment": {"id": "fork"}}
            )["project"],
        )
        assert same["environment"] == settled["environment"]
        settled = same
        assert recorded["environment"] == {
            "id": "editing",
            "lastSnapshot": "12",
            "lastRunAt": "now",
        }
    with LibraryClient(tmp_path) as reopened:
        assert reopened.call("projects.get", id="film")["project"] == settled


def test_api_removal_impact_pages_engines_and_their_projects(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    register = tomllib.loads((root / "docs/library-api.toml").read_text())
    with LibraryClient(tmp_path) as helper:
        version = register["versions"][str(helper.hello["apiMax"])]
        assert set(version["capabilities"]) == set(cast("list[str]", helper.hello["capabilities"]))
        assert (root / version["adapter"]).is_file()
        for environment, api in (("alpha", 1), ("middle", 2), ("omega", 3)):
            helper.call(
                "engines.seen",
                environment=environment,
                snapshot="10",
                engineVersion=f"build-{api}",
                apiVersion=api,
            )
        for identifier, snapshot in (("film", "10"), ("later", "11"), ("deleted", "10")):
            helper.call("projects.register", id=identifier, name=identifier.title())
            helper.call(
                "projects.patch",
                id=identifier,
                revision=1,
                patch={
                    "environment": {"id": "alpha", "lastSnapshot": snapshot, "lastRunAt": "now"}
                },
            )
        helper.call("projects.delete", id="deleted", revision=2)
        cursor: object = None
        items: list[dict[str, object]] = []
        while True:
            page = helper.call("compatibility.affected", apiMin=2, apiMax=2, limit=1, cursor=cursor)
            rows = cast("list[dict[str, object]]", page["items"])
            assert len(rows) == 1
            items.extend(rows)
            cursor = page["cursor"]
            if cursor is None:
                break
        assert [(row["kind"], row.get("id"), row.get("environment")) for row in items] == [
            ("project", "film", "alpha"),
            ("snapshot", None, "alpha"),
            ("snapshot", None, "omega"),
        ]
        assert items[0]["name"] == "Film"
        assert [(row["snapshot"], row["apiVersion"]) for row in items[1:]] == [("10", 1), ("10", 3)]
        assert helper.call("compatibility.affected", apiMin=1, apiMax=3)["items"] == []
        with pytest.raises(LibraryError) as invalid:
            helper.call("compatibility.affected", apiMin=3, apiMax=2)
        assert invalid.value.code == "invalid"


def test_helpers_share_verified_hashes_across_libraries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_assets import digest_bytes

    monkeypatch.setenv("DINKSTER_HOME", str(tmp_path / "account"))
    path = tmp_path / "model.bin"
    path.write_bytes(b"first model bytes")
    created = path.stat()
    # Keep modification and creation times distinct for Windows stat/fstat.
    os.utime(path, ns=(created.st_atime_ns, created.st_mtime_ns + 2_000_000_000))
    with LibraryClient(tmp_path / "first") as first, LibraryClient(tmp_path / "second") as second:
        with ThreadPoolExecutor(2) as pool:
            results = list(
                pool.map(lambda client: client.call("index.hash", path=str(path)), (first, second))
            )
        assert [result["digest"] for result in results] == [digest_bytes(b"first model bytes")] * 2
        assert sorted(result["hashed"] for result in results) == [False, True]
        assert first.call("index.hash", path=str(path)) == {
            "digest": digest_bytes(b"first model bytes"),
            "hashed": False,
        }
        before = path.stat()
        path.write_bytes(b"other model bytes")
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
        assert second.call("index.hash", path=str(path)) == {
            "digest": digest_bytes(b"other model bytes"),
            "hashed": True,
        }
        assert first.call("index.hash", path=str(path)) == {
            "digest": digest_bytes(b"other model bytes"),
            "hashed": False,
        }
        for invalid in ("relative.bin", str(tmp_path / "missing.bin"), str(tmp_path)):
            with pytest.raises(LibraryError):
                first.call("index.hash", path=invalid)
        assert first.call("projects.get", id="default")["project"] is not None
    with LibraryClient(tmp_path / "third") as third:
        assert third.call("index.hash", path=str(path)) == {
            "digest": digest_bytes(b"other model bytes"),
            "hashed": False,
        }


def _finished_scan(client: LibraryClient, **arguments: object) -> dict[str, object]:
    result = client.call("index.scan", **arguments)
    deadline = time.monotonic() + 15
    while result["state"] in ("queued", "scanning") and time.monotonic() < deadline:
        assert client.call("projects.get", id="default")["project"] is not None
        time.sleep(0.02)
        result = client.call("index.progress", id=result["id"])
    assert result["state"] == "complete", result
    return result


def test_helper_scans_share_account_hashes_and_publish_unchanged_worker_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_assets import IndexedAssetResolver, digest_bytes

    monkeypatch.setenv("DINKSTER_HOME", str(tmp_path / "account"))
    root = tmp_path / "models"
    root.mkdir()
    (root / "first.bin").write_bytes(b"first model")
    (root / "second.bin").write_bytes(b"a different second model")
    for ignored in (".hidden.bin", "put_model_here", "pending.partial", "Thumbs.db"):
        (root / ignored).write_bytes(b"not an asset")
    outputs = [tmp_path / name / "worker-state" / "models.json" for name in ("one", "two", "three")]
    with LibraryClient(tmp_path / "one") as first, LibraryClient(tmp_path / "two") as second:
        with ThreadPoolExecutor(2) as pool:
            results = list(
                pool.map(
                    lambda pair: _finished_scan(pair[0], root=str(root), output=str(pair[1])),
                    ((first, outputs[0]), (second, outputs[1])),
                )
            )
        assert sum(cast(int, result["hashed"]) for result in results) == 2
        assert [result["filesDone"] for result in results] == [2, 2]
        for output in outputs[:2]:
            rows = json.loads(output.read_text("utf-8"))
            assert set(rows) == {"first.bin", "second.bin"}
            resolver = IndexedAssetResolver(root, index_path=output)
            for name in ("first.bin", "second.bin"):
                expected = digest_bytes((root / name).read_bytes())
                assert rows[name]["digest"] == expected
                assert resolver.resolve(expected) == root / name
        assert _finished_scan(first, root=str(root), output=str(outputs[2]))["hashed"] == 0
        (root / "second.bin").write_bytes(b"changed second model")
        changed = _finished_scan(second, root=str(root), output=str(outputs[1]))
        assert changed["hashed"] == 1
        rows = json.loads(outputs[1].read_text("utf-8"))
        assert rows["second.bin"]["digest"] == digest_bytes(b"changed second model")
        assert rows["second.bin"]["missingVersions"][0]["digest"] == digest_bytes(
            b"a different second model"
        )
        before_selection = outputs[1].read_bytes()
        (root / "selected.bin").write_bytes(b"only the selected new file")
        (root / "unselected.bin").write_bytes(b"must not be scanned yet")
        selected = _finished_scan(
            second, root=str(root), output=str(outputs[1]), relative="selected.bin"
        )
        assert (selected["filesDone"], selected["hashed"]) == (1, 1)
        rows = json.loads(outputs[1].read_text("utf-8"))
        assert "unselected.bin" not in rows
        for name, row in json.loads(before_selection).items():
            assert rows[name] == row
        assert rows["selected.bin"]["digest"] == digest_bytes(b"only the selected new file")
        assert (
            _finished_scan(first, root=str(root), output=str(outputs[0]), relative="selected.bin")[
                "hashed"
            ]
            == 0
        )
        for excluded in (".hidden.bin", "pending.partial", "missing.bin"):
            result = _finished_scan(
                second, root=str(root), output=str(outputs[1]), relative=excluded
            )
            assert (result["filesDone"], result["hashed"]) == (0, 0)
        for unsafe in ("../outside.bin", "/absolute.bin", "C:/outside.bin", "a/../first.bin"):
            with pytest.raises(LibraryError) as invalid:
                second.call("index.scan", root=str(root), output=str(outputs[1]), relative=unsafe)
            assert invalid.value.code == "invalid"
        (root / "first.bin").unlink()
        assert _finished_scan(first, root=str(root), output=str(outputs[0]))["hashed"] == 1
        assert json.loads(outputs[0].read_text("utf-8"))["first.bin"]["missing"] is True


def test_helper_scan_wait_keeps_requests_live_and_resumes_committed_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    from filelock import FileLock

    home = tmp_path / "account"
    monkeypatch.setenv("DINKSTER_HOME", str(home))
    root = tmp_path / "models"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (root / "first.bin").write_bytes(b"committed before shutdown")
    (nested / "second.bin").write_bytes(b"not yet read before shutdown")
    locks = home / "index-locks"
    locks.mkdir(parents=True)
    key = hashlib.sha256(os.path.normcase(str(nested.resolve())).encode("utf-8")).hexdigest()
    output = tmp_path / "worker-state" / "models.json"
    with FileLock(locks / f"{key}.lock"):
        with LibraryClient(tmp_path / "library") as client:
            result = client.call("index.scan", root=str(root), output=str(output))
            deadline = time.monotonic() + 10
            while result["filesDone"] != 1 and time.monotonic() < deadline:
                time.sleep(0.02)
                result = client.call("index.progress", id=result["id"])
            assert result["filesDone"] == 1
            assert result["hashed"] == 1
            assert result["state"] == "scanning"
            with pytest.raises(LibraryError) as busy:
                client.call("index.scan", root=str(root), output=str(output))
            assert busy.value.code == "busy"
            assert client.call("projects.register", id="while-scanning")["project"] is not None
        # EOF must stop a helper even while its scan thread waits for a folder lock.
    with LibraryClient(tmp_path / "library") as restarted:
        result = _finished_scan(restarted, root=str(root), output=str(output))
        assert result["filesDone"] == 2
        assert result["hashed"] == 1
        assert restarted.call("projects.get", id="while-scanning")["project"] is not None


def test_helper_imports_verified_indexes_but_rehashes_stale_or_unverified_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_assets import AssetWriter, LocalAssetLibrary, MountDef, MountTable, digest_bytes

    monkeypatch.setenv("DINKSTER_HOME", str(tmp_path / "fresh-account"))
    root = tmp_path / "models"
    root.mkdir()
    path = root / "model.bin"
    path.write_bytes(b"original model")
    legacy = tmp_path / "library" / "asset-indexes" / "models.json"
    assert LocalAssetLibrary(root, index_path=legacy).scan() == 1
    original = legacy.read_bytes()
    mounts = MountTable()
    mounts.add(MountDef("models", root, mode="readwrite", kind="model/checkpoint"))
    written = AssetWriter(mounts).save_bytes(
        {"mount": "models", "prefix": "new"}, b"written without a scan", suffix=".bin"
    )
    output = tmp_path / "library" / "worker-state" / "models.json"
    with LibraryClient(tmp_path / "library") as client:
        assert (
            _finished_scan(client, root=str(root), output=str(output), legacy=[str(legacy)])[
                "hashed"
            ]
            == 0
        )
        assert legacy.read_bytes() == original
        rows = json.loads(output.read_text("utf-8"))
        assert rows[written.name]["digest"] == written.digest
        assert isinstance(rows[written.name]["publishedAtNs"], int)
        before = path.stat()
        path.write_bytes(b"changed model with a different length")
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert (
            _finished_scan(client, root=str(root), output=str(output), legacy=[str(legacy)])[
                "hashed"
            ]
            == 1
        )
        assert json.loads(output.read_text("utf-8"))["model.bin"]["digest"] == digest_bytes(
            path.read_bytes()
        )
    monkeypatch.setenv("DINKSTER_HOME", str(tmp_path / "another-fresh-account"))
    metadata = path.stat()
    legacy.write_text(
        json.dumps(
            {
                "model.bin": {
                    "digest": digest_bytes(b"wrong bytes"),
                    "size": metadata.st_size,
                    "mtimeNs": metadata.st_mtime_ns,
                }
            }
        ),
        "utf-8",
    )
    with LibraryClient(tmp_path / "other-library") as client:
        result = _finished_scan(
            client, root=str(root), output=str(tmp_path / "other-output.json"), legacy=[str(legacy)]
        )
        assert result["hashed"] == 1
        assert json.loads((tmp_path / "other-output.json").read_text("utf-8"))["model.bin"][
            "digest"
        ] == digest_bytes(path.read_bytes())


def test_shipped_list_is_a_copy_safe_hint_not_a_content_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    from dinkster_assets import IndexedAssetResolver, LocalAssetLibrary, digest_bytes
    from dinkster_assets.integrity import AssetIntegrityError, open_verified

    monkeypatch.setenv("DINKSTER_HOME", str(tmp_path / "fresh-account"))
    prepared = tmp_path / "prepared-models"
    prepared.mkdir()
    contents = {
        "valid.bin": b"a prepared model",
        "wrong.bin": b"original bytes",
        "resized.bin": b"old size",
        "retimed.bin": b"same size, new time",
        "invalid.bin": b"bad digest row",
    }
    for name, data in contents.items():
        (prepared / name).write_bytes(data)
    assert LocalAssetLibrary(prepared).scan() == len(contents)
    root = tmp_path / "copied-models"
    shutil.copytree(prepared, root, copy_function=shutil.copy2)
    shipped = root / ".dinkster-asset-index.json"
    original = json.loads(shipped.read_text("utf-8"))
    assert original["valid.bin"]["verification"]["inode"] != (root / "valid.bin").stat().st_ino
    preview = tmp_path / "first-worker-state" / "models.json"
    with LibraryClient(tmp_path / "first-library") as client:
        initial = _finished_scan(client, root=str(root), output=str(preview))
        assert (initial["filesDone"], initial["hashed"]) == (5, 0)
        catalog = LocalAssetLibrary(root, index_path=preview)
        assert catalog.refresh_index() == 5
        assert {entry.digest for entry in catalog.entries()} == {
            digest_bytes(data) for data in contents.values()
        }
    metadata = (root / "wrong.bin").stat()
    (root / "wrong.bin").write_bytes(b"replaced bytes")
    os.utime(root / "wrong.bin", ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
    (root / "resized.bin").write_bytes(b"a new and larger model")
    metadata = (root / "retimed.bin").stat()
    os.utime(root / "retimed.bin", ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 2_000_000_000))
    original["invalid.bin"]["digest"] = "not-a-digest"
    shipped.write_text(json.dumps(original), "utf-8")
    shipped_bytes = shipped.read_bytes()
    output = tmp_path / "worker-state" / "models.json"
    with LibraryClient(tmp_path / "library") as client:
        result = _finished_scan(client, root=str(root), output=str(output))
        assert (result["filesDone"], result["hashed"]) == (5, 3)
        rows = json.loads(output.read_text("utf-8"))
        resolver = IndexedAssetResolver(root, index_path=output)
        for name in ("valid.bin", "wrong.bin"):
            digest = digest_bytes(contents[name])
            assert rows[name]["digest"] == digest
            assert "verification" not in rows[name]
            resolution = resolver.resolve_asset(digest)
            assert resolution is not None
            assert resolution.path == root / name
            assert resolution.verification is None
            if name == "wrong.bin":
                with pytest.raises(AssetIntegrityError, match="digest_mismatch"):
                    open_verified(resolution.path, digest, resolution.verification)
            else:
                with open_verified(resolution.path, digest, resolution.verification) as handle:
                    assert handle.read() == contents[name]
        for name in ("resized.bin", "retimed.bin", "invalid.bin"):
            assert rows[name]["digest"] == digest_bytes((root / name).read_bytes())
            assert "verification" in rows[name]
        # Selected resolution must hash a hinted file, not promote its copied metadata.
        selected = _finished_scan(client, root=str(root), output=str(output), relative="wrong.bin")
        assert selected["hashed"] == 1
        rows = json.loads(output.read_text("utf-8"))
        assert rows["wrong.bin"]["digest"] == digest_bytes(b"replaced bytes")
        assert "verification" in rows["wrong.bin"]
        assert client.call("index.hash", path=str(root / "valid.bin")) == {
            "digest": digest_bytes(contents["valid.bin"]),
            "hashed": True,
        }
        assert _finished_scan(client, root=str(root), output=str(output))["hashed"] == 0
        assert shipped.read_bytes() == shipped_bytes
        # A replacement list cannot downgrade a known changed file to a hint.
        (root / "valid.bin").write_bytes(b"changed prepared model")
        metadata = (root / "valid.bin").stat()
        original["valid.bin"].update(size=metadata.st_size, mtimeNs=metadata.st_mtime_ns)
        shipped.write_text(json.dumps(original), "utf-8")
        shipped_bytes = shipped.read_bytes()
        assert _finished_scan(client, root=str(root), output=str(output))["hashed"] == 1
        assert json.loads(output.read_text("utf-8"))["valid.bin"]["digest"] == digest_bytes(
            b"changed prepared model"
        )
        assert shipped.read_bytes() == shipped_bytes


def test_rejected_api_does_not_open_library(tmp_path: Path) -> None:
    with pytest.raises(LibraryError, match="offers 1 to 1") as failure:
        LibraryClient(tmp_path, api=9)
    assert failure.value.code == "api"
    assert not (tmp_path / "records.sqlite").exists()


@pytest.mark.parametrize("hold_backup", [False, True])
def test_format_upgrade_waits_for_detach_and_backs_up_committed_records(
    tmp_path: Path, hold_backup: bool
) -> None:
    from filelock import FileLock

    document = {
        "id": "film",
        "uid": "42145cb2-30fd-43b6-a2bd-9abb20bcae86",
        "name": "Film",
        "revision": 7,
        "deleted": False,
        "packs": {"extra": []},
        "futureField": {"keep": True},
    }
    with closing(sqlite3.connect(tmp_path / "records.sqlite")) as database, database:
        database.executescript(
            "CREATE TABLE metadata (minimum_code INTEGER NOT NULL);"
            "INSERT INTO metadata VALUES (1);"
            "CREATE TABLE projects (id TEXT PRIMARY KEY, document TEXT NOT NULL);"
            "CREATE TABLE changes (sequence INTEGER PRIMARY KEY AUTOINCREMENT, "
            "kind TEXT NOT NULL,id TEXT NOT NULL,revision INTEGER NOT NULL);"
            "CREATE TABLE workflow_facts (scope TEXT NOT NULL,id TEXT NOT NULL,"
            "project TEXT NOT NULL,document TEXT NOT NULL,PRIMARY KEY(scope,id));"
            "CREATE TABLE engines (environment TEXT NOT NULL,snapshot TEXT NOT NULL,"
            "document TEXT NOT NULL,PRIMARY KEY(environment,snapshot));"
            "PRAGMA user_version=1;"
        )
        database.execute("INSERT INTO projects VALUES (?,?)", ("film", json.dumps(document)))
    with FileLock(tmp_path / "helper-older.lock"):
        with pytest.raises(LibraryError) as attached:
            LibraryClient(tmp_path)
        assert attached.value.code == "attached"
        assert not (tmp_path / "backups").exists()
    previous_backups = []
    for index in range(3):
        previous = tmp_path / "backups" / f"before-format-1-{index}" / "records.sqlite"
        previous.parent.mkdir(parents=True)
        with (
            closing(sqlite3.connect(tmp_path / "records.sqlite")) as source,
            closing(sqlite3.connect(previous)) as backup,
        ):
            source.backup(backup)
        timestamp = (index + 1) * 1_000_000_000
        os.utime(previous, ns=(timestamp, timestamp))
        previous_backups.append(previous)
    held = sqlite3.connect(previous_backups[0]) if hold_backup else None
    try:
        with LibraryClient(tmp_path) as helper:
            assert helper.hello["format"] == 2
            assert helper.call("projects.get", id="film") == {"project": document}
            changed = helper.call(
                "projects.patch", id="film", revision=7, patch={"name": "Renamed"}
            )
            assert cast("dict[str, object]", changed["project"])["futureField"] == {"keep": True}
        deferred = hold_backup and os.name == "nt"
        assert len(list((tmp_path / "backups").glob("*/records.sqlite"))) == (4 if deferred else 3)
        assert previous_backups[0].exists() == deferred
    finally:
        if held is not None:
            held.close()
    with LibraryClient(tmp_path) as helper:
        assert (
            cast("dict[str, object]", helper.call("projects.get", id="film")["project"])["name"]
            == "Renamed"
        )
    backups = list((tmp_path / "backups").glob("*/records.sqlite"))
    assert len(backups) == 3
    assert not previous_backups[0].exists()
    assert all(path.exists() for path in previous_backups[1:])
    newest = next(path for path in backups if path not in previous_backups)
    with closing(sqlite3.connect(newest)) as backup:
        assert backup.execute("PRAGMA user_version").fetchone()[0] == 1
        assert backup.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert (
            json.loads(
                backup.execute("SELECT document FROM projects WHERE id='film'").fetchone()[0]
            )
            == document
        )
    with LibraryClient(tmp_path):
        pass
    assert list((tmp_path / "backups").glob("*/records.sqlite")) == backups
    assert not (tmp_path / "library.sqlite").exists()


def test_client_refuses_pipe_reuse_after_a_malformed_framed_reply(tmp_path: Path) -> None:
    program = """import json, sys
for line in sys.stdin:
    request = json.loads(line)
    if request["operation"] == "broken":
        print(json.dumps({"chunk": "eA==", "offset": 1, "complete": False}), flush=True)
    else:
        print(json.dumps({"result": {"capabilities": ["transport.chunks"]}}), flush=True)
"""
    with LibraryClient(tmp_path, command=[sys.executable, "-c", program]) as client:
        with pytest.raises(LibraryError) as broken:
            client.call("broken")
        assert broken.value.code == "disconnected"
        with pytest.raises(LibraryError) as reuse:
            client.call("hello")
        assert reuse.value.code == "disconnected"


def test_client_keeps_ordinary_calls_compatible_with_helpers_without_chunks(tmp_path: Path) -> None:
    program = """import json, sys
for line in sys.stdin:
    request = json.loads(line)
    if request["operation"] == "hello":
        result = {"apiMin": 1, "apiMax": 1, "capabilities": []}
    else:
        result = {"arguments": request["arguments"], "transport": request.get("transport")}
    print(json.dumps({"result": result}), flush=True)
"""
    with LibraryClient(tmp_path, command=[sys.executable, "-c", program]) as client:
        assert client.call("echo", text="unchanged") == {
            "arguments": {"text": "unchanged"},
            "transport": None,
        }
        with pytest.raises(LibraryError) as oversized:
            client.call("echo", text="x" * (17 * 1024 * 1024))
        assert oversized.value.code == "message"
        assert client.call("echo", text="still usable")["arguments"] == {"text": "still usable"}


def test_pipe_chunks_are_atomic_ordered_and_disappear_with_their_helper(tmp_path: Path) -> None:
    from threading import Timer

    process = subprocess.Popen(
        [sys.executable, "-m", "dinkster_library", "--library", str(tmp_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    assert process.stdin is not None and process.stdout is not None
    request = b'{"api":1,"operation":"projects.register","arguments":{"id":"chunked"}}\n'
    watchdog = Timer(30, process.kill)
    watchdog.start()

    def frame(data: bytes, offset: object, complete: object) -> None:
        assert process.stdin is not None
        process.stdin.write(
            json.dumps(
                {
                    "chunk": base64.b64encode(data).decode(),
                    "offset": offset,
                    "complete": complete,
                }
            ).encode()
            + b"\n"
        )
        process.stdin.flush()

    try:
        with LibraryClient(tmp_path) as reader:
            for wrong_offset, final in ((1, True), (True, True), (1, False)):
                frame(request[:20], 0, False)
                assert reader.call("projects.get", id="chunked") == {"project": None}
                frame(request[20:] if final else request[20:40], wrong_offset, final)
                reply = json.loads(process.stdout.readline())
                assert reply["error"]["code"] == "message"
                if not final:
                    frame(request[40:], 40, True)
                    process.stdin.write(b'{"api":1,"operation":"hello","arguments":{}}\n')
                    process.stdin.flush()
                    assert json.loads(process.stdout.readline())["result"]["apiMin"] == 1
                assert reader.call("projects.get", id="chunked") == {"project": None}
            for invalid in (b"{\n", b"[]\n", b'{"padding":"' + b"x" * (16 * 1024 * 1024) + b'"}\n'):
                frame(request[:20], 0, False)
                frame(request[20:40], 1, False)
                assert json.loads(process.stdout.readline())["error"]["code"] == "message"
                if len(invalid) > 16 * 1024 * 1024:
                    process.stdin.write(invalid[:-3])
                    process.stdin.flush()
                    assert json.loads(process.stdout.readline())["error"]["code"] == "message"
                    process.stdin.write(invalid[-3:])
                else:
                    process.stdin.write(invalid)
                    process.stdin.flush()
                    assert json.loads(process.stdout.readline())["error"]["code"] == "message"
                process.stdin.write(b'{"api":1,"operation":"hello","arguments":{}}\n')
                process.stdin.flush()
                assert json.loads(process.stdout.readline())["result"]["apiMin"] == 1
                assert reader.call("projects.get", id="chunked") == {"project": None}
            frame(request[:20], 0, False)
            frame(request[20:], 20, True)
            assert json.loads(process.stdout.readline())["result"]["project"]["id"] == "chunked"
            # A normal API 1 caller still gets normal JSON lines after a transfer.
            process.stdin.write(b'{"api":1,"operation":"hello","arguments":{}}\n')
            process.stdin.flush()
            assert json.loads(process.stdout.readline())["result"]["apiMin"] == 1
            pending = request.replace(b"chunked", b"unfinished")
            frame(pending[:20], 0, False)
            process.stdin.close()
            assert process.wait(timeout=2) == 0
            assert reader.call("projects.get", id="unfinished") == {"project": None}
    finally:
        watchdog.cancel()
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stdin.close()
        process.stdout.close()


def test_pipe_eof_stops_helper_while_write_is_blocked(tmp_path: Path) -> None:
    process = subprocess.Popen(
        [sys.executable, "-m", "dinkster_library", "--library", str(tmp_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    assert process.stdin is not None and process.stdout is not None
    db = None
    try:
        process.stdin.write(b'{"api":1,"operation":"hello","arguments":{}}\n')
        process.stdin.flush()
        assert json.loads(process.stdout.readline())["result"]["apiMin"] == 1
        db = sqlite3.connect(tmp_path / "records.sqlite")
        db.execute("BEGIN IMMEDIATE")
        process.stdin.write(
            b'{"api":1,"operation":"projects.register","arguments":{"id":"blocked"}}\n'
        )
        process.stdin.flush()
        # Let the request reach SQLite's busy wait; the held write lock is
        # the timing control, not a deadline on how quickly the write runs.
        time.sleep(0.2)
        assert process.poll() is None
        process.stdin.close()
        assert process.wait(timeout=2) == 0
        db.rollback()
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert db.execute("SELECT id FROM projects WHERE id='blocked'").fetchone() is None
    finally:
        if db is not None:
            db.close()
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stdin.close()
        process.stdout.close()
