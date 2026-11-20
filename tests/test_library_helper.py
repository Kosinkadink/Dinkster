"""Public pipe contracts for shared project records and helper lifetime."""

from __future__ import annotations

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
from typing import cast

import pytest
from dinkster_library import LibraryClient, LibraryError


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
            }
        second.call("history.append", scope="local", record=record("default", 99.0))
        private = dict(record("private", 100.0, "private"), assetDigests=["blake3:" + "ef" * 32])
        second.call("history.append", scope="private", record=private)
        duplicate = dict(documents[-1], finishedAt=200.0, futureField="do not overwrite")
        assert second.call("history.append", scope="local", project="film", record=duplicate) == {
            "record": documents[-1],
            "project": "film",
        }
        page = second.call("history.list", scope="local", project="film", limit=1)
        assert page["items"] == [documents[-1]]
        assert page["cursor"] is not None
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
        assert results[0] == results[1]
        assert results[0]["record"] in contenders
        assert first.call("history.get", scope="local", id="concurrent") == results[0]
        first.call(
            "history.append", scope="local", record=dict(record("unknown", 60.0), assetDigests=None)
        )
        with pytest.raises(LibraryError) as unknown:
            second.call("history.references.acquire")
        assert unknown.value.code == "references"
        assert first.call("history.delete", scope="local", id="unknown")["deleted"]
        with pytest.raises(LibraryError):
            first.call("history.references.acquire", limit=0)
        reserved = first.call("history.references.acquire", limit=1)
        assert reserved["digests"] == ["blake3:" + "ab" * 32]
        assert reserved["cursor"] == "blake3:" + "ab" * 32
        for client, operation, arguments in (
            (first, "history.clear", {"scope": "local"}),
            (second, "history.references.acquire", {}),
            (second, "history.references.release", {"lease": reserved["lease"]}),
        ):
            with pytest.raises(LibraryError):
                client.call(operation, **arguments)
        page = first.call(
            "history.references.page", lease=reserved["lease"], cursor=reserved["cursor"], limit=1
        )
        assert page["digests"] == ["blake3:" + "cd" * 32]
        assert page["cursor"] == "blake3:" + "cd" * 32
        page = first.call(
            "history.references.page", lease=reserved["lease"], cursor=page["cursor"], limit=1
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
                first.call("history.references.release", lease=reserved["lease"])
            assert pending_write.result(timeout=5) == {"record": later, "project": "default"}
        with pytest.raises(LibraryError):
            first.call("history.references.page", lease=reserved["lease"])
        first.call("history.references.acquire")
        os.kill(first.pid, signal.SIGTERM)
        first.close()
        reserved = second.call("history.references.acquire")
        second.call("history.references.release", lease=reserved["lease"])
    with LibraryClient(tmp_path) as reopened:
        assert reopened.call("history.get", scope="local", id="z") == {"record": None}
        assert reopened.call("history.get", scope="local", id="y")["record"] == documents[1]
        assert reopened.call("history.get", scope="local", id="default")["project"] == "default"
        assert reopened.call("history.clear", scope="private", before=100.0) == {"deleted": 0}
        assert reopened.call("history.clear", scope="private") == {"deleted": 1}
    assert not (tmp_path / "library.sqlite").exists()
    assert not (tmp_path / "history.sqlite").exists()


def test_helper_history_refuses_to_hide_unmigrated_runs(tmp_path: Path) -> None:
    legacy = tmp_path / "history.sqlite"
    with closing(sqlite3.connect(legacy)) as database:
        database.execute("CREATE TABLE history (run_id TEXT PRIMARY KEY)")
        database.execute("INSERT INTO history VALUES ('existing-run')")
        database.commit()
    before = legacy.read_bytes()
    with LibraryClient(tmp_path) as helper:
        for operation in ("history.list", "history.get", "history.delete", "history.clear"):
            with pytest.raises(LibraryError) as refused:
                helper.call(operation, scope="local", id="existing-run")
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
