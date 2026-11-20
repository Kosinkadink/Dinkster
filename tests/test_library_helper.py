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
            assert pending_write.result(timeout=5) == {"record": later, "project": "default"}
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
                ("retention.clients.initialize", {"pristine": True}),
                ("retention.clients.open", {"principalId": "local"}),
                ("retention.clients.get", {"principalId": "local", "generation": "12" * 16}),
                ("retention.owners.put", {"scope": "local"}),
                ("retention.owners.get", {"scope": "local"}),
                ("retention.owners.release", {"scope": "local"}),
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
