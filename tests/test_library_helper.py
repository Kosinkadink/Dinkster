"""Public pipe contracts for shared project records and helper lifetime."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from typing import cast

import pytest
from dinkster_library import LibraryClient, LibraryError


def test_helper_describes_api_without_opening_a_library(tmp_path: Path) -> None:
    library = tmp_path / "not-created"
    result = subprocess.run(
        [sys.executable, "-m", "dinkster_library", "--describe", "--library", str(library)],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert json.loads(result.stdout) == {"apiMin": 1, "apiMax": 1}
    assert not library.exists()


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
        (root / "first.bin").unlink()
        assert _finished_scan(first, root=str(root), output=str(outputs[0]))["hashed"] == 0
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
