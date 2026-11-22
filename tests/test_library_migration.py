"""Inactive conversion preserves sources and resumes from the committed snapshot."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest
from dinkster_assets import LibraryStore, digest_bytes
from dinkster_library import LibraryClient, LibraryError
from dinkster_library._migration import stage_records
from dinkster_server import HistoryRecord, HistoryStore


def test_staging_preserves_legacy_records_and_does_not_activate(tmp_path: Path) -> None:
    root, account = tmp_path / "library", tmp_path / "account"
    store = LibraryStore(root / "library.sqlite")
    digest = digest_bytes(b"retained workflow")
    record = store.create("private", "Original", digest, "application/json", labels=["kept"])
    store.close()
    history = HistoryStore(root / "history.sqlite")
    finished = HistoryRecord(
        run_id="run-unique",
        scope="private",
        client_id="client",
        job_id="job",
        state="failed",
        priority=7,
        submitted_at=10,
        started_at=11,
        finished_at=13,
        principal_id="person",
        attempt=3,
        error={"message": "failure"},
        executed=2,
        cached=4,
        skipped=1,
        asset_digests=(digest,),
    )
    history.put(finished)
    history.close()
    with closing(sqlite3.connect(root / "library.sqlite")) as db, db:
        db.execute(
            "INSERT INTO asset_retention_clients VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("client", "person", "generation", 9, 0, 0, 2, 1, 123.5, 1),
        )
        db.execute("INSERT INTO asset_retention_quarantine VALUES (?,?,?)", ("client", digest, 47))
    with closing(sqlite3.connect(root / "principals.sqlite")) as db, db:
        db.execute(
            "CREATE TABLE principal_permissions (principal_id TEXT, category TEXT, enabled INT)"
        )
        db.execute(
            "INSERT INTO principal_permissions VALUES ('person','unknown-future-category',0)"
        )
    account.mkdir()
    settings = b'{"locations": [], "disabled": true}\n'
    (account / "model-settings.json").write_bytes(settings)
    sources = {p.name: p.read_bytes() for p in root.glob("*.sqlite")}
    with stage_records(root, account, "engine-one") as (staged, backup):
        saved = staged.db.execute("SELECT id,revision,document FROM saved_records").fetchone()
        assert saved[:2] == (record.id, 1)
        assert json.loads(saved[2]) == dict(record.to_wire(), project="default")
        row = staged.db.execute("SELECT document FROM finished_history").fetchone()
        assert json.loads(row[0]) == dict(finished.to_wire(), assetDigests=[digest])
        row = staged.db.execute("SELECT document FROM retention_clients").fetchone()
        client = json.loads(row[0])
        assert (
            client["generation"],
            client["revision"],
            client["lastActivity"],
            client["departed"],
            client["pendingRequests"],
            client["activeStreams"],
        ) == ("generation", 9, 123.5, True, 2, 1)
        assert staged.db.execute("SELECT * FROM retention_quarantine").fetchall() == [
            ("engine-one", "client", digest, 47)
        ]
        assert staged.db.execute("SELECT document FROM permission_records").fetchone() == (
            '{"unknown-future-category":false}',
        )
        assert (backup / "account/model-settings.json").read_bytes() == settings
    assert {p.name: p.read_bytes() for p in root.glob("*.sqlite")} == sources
    # Ordinary helper startup still creates only its established project schema.
    with LibraryClient(root) as helper:
        capabilities = helper.hello["capabilities"]
        assert isinstance(capabilities, list) and "projects" in capabilities
    with closing(sqlite3.connect(root / "records.sqlite")) as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "projects" in tables and "saved_records" not in tables
    assert not (root / "library-format.json").exists()


@pytest.mark.parametrize("damage", (None, "missing", "changed", "manifest"))
def test_killed_staging_retries_the_verified_wal_snapshot(
    tmp_path: Path, damage: str | None
) -> None:
    root, account = tmp_path / "library", tmp_path / "account"
    root.mkdir()
    # Keep the connection open: the committed row has not been checkpointed out of WAL.
    with closing(sqlite3.connect(root / "library.sqlite")) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("CREATE TABLE committed (value TEXT)")
        db.execute("INSERT INTO committed VALUES ('original')")
        db.commit()
        script = """
import sys
from pathlib import Path
from dinkster_library._migration import record_backup
with record_backup(Path(sys.argv[1]), Path(sys.argv[2])):
    print('snapshot committed', flush=True)
    sys.stdin.read()
"""
        child = subprocess.Popen(
            [sys.executable, "-c", script, str(root), str(account)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            assert child.stdout is not None
            assert child.stdout.readline().strip() == "snapshot committed"
        finally:
            child.kill()
            child.communicate(timeout=10)
        journal = (root / "record-migration.json").read_bytes()
        backup = root / "backups" / json.loads(journal)["backup"]
        with closing(sqlite3.connect(backup / "library/library.sqlite")) as snapshot:
            assert snapshot.execute("SELECT value FROM committed").fetchall() == [("original",)]
        db.execute("UPDATE committed SET value='partially replaced live source'")
        db.commit()
        if damage == "missing":
            (backup / "library/library.sqlite").unlink()
        elif damage == "changed":
            with (backup / "library/library.sqlite").open("ab") as stream:
                stream.write(b"corrupt")
        elif damage == "manifest":
            manifest = backup / "manifest.json"
            document = json.loads(manifest.read_text())
            document["files"] = {}
            manifest.write_text(json.dumps(document))
        from dinkster_library._migration import record_backup

        if damage:
            with pytest.raises(LibraryError):
                with record_backup(root, account):
                    pytest.fail("damaged snapshot accepted")
        else:
            with record_backup(root, account) as recovered:
                assert recovered == backup
                with closing(sqlite3.connect(recovered / "library/library.sqlite")) as snapshot:
                    assert snapshot.execute("SELECT value FROM committed").fetchall() == [
                        ("original",)
                    ]
        assert (root / "record-migration.json").read_bytes() == journal


def test_staging_refuses_attached_helper_and_competing_direct_engine(tmp_path: Path) -> None:
    root, account = tmp_path / "library", tmp_path / "account"
    with LibraryClient(root):
        with pytest.raises(LibraryError, match="Stop library helpers"):
            with stage_records(root, account, "engine"):
                pytest.fail("attached helper accepted")
    with stage_records(root, account, "engine"):
        script = """
import sys
from pathlib import Path
from dinkster_store.library_lock import engine_lock, LibraryInUse
try:
    with engine_lock(Path(sys.argv[1]), 'competing engine'):
        raise SystemExit('migration did not exclude direct engine')
except LibraryInUse:
    raise SystemExit(42)
"""
        assert (
            subprocess.run([sys.executable, "-c", script, str(root)], check=False).returncode == 42
        )
