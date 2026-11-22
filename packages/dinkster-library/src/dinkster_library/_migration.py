"""Restartable source snapshots for the helper's complete record conversion."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import uuid
from collections.abc import Generator
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from ._records import Records

from filelock import FileLock, Timeout

from .client import LibraryError

_WORKING_STORES = ("execution.sqlite", "sessions.sqlite", "worker-state", "value-store")
_LIBRARY_STORES = (
    *_WORKING_STORES,
    "records.sqlite",
    "library.sqlite",
    "history.sqlite",
    "principals.sqlite",
    "project-folders.json",
    "mounts.toml",
    "provenance.json",
    "resolver-indexes.json",
    "public-acquisition-receipts.json",
    "settings.json",
    "pack-settings",
)
_ACCOUNT_STORES = ("model-settings.json", "model-file-selections.json", "model-settings.sqlite")


def _sync_directory(path: Path) -> None:
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _regular(path: Path) -> None:
    if any(part.is_symlink() or part.is_junction() for part in (path, *path.parents)):
        raise LibraryError("migration", "Record migration refuses linked paths")
    if not path.is_file() or path.stat().st_nlink != 1:
        raise LibraryError("migration", "Record migration requires regular unshared files")


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _publish(path: Path, value: dict[str, object]) -> None:
    pending = path.with_name(path.name + ".pending")
    if os.path.lexists(pending):
        _regular(pending)
        pending.unlink()
    with pending.open("xb") as stream:
        stream.write(json.dumps(value, ensure_ascii=True, allow_nan=False).encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)
    _sync_directory(path.parent)


def _verify(backup: Path, root: Path, account: Path) -> None:
    manifest_path = backup / "manifest.json"
    _regular(manifest_path)
    raw: object = json.loads(manifest_path.read_text("utf-8"))
    if not isinstance(raw, dict):
        raise LibraryError("migration", "Invalid record migration backup")
    document = cast("dict[str, object]", raw)
    if (
        document.get("format") != "dinkster.record-backup/1"
        or document.get("library") != str(root)
        or document.get("account") != str(account)
        or not isinstance(document.get("files"), dict)
    ):
        raise LibraryError("migration", "Invalid record migration backup")
    files = cast("dict[str, object]", document["files"])
    present = {
        path.relative_to(backup).as_posix()
        for path in backup.rglob("*")
        if not path.is_dir() or path.is_symlink() or path.is_junction()
    }
    if present != {"manifest.json", *files}:
        raise LibraryError("migration", "Record migration backup contains unverified entries")
    for name, expected in files.items():
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or len(relative.parts) < 2
            or relative.parts[0] not in ("library", "account")
        ):
            raise LibraryError("migration", "Invalid record migration backup path")
        path = backup / relative
        _regular(path)
        if _digest(path) != expected:
            raise LibraryError("migration", "Record migration backup verification failed")


@contextmanager
def record_backup(root: Path, account: Path) -> Generator[Path]:
    """Hold exclusive migration admission and yield the immutable source snapshot.

    Call before starting an engine or attaching a helper, and hold this context
    through conversion and publication. A retry always uses the first complete
    snapshot, even when live stores have already been partially replaced.
    """
    root, account = root.absolute(), account.absolute()
    for directory in (root, account):
        if ".." in directory.parts or any(
            path.is_symlink() or path.is_junction() for path in (directory, *directory.parents)
        ):
            raise LibraryError("migration", "Record migration requires unlinked absolute roots")
    root.mkdir(parents=True, exist_ok=True)
    account.mkdir(parents=True, exist_ok=True)
    for lock_path in (
        root / ".engine.lock",
        account / "model-settings.lock",
        root / "records-format.lock",
    ):
        if os.path.lexists(lock_path):
            _regular(lock_path)
    # FileLock uses the same first-byte Windows lock or Unix flock as the tools'
    # direct-engine gate. Keep it until the stop-file and converted stores exist.
    with (
        FileLock(root / ".engine.lock", timeout=0, preserve_lock_file=True),
        FileLock(account / "model-settings.lock", timeout=0, preserve_lock_file=True),
        FileLock(root / "records-format.lock", timeout=0),
    ):
        for path in root.glob("helper-*.lock"):
            _regular(path)
            lock = FileLock(path)
            try:
                lock.acquire(timeout=0)
            except Timeout as error:
                raise LibraryError("attached", "Stop library helpers before migrating") from error
            else:
                lock.release()
        account_gate = account / "model-settings-migration.json"
        if os.path.lexists(account_gate):
            _regular(account_gate)
            if json.loads(account_gate.read_text("utf-8")) != {"library": str(root)}:
                raise LibraryError(
                    "migration", "Finish the other library's account migration first"
                )
        journal = root / "record-migration.json"
        if os.path.lexists(journal):
            _regular(journal)
            raw: object = json.loads(journal.read_text("utf-8"))
            if not isinstance(raw, dict):
                raise LibraryError("migration", "Invalid record migration journal")
            document = cast("dict[str, object]", raw)
            name = document.get("backup")
            if (
                not isinstance(name, str)
                or not name.startswith("before-migration-")
                or Path(name).name != name
                or document.get("format") != "dinkster.record-migration/1"
            ):
                raise LibraryError("migration", "Invalid record migration journal")
            backup = root / "backups" / name
            manifest_digest = document.get("manifestDigest")
        else:
            backups = root / "backups"
            if any(path.is_symlink() or path.is_junction() for path in (backups,)):
                raise LibraryError("migration", "Record backups must not be linked")
            backups.mkdir(exist_ok=True)
            name = (
                "before-migration-"
                + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-")
                + uuid.uuid4().hex
            )
            backup = backups / name
            backup.mkdir(mode=0o700)
            files: dict[str, object] = {}
            for namespace, directory, stores in (
                ("library", root, _LIBRARY_STORES),
                ("account", account, _ACCOUNT_STORES),
            ):
                for store in stores:
                    source = directory / store
                    if not os.path.lexists(source):
                        continue
                    if source.is_symlink() or source.is_junction():
                        raise LibraryError("migration", "Record migration refuses linked stores")
                    sources = sorted(source.rglob("*")) if source.is_dir() else [source]
                    for original in sources:
                        if (
                            original.is_dir()
                            and not original.is_symlink()
                            and not original.is_junction()
                        ):
                            continue
                        _regular(original)
                        sidecar = False
                        for suffix in ("-wal", "-shm", "-journal"):
                            if original.name.endswith(suffix):
                                database = original.with_name(original.name[: -len(suffix)])
                                if database.suffix == ".sqlite":
                                    _regular(database)
                                    sidecar = True
                        if sidecar:
                            # The logical SQLite backup already contains committed WAL data.
                            continue
                        relative = Path(namespace) / original.relative_to(directory)
                        target = backup / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if original.suffix == ".sqlite":
                            for suffix in ("-wal", "-shm", "-journal"):
                                sidecar = original.with_name(original.name + suffix)
                                if os.path.lexists(sidecar):
                                    _regular(sidecar)
                            # Copy the logical database, including committed WAL pages.
                            # Copying only the main file silently loses recent records.
                            with closing(
                                sqlite3.connect(original.as_uri() + "?mode=ro", uri=True)
                            ) as src:
                                with closing(sqlite3.connect(target)) as dst:
                                    src.backup(dst)
                                    if dst.execute("PRAGMA integrity_check").fetchall() != [
                                        ("ok",)
                                    ]:
                                        raise LibraryError(
                                            "migration", "Invalid record database backup"
                                        )
                        else:
                            with original.open("rb") as src, target.open("xb") as dst:
                                while chunk := src.read(1024 * 1024):
                                    dst.write(chunk)
                        with target.open("rb") as stream:
                            os.fsync(stream.fileno())
                        files[relative.as_posix()] = _digest(target)
            # Child directory entries must reach disk before publishing the journal
            # that lets a retry trust this snapshot instead of the live stores.
            for directory in sorted(
                (path for path in backup.rglob("*") if path.is_dir()),
                key=lambda path: len(path.parts),
                reverse=True,
            ):
                _sync_directory(directory)
            _publish(
                backup / "manifest.json",
                {
                    "format": "dinkster.record-backup/1",
                    "engineWorkingState": 1,
                    "library": str(root),
                    "account": str(account),
                    "files": files,
                },
            )
            _sync_directory(backups)
            # A process killed while staging this pointer left no committed
            # migration. Its exclusive owner can discard only that staging file.
            pending = journal.with_name(journal.name + ".pending")
            if os.path.lexists(pending):
                _regular(pending)
                pending.unlink()
            manifest_digest = _digest(backup / "manifest.json")
            _publish(
                journal,
                {
                    "format": "dinkster.record-migration/1",
                    "backup": name,
                    "manifestDigest": manifest_digest,
                },
            )
        _regular(backup / "manifest.json")
        if _digest(backup / "manifest.json") != manifest_digest:
            raise LibraryError("migration", "Record migration manifest verification failed")
        _verify(backup, root, account)
        yield backup


def _stage_working_state(backup: Path, destination: Path) -> None:
    """Copy frozen engine work without placing finished records in its database."""
    destination.mkdir()
    manifest: dict[str, object] = json.loads((backup / "manifest.json").read_text("utf-8"))
    if type(manifest.get("engineWorkingState")) is not int or manifest["engineWorkingState"] != 1:
        raise LibraryError("migration", "Record backup does not include engine working state")
    for name in cast("dict[str, object]", manifest["files"]):
        relative = Path(name)
        if relative.parts[0] != "library" or relative.parts[1] not in _WORKING_STORES:
            continue
        source = backup / relative
        target = destination / Path(*relative.parts[1:])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        with target.open("rb") as stream:
            os.fsync(stream.fileno())
    history = backup / "library/history.sqlite"
    if "library/history.sqlite" in cast("dict[str, object]", manifest["files"]):
        with closing(
            sqlite3.connect(history.as_uri() + "?mode=ro&immutable=1", uri=True)
        ) as source:
            tables = {
                row[0]
                for row in source.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
            }
            if tables - {"history", "accepted"}:
                raise LibraryError("migration", "Unknown direct-engine history tables")
            if "accepted" in tables:
                target_path = destination / "accepted.sqlite"
                with closing(sqlite3.connect(target_path)) as target:
                    source.backup(target)
                    with target:
                        target.execute("DROP TABLE history")
                with target_path.open("rb") as stream:
                    os.fsync(stream.fileno())
    for directory in sorted(
        (path for path in destination.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        _sync_directory(directory)
    _sync_directory(destination)


@contextmanager
def stage_records(
    root: Path, account: Path, engine: str, *, environment: str | None = None
) -> Generator[tuple[Records, Path]]:
    """Rebuild an unpublished conversion from its durable source snapshot.

    Live stores remain untouched. Publication belongs after conversion of all
    file-backed preferences and sources, not after an individual database.
    """
    with record_backup(root, account) as backup:
        with _converted_records(root, backup, engine, environment) as records:
            yield records, backup


@contextmanager
def _converted_records(
    root: Path, backup: Path, engine: str, environment: str | None
) -> Generator[Records]:
    from ._legacy_files import convert_preferences, convert_sources
    from ._legacy_records import convert_databases
    from ._records import Records

    with TemporaryDirectory(prefix=".record-conversion-", dir=root) as directory:
        stage = Path(directory)
        previous = backup / "library" / "records.sqlite"
        if previous.exists():
            with (
                closing(
                    sqlite3.connect(previous.as_uri() + "?mode=ro&immutable=1", uri=True)
                ) as source,
                closing(sqlite3.connect(stage / "records.sqlite")) as target,
            ):
                source.backup(target)
        records = Records(stage)
        try:
            convert_databases(backup, records.db, engine)
            convert_preferences(backup, records, environment)
            convert_sources(backup, records)
            _stage_working_state(backup, stage / "engine-state")
            yield records
        finally:
            records.close()


def _replace_database(source: Path, target: Path) -> None:
    """Publish one checkpointed database while its exclusive admission gates are held."""
    pending = target.with_name(target.name + ".pending")
    for path in (
        pending,
        *(target.with_name(target.name + suffix) for suffix in ("-wal", "-shm", "-journal")),
    ):
        if os.path.lexists(path):
            _regular(path)
            path.unlink()
    with (
        closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as src,
        closing(sqlite3.connect(pending)) as dst,
    ):
        src.backup(dst)
        if dst.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise LibraryError("migration", "Invalid converted record database")
        dst.execute("PRAGMA journal_mode=DELETE")
    with pending.open("rb") as stream:
        os.fsync(stream.fileno())
    if os.path.lexists(target):
        _regular(target)
    os.replace(pending, target)
    _sync_directory(target.parent)


def _retire(path: Path) -> None:
    if not os.path.lexists(path):
        return
    if path.is_symlink() or path.is_junction():
        raise LibraryError("migration", "Record migration refuses linked stores")
    if path.is_dir():
        for child in path.iterdir():
            _retire(child)
        path.rmdir()
    else:
        _regular(path)
        path.unlink()
    _sync_directory(path.parent)


def publish_records(
    root: Path,
    account: Path,
    engine: str,
    state: Path,
    *,
    environment: str | None = None,
) -> None:
    """Publish the complete conversion once, or resume its original frozen snapshot.

    Call from the selected tools helper before any engine or helper attaches.
    The destination state is private to the first engine; a retry cannot select
    another destination or replay the backup after completion.
    """
    if re.fullmatch(r"[0-9a-f]{32}", engine) is None:
        raise LibraryError("invalid", "engine identity must be 32 lowercase hex digits")
    root, account, state = root.absolute(), account.absolute(), state.absolute()
    if any(path.is_symlink() or path.is_junction() for path in (state, *state.parents)):
        raise LibraryError("migration", "Engine state must not contain linked directories")
    if ".." in state.parts or state.is_relative_to(root) or root.is_relative_to(state):
        raise LibraryError("migration", "Engine state must not overlap the library")
    with record_backup(root, account) as backup:
        journal = root / "record-migration.json"
        document: dict[str, object] = json.loads(journal.read_text("utf-8"))
        account_gate = account / "model-settings-migration.json"
        if document.get("complete") is True:
            # Completion precedes removal of the account gate. A killed cleanup
            # must release it without overwriting any subsequent library writes.
            if account_gate.exists():
                account_gate.unlink()
                _sync_directory(account)
            return
        binding: dict[str, object] = {
            "engine": engine,
            "state": str(state),
            "environment": environment,
        }
        previous = document.get("publication")
        if previous is not None and previous != binding:
            raise LibraryError("migration", "Resume migration with its original engine state")
        state.mkdir(parents=True, exist_ok=True)
        with FileLock(state / ".state.lock", timeout=0, preserve_lock_file=True):
            if previous is None:
                if any(path.name != ".state.lock" for path in state.iterdir()):
                    raise LibraryError("migration", "Migration requires unused engine state")
                document["publication"] = binding
                _publish(journal, document)
            with _converted_records(root, backup, engine, environment) as records:
                stage = records.root
                # The stop-file goes first. A killed publication can never leave
                # a direct engine free to create or modify a partially moved library.
                stop = stage / "stop.sqlite"
                with closing(sqlite3.connect(stop)) as db, db:
                    db.execute("CREATE TABLE alembic_version (version_num TEXT NOT NULL)")
                    db.execute(
                        "INSERT INTO alembic_version VALUES ('library-code-owns-this-library')"
                    )
                _replace_database(stop, root / "library.sqlite")
                _replace_database(stage / "records.sqlite", root / "records.sqlite")
                for source in (stage / "engine-state").rglob("*"):
                    if source.is_dir():
                        continue
                    target = state / source.relative_to(stage / "engine-state")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if source.suffix == ".sqlite":
                        _replace_database(source, target)
                    else:
                        pending = target.with_name(target.name + ".pending")
                        _retire(pending)
                        shutil.copyfile(source, pending)
                        with pending.open("rb") as stream:
                            os.fsync(stream.fileno())
                        os.replace(pending, target)
                        _sync_directory(target.parent)
                _publish(
                    state / "owner.json",
                    {"format": "dinkster.engine-state/1", "library": str(root)},
                )
                for directory in sorted(
                    (path for path in state.rglob("*") if path.is_dir()),
                    key=lambda path: len(path.parts),
                    reverse=True,
                ):
                    _sync_directory(directory)
                _sync_directory(state)
                _sync_directory(state.parent)
                model_database = stage / "account/model-settings.sqlite"
                # An already converted account remains live across other libraries.
                # Only a conversion of legacy account files may replace its database.
                if any(
                    (backup / "account" / name).exists()
                    for name in ("model-settings.json", "model-file-selections.json")
                ):
                    _publish(account_gate, {"library": str(root)})
                    _replace_database(model_database, account / "model-settings.sqlite")
                    for name in ("model-settings.json", "model-file-selections.json"):
                        _retire(account / name)
                for name in _LIBRARY_STORES:
                    if name not in ("records.sqlite", "library.sqlite"):
                        _retire(root / name)
                        if name.endswith(".sqlite"):
                            for suffix in ("-wal", "-shm", "-journal"):
                                _retire(root / (name + suffix))
                _publish(
                    root / "library-format.json",
                    {
                        "format": "dinkster.library-format/1",
                        "requiredCapabilities": ["library-helper/1"],
                        "libraryApi": 1,
                    },
                )
                document["complete"] = True
                _publish(journal, document)
                if account_gate.exists():
                    account_gate.unlink()
                    _sync_directory(account)
            complete_backups = sorted(
                path
                for path in (root / "backups").glob("before-migration-*")
                if (path / "manifest.json").is_file()
            )
            for obsolete in complete_backups[:-3]:
                if obsolete != backup:
                    _retire(obsolete)
