"""Folder scans run outside the request loop; only helpers open the digest cache."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PureWindowsPath
from typing import cast

from filelock import FileLock

from ._index import index_file
from .client import LibraryError

_PARTIAL = {
    ".part",
    ".partial",
    ".crdownload",
    ".download",
    ".tmp",
    ".aria2",
    ".!qb",
    ".opdownload",
}
_DEFAULT_IGNORE = [
    "put_*_here",
    "Thumbs.db",
    "desktop.ini",
    "*.tmp",
    "*.part",
    "*.crdownload",
    "*.aria2",
]


def _pattern(value: str) -> re.Pattern[str]:
    parts: list[str] = []
    offset = 0
    while offset < len(value):
        char = value[offset]
        if value[offset : offset + 2] == "**":
            parts.append(".*")
            offset += 2
            continue
        parts.append("[^/]*" if char == "*" else "[^/]" if char == "?" else re.escape(char))
        offset += 1
    return re.compile("".join(parts) + r"\Z")


def _rows(path: Path) -> dict[str, dict[str, object]]:
    try:
        value: object = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(value, dict):
        return {}
    return {
        name: cast("dict[str, object]", row)
        for name, row in cast("dict[str, object]", value).items()
        if isinstance(row, dict)
    }


def _publish(path: Path, rows: dict[str, dict[str, object]]) -> None:
    encoded = json.dumps(rows, indent=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    try:
        temporary.write_text(encoded, "utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class Scans:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, object]] = {}
        self._outputs: dict[Path, str] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="library-scan")

    def start(self, arguments: dict[str, object]) -> dict[str, object]:
        root_value, output_value = arguments.get("root"), arguments.get("output")
        if (
            not isinstance(root_value, str)
            or not Path(root_value).is_absolute()
            or not isinstance(output_value, str)
            or not Path(output_value).is_absolute()
        ):
            raise LibraryError("invalid", "scan root and output must be absolute paths")
        root, output = Path(root_value).resolve(strict=True), Path(output_value).resolve()
        if not root.is_dir():
            raise LibraryError("invalid", "scan root must be a directory")
        relative = arguments.get("relative")
        if relative is not None and (
            not isinstance(relative, str)
            or "\\" in relative
            or PureWindowsPath(relative).drive
            or any(part in ("", ".", "..") for part in relative.split("/"))
        ):
            raise LibraryError("invalid", "selected file must be a safe relative path")
        legacy = arguments.get("legacy", [])
        ignore = arguments.get("ignore", _DEFAULT_IGNORE)
        if not isinstance(legacy, list) or any(
            not isinstance(item, str) or not Path(item).is_absolute()
            for item in cast("list[object]", legacy)
        ):
            raise LibraryError("invalid", "legacy indexes must be absolute paths")
        if not isinstance(ignore, list) or any(
            not isinstance(item, str) for item in cast("list[object]", ignore)
        ):
            raise LibraryError("invalid", "ignore must be a list of patterns")
        with self._lock:
            previous = self._outputs.get(output)
            if previous is not None:
                if self._jobs[previous]["state"] in ("queued", "scanning"):
                    raise LibraryError("busy", "a scan is already writing this output")
                del self._jobs[previous]
            identifier = uuid.uuid4().hex
            state: dict[str, object] = {
                "id": identifier,
                "state": "queued",
                "filesDone": 0,
                "filesTotal": 0,
                "bytesDone": 0,
                "bytesTotal": 0,
                "hashed": 0,
                "elapsed": 0.0,
            }
            self._jobs[identifier] = state
            self._outputs[output] = identifier
            self._executor.submit(
                self._run,
                state,
                root,
                output,
                [Path(item) for item in cast("list[str]", legacy)],
                cast("list[str]", ignore),
                relative,
            )
            return dict(state)

    def progress(self, identifier: object) -> dict[str, object]:
        with self._lock:
            if not isinstance(identifier, str) or identifier not in self._jobs:
                raise LibraryError("invalid", "unknown scan")
            return dict(self._jobs[identifier])

    def _run(
        self,
        state: dict[str, object],
        root: Path,
        output: Path,
        legacy: list[Path],
        ignore: list[str],
        selected: str | None,
    ) -> None:
        started = time.monotonic()
        try:
            home = Path(os.environ.get("DINKSTER_HOME", Path.home() / ".dinkster")).expanduser()
            locks = home / "scan-locks"
            locks.mkdir(parents=True, exist_ok=True)
            key = hashlib.sha256(os.path.normcase(str(root)).encode("utf-8")).hexdigest()
            with FileLock(locks / f"{key}.lock"):
                with self._lock:
                    state["state"] = "scanning"
                rules = [("/" in item, _pattern(item)) for item in ignore]
                candidates: list[tuple[Path, str, os.stat_result]] = []
                paths = sorted(root.rglob("*")) if selected is None else [root / selected]
                for path in paths:
                    relative = path.relative_to(root).as_posix()
                    if (
                        any(part.startswith(".") for part in Path(relative).parts)
                        or path == output
                        or path.suffix.lower() in _PARTIAL
                        or any(
                            rule.match(relative if anchored else path.name)
                            for anchored, rule in rules
                        )
                        or not path.is_file()
                    ):
                        continue
                    try:
                        if not path.resolve(strict=True).is_relative_to(root):
                            continue
                        candidates.append((path, relative, path.stat()))
                    except FileNotFoundError:
                        continue
                seeds = [
                    _rows(path) for path in [output, *legacy, root / ".dinkster-asset-index.json"]
                ]
                writes: dict[str, dict[str, object]] = {}
                try:
                    lines = (root / ".dinkster-asset-writes.jsonl").read_text("utf-8").splitlines()
                except OSError:
                    lines = []
                for line in lines:
                    try:
                        entry: object = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(entry, dict):
                        row = cast("dict[str, object]", entry)
                        if isinstance(row.get("path"), str):
                            writes[cast(str, row["path"])] = row
                seeds.insert(0, writes)
                rows: dict[str, dict[str, object]] = {}
                for seed in reversed(seeds):
                    rows.update(
                        {
                            name: {**row, "missing": True}
                            if selected is None or name == selected
                            else dict(row)
                            for name, row in seed.items()
                        }
                    )
                with self._lock:
                    state.update(
                        filesTotal=len(candidates),
                        bytesTotal=sum(metadata.st_size for _, _, metadata in candidates),
                    )
                if candidates:
                    time.sleep(0.1)
                last_publish = started
                files_done = bytes_done = hashed_count = 0
                for path, relative, metadata in candidates:
                    try:
                        current = path.stat()
                        if (metadata.st_size, metadata.st_mtime_ns) != (
                            current.st_size,
                            current.st_mtime_ns,
                        ):
                            continue
                        row, hashed = index_file(
                            str(path),
                            seeds=tuple(seed[relative] for seed in seeds if relative in seed),
                        )
                    except FileNotFoundError:
                        continue
                    previous = rows.get(relative, {})
                    if (
                        previous.get("digest") == row["digest"]
                        and previous.get("verification") == row.get("verification")
                        and previous.get("size") == row["size"]
                        and previous.get("mtimeNs") == row["mtimeNs"]
                        and type(previous.get("publishedAtNs")) is int
                    ):
                        row["publishedAtNs"] = previous["publishedAtNs"]
                    row.pop("path", None)
                    versions = previous.get("missingVersions", [])
                    retained = (
                        [
                            dict(cast("dict[str, object]", item))
                            for item in cast("list[object]", versions)
                            if isinstance(item, dict)
                            and cast("dict[str, object]", item).get("digest") != row["digest"]
                        ]
                        if isinstance(versions, list)
                        else []
                    )
                    if previous.get("digest") not in (None, row["digest"]):
                        retained.append(
                            {
                                name: value
                                for name, value in previous.items()
                                if name != "missingVersions"
                            }
                        )
                    if retained:
                        row["missingVersions"] = retained
                    row.pop("missing", None)
                    rows[relative] = row
                    files_done += 1
                    bytes_done += cast(int, row["size"])
                    hashed_count += int(hashed)
                    if time.monotonic() - last_publish >= 1:
                        _publish(output, rows)
                        last_publish = time.monotonic()
                    with self._lock:
                        state.update(
                            filesDone=files_done,
                            bytesDone=bytes_done,
                            hashed=hashed_count,
                            elapsed=time.monotonic() - started,
                        )
                _publish(output, rows)
                with self._lock:
                    state.update(state="complete", elapsed=time.monotonic() - started)
        except Exception as error:
            with self._lock:
                state.update(state="failed", error=str(error), elapsed=time.monotonic() - started)
