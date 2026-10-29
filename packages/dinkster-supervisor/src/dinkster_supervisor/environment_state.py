"""Durable project pointers and immutable environment generation records."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


class EnvironmentError(ValueError):
    """An environment cannot be safely read or changed."""


@contextmanager
def writer_lock(root: Path) -> Generator[None]:
    """Hold a persistent inode's OS lock; never unlink another writer's lock."""
    with (root / ".lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise EnvironmentError(f"another environment writer holds {root}") from exc
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise EnvironmentError(f"another environment writer holds {root}") from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def content_digest(path: Path) -> str:
    with path.open("rb") as handle:
        return "sha256:" + hashlib.file_digest(handle, "sha256").hexdigest()


def digest(value: object) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise EnvironmentError("expected a sha256: content digest")
    return value


def relative_file(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise EnvironmentError("expected a portable relative file path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or any(ord(character) < 32 for character in value)
    ):
        raise EnvironmentError("expected a portable relative file path")
    return value


def read_document(path: Path) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EnvironmentError(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise EnvironmentError(f"{path.name}: expected a JSON object")
    return cast("dict[str, object]", value)


def atomic_write(path: Path, text: str) -> None:
    """Flush the new contents before publishing a pointer or journal stage."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_document(path: Path, value: dict[str, object]) -> None:
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n")


def durable_unlink(path: Path) -> None:
    """Flush removal even when retrying an interrupted, unsynced unlink."""
    path.unlink(missing_ok=True)
    if os.name != "nt":
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def generation_number(value: object) -> int:
    if type(value) is not int or value < 1:
        raise EnvironmentError("generation must be a positive integer")
    return value


@dataclass(frozen=True)
class ArtifactLocation:
    kind: str
    path: str = ""
    repository: str = ""
    tag: str = ""
    asset: str = ""

    @classmethod
    def read(cls, value: object, expected_path: str) -> ArtifactLocation:
        if not isinstance(value, dict):
            raise EnvironmentError("artifact location must be an object")
        row = cast("dict[str, object]", value)
        if row.get("type") == "mirror":
            path = relative_file(row.get("path"))
            if path != expected_path:
                raise EnvironmentError("mirror manifest path must name its exact content digest")
            return cls("mirror", path=path)
        if row.get("type") == "github":
            repository = row.get("repository")
            if (
                not isinstance(repository, str)
                or re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", repository) is None
            ):
                raise EnvironmentError("GitHub location must name owner/repository")
            tag = relative_file(row.get("tag"))
            if tag in {"latest", "stable", "main"}:
                raise EnvironmentError("GitHub manifest locations must not use alias tags")
            asset = relative_file(row.get("asset"))
            if asset != expected_path.rsplit("/", 1)[-1]:
                raise EnvironmentError("GitHub location must name the expected manifest asset")
            return cls("github", repository=repository, tag=tag, asset=asset)
        raise EnvironmentError("unsupported artifact location type")


def _locations(value: object, expected_path: str) -> tuple[ArtifactLocation, ...]:
    if not isinstance(value, list) or not value:
        raise EnvironmentError("artifact locations must be a nonempty ordered array")
    return tuple(ArtifactLocation.read(row, expected_path) for row in cast("list[object]", value))


@dataclass(frozen=True)
class Channel:
    engine_version: str
    engine_release: str
    bases: tuple[tuple[str, str], ...]
    minimum_desktop_version: str
    engine_locations: tuple[ArtifactLocation, ...]
    base_locations: tuple[tuple[str, tuple[ArtifactLocation, ...]], ...]

    @classmethod
    def read(cls, path: Path) -> Channel:
        return cls.parse(read_document(path))

    @classmethod
    def parse(cls, value: dict[str, object]) -> Channel:
        if value.get("format") != "dinkster.channel/1":
            raise EnvironmentError("unsupported channel format")
        engine, bases = value.get("engine"), value.get("base")
        minimum = value.get("minimumDesktopVersion")
        if not isinstance(engine, dict) or not isinstance(bases, dict) or not bases:
            raise EnvironmentError("channel must name an engine and at least one base")
        engine = cast("dict[str, object]", engine)
        bases = cast("dict[str, object]", bases)
        version = engine.get("version")
        if not isinstance(version, str) or not version:
            raise EnvironmentError("channel engine version must be a nonempty string")
        if not isinstance(minimum, str) or re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", minimum) is None:
            raise EnvironmentError("minimumDesktopVersion must be X.Y.Z")
        if value.get("signature") is not None:
            raise EnvironmentError("signed channel verification is not supported")
        release = digest(engine.get("release"))
        base_ids: list[tuple[str, str]] = []
        base_locations: list[tuple[str, tuple[ArtifactLocation, ...]]] = []
        for name, entry in sorted(bases.items()):
            if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) is None:
                raise EnvironmentError("channel variant must be a portable variant name")
            if not isinstance(entry, dict):
                raise EnvironmentError("channel base must name id and locations")
            row = cast("dict[str, object]", entry)
            sha = digest(row.get("id"))
            base_ids.append((name, sha))
            base_locations.append(
                (name, _locations(row.get("locations"), f"base/{name}/{sha[7:]}/manifest.json"))
            )
        return cls(
            version,
            release,
            tuple(base_ids),
            minimum,
            _locations(engine.get("locations"), f"engine/{release[7:]}/release-manifest.json"),
            tuple(base_locations),
        )

    def base_for(self, variant: str) -> str:
        for name, sha in self.bases:
            if name == variant:
                return sha
        raise EnvironmentError(f"channel does not provide variant {variant}; choose explicitly")


@dataclass(frozen=True)
class Generation:
    number: int
    base: str
    engine_version: str
    engine_release: str
    constraints: str
    variant: str
    worker_protocol: int
    files: tuple[tuple[str, str], ...]
    in_process_packs: tuple[str, ...] = ()

    @classmethod
    def read(cls, path: Path) -> Generation:
        value = read_document(path)
        if value.get("format") != "dinkster.generation/1":
            raise EnvironmentError("unsupported generation format")
        engine = value.get("engine")
        if not isinstance(engine, dict):
            raise EnvironmentError("generation engine must name version and release")
        engine = cast("dict[str, object]", engine)
        version, variant = engine.get("version"), value.get("variant")
        if not isinstance(version, str) or not version:
            raise EnvironmentError("generation engine version must be a nonempty string")
        if (
            not isinstance(variant, str)
            or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", variant) is None
        ):
            raise EnvironmentError("generation variant must be a portable variant name")
        protocol = value.get("workerProtocol")
        if type(protocol) is not int or protocol < 1:
            raise EnvironmentError("workerProtocol must be a positive integer")
        files = value.get("files")
        if not isinstance(files, dict) or not files:
            raise EnvironmentError("generation must record its file hashes")
        hashes = cast("dict[str, object]", files)
        constraints = digest(value.get("constraints"))
        if hashes.get("constraints.txt") != constraints:
            raise EnvironmentError("constraints digest must match its recorded file hash")
        hosting = value.get("inProcessPacks", [])
        if not isinstance(hosting, list):
            raise EnvironmentError("inProcessPacks must be an array of pack ids")
        entries = cast("list[object]", hosting)
        if any(
            not isinstance(pack, str) or re.fullmatch(r"[a-z0-9][a-z0-9_-]*", pack) is None
            for pack in entries
        ):
            raise EnvironmentError("inProcessPacks must be an array of pack ids")
        packs = cast("list[str]", entries)
        if len(set(packs)) != len(packs):
            raise EnvironmentError("inProcessPacks must not repeat a pack")
        return cls(
            generation_number(value.get("generation")),
            digest(value.get("base")),
            version,
            digest(engine.get("release")),
            constraints,
            variant,
            protocol,
            tuple(sorted((relative_file(name), digest(sha)) for name, sha in hashes.items())),
            tuple(packs),
        )

    def write(self, path: Path) -> None:
        write_document(
            path,
            {
                "format": "dinkster.generation/1",
                "generation": self.number,
                "base": self.base,
                "engine": {"version": self.engine_version, "release": self.engine_release},
                "constraints": self.constraints,
                "variant": self.variant,
                "workerProtocol": self.worker_protocol,
                "files": dict(self.files),
                "inProcessPacks": list(self.in_process_packs),
            },
        )

    def drift(self, root: Path) -> tuple[str, ...]:
        """Compare the complete owned tree, not just installed package versions."""
        expected = dict(self.files)
        changes: list[str] = []
        for name, sha in expected.items():
            path = root / name
            if not path.is_file():
                changes.append(f"missing: {name}")
            elif path.is_symlink() or content_digest(path) != sha:
                changes.append(f"modified: {name}")
        for path in sorted(root.rglob("*")):
            name = path.relative_to(root).as_posix()
            # Pack venvs have their own locks and doctor; bytecode is runtime output.
            if name.startswith("packs/") or "__pycache__" in path.relative_to(root).parts:
                continue
            if path.is_file() and name != "generation.json" and name not in expected:
                changes.append(f"unexpected: {name}")
        return tuple(changes)


def read_current(root: Path) -> int | None:
    path = root / "current"
    if not path.exists():
        return None
    text = path.read_text(encoding="ascii")
    if re.fullmatch(r"[1-9][0-9]*\n", text) is None:
        raise EnvironmentError("current must contain an ASCII generation number and newline")
    return int(text)


def set_current(root: Path, number: int) -> None:
    generation_number(number)
    path = root / "generations" / str(number) / "generation.json"
    if Generation.read(path).number != number:
        raise EnvironmentError("generation directory and recorded number disagree")
    atomic_write(root / "current", f"{number}\n")


def interpreter(generation: Path) -> Path:
    return generation / "python" / ("python.exe" if os.name == "nt" else "bin/python3")


@dataclass(frozen=True)
class Journal:
    operation: str
    previous: int | None
    target: int
    stage: str
    error: str | None = None
    bindings: tuple[dict[str, object], ...] = ()
    setup_registration: dict[str, object] | None = None
    project_settings: dict[str, str] | None = None

    @classmethod
    def read(cls, path: Path) -> Journal:
        value = read_document(path)
        if value.get("format") != "dinkster.journal/1":
            raise EnvironmentError("unsupported journal format")
        operation, stage, error = value.get("operation"), value.get("stage"), value.get("error")
        if not isinstance(operation, str) or operation not in {
            "setup",
            "update",
            "rollback",
            "move",
            "repair",
            "gc",
        }:
            raise EnvironmentError("unsupported journal operation")
        if not isinstance(stage, str) or stage not in {
            "planned",
            "building",
            "starting",
            "switching",
            "complete",
            "rolled-back",
        }:
            raise EnvironmentError("unsupported journal stage")
        if error is not None and not isinstance(error, str):
            raise EnvironmentError("journal error must be a string")
        bindings = value.get("bindings", [])
        if not isinstance(bindings, list):
            raise EnvironmentError("journal bindings must be an array")
        checked: list[dict[str, object]] = []
        names: set[str] = set()
        ports: set[int] = set()
        for entry in cast("list[object]", bindings):
            if not isinstance(entry, dict):
                raise EnvironmentError("journal binding must be an object")
            binding = cast("dict[str, object]", entry)
            name, port, target_port, gpus, running = (
                binding.get("name"),
                binding.get("port"),
                binding.get("targetPort"),
                binding.get("gpus"),
                binding.get("running"),
            )
            if (
                not isinstance(name, str)
                or re.fullmatch(r"[a-z0-9][a-z0-9-]*", name) is None
                or name in names
                or type(port) is not int
                or not 1 <= port <= 65535
                or type(target_port) is not int
                or not 1 <= target_port <= 65535
                or target_port in ports
                or not isinstance(gpus, list)
                or type(running) is not bool
            ):
                raise EnvironmentError("invalid journal binding")
            if any(
                not isinstance(gpu, str)
                or re.fullmatch(r"(?:[0-9]+|GPU-[0-9A-Fa-f-]+)", gpu) is None
                for gpu in cast("list[object]", gpus)
            ):
                raise EnvironmentError("invalid journal GPU assignment")
            names.add(name)
            ports.add(target_port)
            checked.append(binding)
        previous = value.get("previous")
        if previous == value.get("target") and operation in {"update", "rollback", "repair"}:
            raise EnvironmentError("activation journal must name distinct generations")
        settings = value.get("projectSettings")
        if settings is not None:
            if operation not in {"update", "rollback"} or not isinstance(settings, dict):
                raise EnvironmentError("project settings snapshot requires update or rollback")
            settings = cast("dict[str, object]", settings)
            if set(settings) != {"previous", "target"}:
                raise EnvironmentError("project settings snapshot requires previous and target")
            for text in settings.values():
                if not isinstance(text, str):
                    raise EnvironmentError("project settings snapshots must be JSON text")
                try:
                    document: object = json.loads(text)
                except ValueError as exc:
                    raise EnvironmentError("project settings snapshots must be JSON text") from exc
                if (
                    not isinstance(document, dict)
                    or cast("dict[str, object]", document).get("format") != "dinkster.project/1"
                ):
                    raise EnvironmentError("project settings snapshots require project documents")
        registration = value.get("setupRegistration")
        if registration is not None:
            if operation != "setup" or previous is not None or not isinstance(registration, dict):
                raise EnvironmentError("registration snapshot requires initial setup")
            registration = cast("dict[str, object]", registration)
            for key in ("store", "registry"):
                text = registration.get(key)
                if not isinstance(text, str) or not Path(text).is_absolute():
                    raise EnvironmentError("registration snapshot requires absolute paths")
            for key in ("project", "pointer"):
                if key not in registration or (
                    registration[key] is not None and not isinstance(registration[key], str)
                ):
                    raise EnvironmentError("registration snapshot requires prior metadata text")
            for key in ("addedProject", "addedVolume", "hadProjects", "hadRegistry"):
                if type(registration.get(key)) is not bool:
                    raise EnvironmentError("registration snapshot requires reference flags")
        return cls(
            operation,
            None if previous is None else generation_number(previous),
            generation_number(value.get("target")),
            stage,
            error,
            tuple(checked),
            registration,
            cast("dict[str, str] | None", settings),
        )

    @property
    def incomplete(self) -> bool:
        return self.stage not in {"complete", "rolled-back"}

    def write(self, path: Path) -> None:
        value: dict[str, object] = {
            "format": "dinkster.journal/1",
            "operation": self.operation,
            "previous": self.previous,
            "target": self.target,
            "stage": self.stage,
        }
        if self.error is not None:
            value["error"] = self.error
        if self.bindings:
            value["bindings"] = list(self.bindings)
        if self.setup_registration is not None:
            value["setupRegistration"] = self.setup_registration
        if self.project_settings is not None:
            value["projectSettings"] = self.project_settings
        write_document(path, value)
