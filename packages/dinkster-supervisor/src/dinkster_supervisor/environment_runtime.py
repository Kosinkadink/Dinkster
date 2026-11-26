"""Identity-checked local supervisor processes for environment engines."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import ssl
import subprocess
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.request import (
    HTTPDefaultErrorHandler,
    HTTPErrorProcessor,
    HTTPHandler,
    HTTPRedirectHandler,
    HTTPSHandler,
    OpenerDirector,
    Request,
    UnknownHandler,
)

from .environment_listener import PublicListener, read_response_compression, read_runtime_offline
from .environment_process import cancel_pending_process, prepare_process, process_state
from .environment_state import (
    EnvironmentError,
    Generation,
    atomic_write,
    interpreter,
    library_root,
    read_current,
    read_document,
    read_environment_identity,
    writer_lock,
)

__all__ = [
    "engine_assignments",
    "engine_provenance",
    "engine_status",
    "generation_runtime_lock",
    "generation_runtime_reason",
    "preserve_stopped_engine",
    "require_idle",
    "start_engine",
    "stop_engine",
    "validate_assignment",
]

_ENGINE_NAME = re.compile(r"[a-z0-9][a-z0-9-]*\Z")
_GPU_ID = re.compile(r"(?:[0-9]+|GPU-[0-9A-Fa-f-]+)\Z")
_STATES = frozenset({"starting", "ready", "failed", "stopped"})
_STARTUP_TIMEOUT = 600.0
_STOP_TIMEOUT = 30.0
_POLL_INTERVAL = 0.05


@contextmanager
def generation_runtime_lock(root: Path) -> Generator[None]:
    """Serialize interpreter acquisition with collection, including direct runtime callers."""
    runtime = root / "runtime"
    if runtime.is_symlink() or runtime.is_junction():
        raise EnvironmentError("runtime must not be linked")
    runtime.mkdir(exist_ok=True)
    with writer_lock(runtime):
        yield


def run_generation(root: Path, command: list[str]) -> int:
    """Keep the active interpreter protected from collection while a tool uses it."""
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise EnvironmentError("run requires a command after --")
    with writer_lock(root):
        current = read_current(root)
        if current is None:
            raise EnvironmentError("environment has no active snapshot; run develop or setup first")
        target = root / "builds" / str(current)
        Generation.read(target / "build.json")
        python = interpreter(target)
        environment = dict(os.environ)
        for name in ("UV_PROJECT", "UV_PROJECT_ENVIRONMENT", "PYTHONHOME", "PYTHONPATH"):
            environment.pop(name, None)
        scripts = subprocess.check_output(
            [str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_path('scripts'))"],
            text=True,
        ).strip()
        environment["VIRTUAL_ENV"] = str(python.parent.parent)
        environment["PATH"] = os.pathsep.join(
            (scripts, str(python.parent), environment.get("PATH", ""))
        )
        environment["DINKSTER_EXECUTION_PYTHON"] = str(python)
        executable = (
            str(python)
            if command[0] == "python"
            else shutil.which(command[0], path=environment["PATH"])
        )
        if executable is None:
            raise EnvironmentError(f"command not found in snapshot or PATH: {command[0]}")
        return subprocess.run([executable, *command[1:]], env=environment).returncode


def sandbox_readiness(root: Path, generation: int) -> dict[str, object]:
    """Ask the selected engine's worker launcher, not the tools host, about isolation."""
    with generation_runtime_lock(root):
        target = root / "builds" / str(generation)
        Generation.read(target / "build.json")
        try:
            completed = subprocess.run(
                [
                    str(interpreter(target)),
                    "-I",
                    "-m",
                    "dinkster_workers.doctor",
                    "--sandbox-readiness",
                    "--json",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            value: object = json.loads(completed.stdout)
            if not isinstance(value, dict):
                raise ValueError("invalid sandbox readiness report")
            report = cast("dict[str, object]", value)
            detail = report.get("detail")
            if (
                report.get("format") != "dinkster.sandbox-readiness/1"
                or type(report.get("available")) is not bool
                or not isinstance(detail, str)
                or not detail.strip()
            ):
                raise ValueError("invalid sandbox readiness report")
            return report
        except (OSError, ValueError, subprocess.SubprocessError):
            return {
                "format": "dinkster.sandbox-readiness/1",
                "available": False,
                "detail": "The selected engine could not report pack sandbox readiness. "
                "Update or repair the engine, then check again.",
            }


def check_library_compatibility(root: Path, generation: int) -> None:
    from dinkster_store.library_compatibility import LibraryCompatibilityError, check_library

    library = library_root(root)
    if not library:
        return
    try:
        target = root / "builds" / str(generation)
        manifest = target / "engine-manifest.json"
        capabilities: object = (
            read_document(manifest).get("capabilities", []) if manifest.is_file() else []
        )
        if not isinstance(capabilities, list) or any(
            not isinstance(value, str) for value in cast("list[object]", capabilities)
        ):
            raise EnvironmentError("engine capabilities must be a list of strings")
        check_library(
            Path(library),
            python=interpreter(target),
            capabilities=tuple(cast("list[str]", capabilities)),
        )
    except (LibraryCompatibilityError, OSError, ValueError, subprocess.SubprocessError) as exc:
        raise EnvironmentError(str(exc)) from exc


def _state_readers(root: Path, generation: int) -> str:
    from packaging.version import Version

    from .environment_history import history
    from .environment_state import digest, generation_number

    selected = Generation.read(root / "builds" / str(generation) / "build.json")
    version = Version(selected.engine_version)
    engines: list[tuple[int, str, Version]] = []
    for entry in history(root):
        snapshot = cast("dict[str, object]", entry["snapshot"])
        engine = snapshot.get("engine")
        if not isinstance(engine, dict):
            raise EnvironmentError("snapshot history must identify its engine")
        engine = cast("dict[str, object]", engine)
        recorded_version = engine.get("version")
        if not isinstance(recorded_version, str):
            raise EnvironmentError("snapshot history must identify its engine version")
        engines.append(
            (
                generation_number(entry.get("generation")),
                digest(engine.get("release")),
                Version(recorded_version),
            )
        )
    # A restore rebuilds old software under a new number. Equal-version releases
    # keep their original order rather than the rebuilt tree's position.
    first_seen = min(
        (number for number, release, _ in engines if release == selected.engine_release),
        default=generation,
    )
    readers = {selected.engine_release}
    readers.update(
        release
        for number, release, recorded_version in engines
        if recorded_version < version or (recorded_version == version and number <= first_seen)
    )
    return json.dumps({"engine": selected.engine_release, "readers": sorted(readers)})


def development_frontend(root: Path, generation: int, name: str) -> tuple[str | None, str | None]:
    """The development frontend folder or dev server an engine start uses when none is given.

    The engine's last start wins, then the snapshot's development selection.
    """
    previous = _load_record(root, name, generation)
    if previous is not None and ("frontendRoot" in previous or "frontendDev" in previous):
        return (
            cast("str | None", previous.get("frontendRoot")),
            cast("str | None", previous.get("frontendDev")),
        )
    development = root / "builds" / str(generation) / "development-sources.json"
    if not development.is_file():
        return None, None
    selection = read_document(development)
    saved_dev = selection.get("frontendDev")
    if saved_dev is None:
        return cast("str | None", selection.get("frontendRoot")), None
    from .environment_development import frontend_development_url

    return cast("str | None", selection.get("frontendRoot")), frontend_development_url(
        str(saved_dev), allow_remote=selection.get("allowFrontendDevRemote") is True
    )


def start_engine(
    root: Path,
    generation: int,
    name: str,
    port: int,
    gpus: tuple[str, ...] = (),
    *,
    activation_journal: Path | None = None,
    frontend_root: str | None = None,
    frontend_dev: str | None = None,
    sandbox_packs: bool = False,
    selected_packs: bool = False,
) -> dict[str, object]:
    """Start one generation through its identity-bound managed supervisor."""
    root = Path(root)
    if sandbox_packs:
        readiness = sandbox_readiness(root, generation)
        if not readiness["available"]:
            raise EnvironmentError(f"pack sandbox unavailable: {readiness['detail']}")
    with generation_runtime_lock(root):
        return _start_engine(
            root,
            generation,
            name,
            port,
            gpus,
            activation_journal,
            frontend_root,
            frontend_dev,
            sandbox_packs,
            selected_packs,
        )


def _start_engine(
    root: Path,
    generation: int,
    name: str,
    port: int,
    gpus: tuple[str, ...],
    activation_journal: Path | None,
    frontend_root: str | None,
    frontend_dev: str | None,
    sandbox_packs: bool,
    selected_packs: bool,
) -> dict[str, object]:
    from .environment_nodes import read_node_exclusions
    from .environment_packs import read_pack_selection
    from .environment_previews import project_preview_storage
    from .environment_scratch import project_scratch_storage

    settings = (
        read_document(root / "environment.json") if (root / "environment.json").is_file() else {}
    )
    compression = read_response_compression(settings.get("responseCompression", False))
    runtime_offline = read_runtime_offline(settings.get("runtimeOffline", False))
    excluded_nodes = read_node_exclusions(settings.get("excludedNodeTypes", []))
    project_scratch_storage(root)
    project_preview_storage(root)
    pack_selection = None
    if selected_packs:
        if "packSelection" not in settings:
            raise EnvironmentError("no saved pack selection; configure it with packs first")
        pack_selection = read_pack_selection(settings["packSelection"])
    validate_assignment(generation, name, port, gpus)
    listener = PublicListener.project(root)
    listener.server_context()
    target = root / "builds" / str(generation)
    Generation.read(target / "build.json")
    python = interpreter(target)
    if not python.is_file():
        raise EnvironmentError(f"snapshot {generation} has no interpreter: {python}")

    previous = _load_record(root, name, generation)
    if frontend_root is None and frontend_dev is None:
        frontend_root, frontend_dev = development_frontend(root, generation, name)
    if frontend_root is not None and frontend_dev is not None:
        raise EnvironmentError("development frontend folder and dev server are exclusive")

    if previous is not None:
        existing = _inspect(previous)
        if existing["state"] == "ready":
            if _assignment(existing) == (generation, port, gpus):
                if runtime_offline != previous.get("runtimeOffline", False):
                    raise EnvironmentError("runtime network policy changed; stop the engine first")
                if compression != previous.get("responseCompression", False):
                    raise EnvironmentError("response compression changed; stop the engine first")
                if pack_selection != previous.get("packSelection"):
                    raise EnvironmentError("pack selection changed; stop the engine first")
                if settings.get("scratchStorage", {}) != previous.get("scratchStorage", {}):
                    raise EnvironmentError("scratch storage changed; stop the engine first")
                if settings.get("previewStorage", {}) != previous.get("previewStorage", {}):
                    raise EnvironmentError("preview storage changed; stop the engine first")
                if excluded_nodes != previous.get("excludedNodeTypes", []):
                    raise EnvironmentError("node exclusions changed; stop the engine first")
                if sandbox_packs != previous.get("sandboxPacks", False):
                    raise EnvironmentError("pack sandbox selection changed; stop the engine first")
                if (
                    frontend_root is not None and frontend_root != previous.get("frontendRoot")
                ) or (frontend_dev is not None and frontend_dev != previous.get("frontendDev")):
                    raise EnvironmentError("frontend selection changed; stop the engine first")
                if existing["pid"] != previous["pid"]:
                    _write_record(root, name, existing)
                return _public_row(existing)
            raise EnvironmentError("engine assignment changed; stop it first")
        if existing["state"] != "stopped":
            raise EnvironmentError("engine record is live but not safely reachable; stop it first")
        # A stopped receipt can survive an interrupted stop or an external SIGTERM.
        stop_engine(root, name, generation=generation)

    check_library_compatibility(root, generation)
    manifest = target / "engine-manifest.json"
    from .environment_tools import library_command

    helper = library_command(root, read_document(manifest) if manifest.is_file() else {})

    identity = read_environment_identity(root)
    if identity is None:
        identity = secrets.token_hex(16)
        atomic_write(root / "environment-id", identity + "\n")

    instance = secrets.token_urlsafe(24)
    token = secrets.token_urlsafe(32)
    lifecycle = _process_directory(_runtime_path(root, name, generation), instance)
    url = listener.url(port)
    library = library_root(root)
    library_path = Path(library).resolve()
    library_path.mkdir(parents=True, exist_ok=True)
    library_stat = library_path.stat()
    record: dict[str, object] = {
        "publicListener": listener.document(),
        "responseCompression": compression,
        "runtimeOffline": runtime_offline,
        "environmentId": identity,
        "library": {
            "path": str(library_path),
            "device": library_stat.st_dev,
            "inode": library_stat.st_ino,
        },
        "name": name,
        "generation": generation,
        "port": port,
        "gpus": list(gpus),
        "instance": instance,
        "shutdownToken": token,
        "pid": 0,
        "url": url,
        "state": "starting",
        "_processState": lifecycle,
        "sandboxPacks": sandbox_packs,
        "packSelection": pack_selection,
        "excludedNodeTypes": excluded_nodes,
        "scratchStorage": settings.get("scratchStorage", {}),
        "previewStorage": settings.get("previewStorage", {}),
        **({"frontendRoot": frontend_root} if frontend_root is not None else {}),
        **({"frontendDev": frontend_dev} if frontend_dev is not None else {}),
    }
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    environment["DINKSTER_LIBRARY_COMMAND"] = json.dumps(helper)
    environment["DINKSTER_ENGINE_IDENTITY"] = json.dumps(
        {
            "environment": identity,
            "snapshot": str(generation),
            "identityVersion": 1,
        }
    )
    manifest = target / "engine-manifest.json"
    capabilities = read_document(manifest).get("capabilities") if manifest.is_file() else None
    bound_endpoint = isinstance(capabilities, list) and "bound-endpoint/1" in capabilities
    from dinkster_store import library_lock

    command = [
        sys.executable,
        "-m",
        "dinkster_supervisor.supervisor",
        f"--instance={instance}",
        f"--shutdown-token={token}",
        "--process-state",
        str(lifecycle),
        "--environment-root",
        str(root),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        *(
            ["--activation-journal", str(activation_journal), "--activation-name", name]
            if activation_journal is not None
            else []
        ),
        *(["--engine-bound-endpoint"] if bound_endpoint else []),
        "--",
        str(python),
        "-I",
        "-B",
        str(library_lock.__file__),
        library,
        f"engine {name} in {root} (snapshot {generation})",
        *(
            [
                "--engine-state",
                str(root.resolve() / "engine-state" / name),
                str(generation),
                _state_readers(root, generation),
                "--",
            ]
            if isinstance(capabilities, list) and "library-helper/1" in capabilities
            else []
        ),
        "--generation",
        str(target),
        "--library-root",
        library,
        "--prepare-stale-catalogs",
        "--allow-mount-changes",
        *(
            ["--node-table-record", str(node_table_record(root, name))]
            if isinstance(capabilities, list) and "node-table-record/1" in capabilities
            else []
        ),
    ]
    if sandbox_packs:
        command.append("--sandbox-packs")
    if selected_packs:
        command.append("--selected-packs")
    if frontend_root is not None:
        command.extend(["--frontend-root", frontend_root])
    if frontend_dev is not None:
        command.extend(["--frontend-dev", frontend_dev])
    prepare_process(lifecycle, instance)
    _write_record(root, name, record)
    log_path = _runtime_path(root, name, generation).with_suffix(".log")
    with log_path.open("ab") as log:
        log_offset = log.tell()
        if os.name != "nt":
            log_path.chmod(0o600)
        try:
            # The managed supervisor must survive its short-lived CLI session, including WSL.
            process = None
            if sys.platform == "darwin":
                from .macos_process_family import launch_supervisor

                launch_supervisor(command, lifecycle, root, environment, log_path)
            else:
                process = subprocess.Popen(
                    command,
                    cwd=root,
                    env=environment,
                    stdout=log,
                    stderr=log,
                    start_new_session=os.name != "nt",
                )
        except (OSError, subprocess.SubprocessError):
            cancel_pending_process(lifecycle, instance)
            record["state"] = "stopped"
            _write_record(root, name, record)
            raise
    deadline = time.monotonic() + _STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        inspected = _inspect(record)
        # A Python launcher may forward to a supervisor with a different PID.
        if _identity_matches(inspected) and (
            inspected["pid"] != record["pid"] or inspected["port"] != record["port"]
        ):
            record = inspected
            _write_record(root, name, record)
        if inspected["state"] == "ready" and _health_is_ready(inspected):
            return _public_row(inspected)
        if sys.platform == "darwin":
            from .macos_process_family import supervisor_exit_code

            exit_code = supervisor_exit_code(lifecycle)
        else:
            exit_code = process.poll()
        if inspected["state"] == "stopped" and exit_code is None:
            # The cleanup receipt precedes the launcher's final traceback and stream flush.
            time.sleep(_POLL_INTERVAL)
            continue
        if inspected["state"] == "stopped" or (
            inspected["state"] == "failed" and _identity_matches(inspected)
        ):
            failure = _startup_failure(
                f"engine {name} failed before becoming ready",
                inspected.get("_engineExitCode")
                if inspected.get("_engineExitCode") is not None
                else exit_code,
                log_path,
                log_offset,
            )
            stop_engine(root, name, generation=generation)
            raise failure
        if exit_code is not None:
            record["state"] = "failed"
            _write_record(root, name, record)
            if sys.platform == "darwin":
                stop_engine(root, name, generation=generation)
            raise _startup_failure(
                f"engine {name} exited before becoming ready (supervisor launcher)",
                exit_code,
                log_path,
                log_offset,
            )
        time.sleep(_POLL_INTERVAL)
    stop_engine(root, name, generation=generation)
    raise _startup_failure(f"engine {name} did not become ready", None, log_path, log_offset)


def _startup_failure(
    message: str, exit_code: object, log_path: Path, log_offset: int
) -> EnvironmentError:
    code = str(exit_code) if type(exit_code) is int else "unavailable"
    try:
        with log_path.open("rb") as log:
            log.seek(0, os.SEEK_END)
            # Bound the error payload and exclude output from previous launches.
            start = max(log_offset, log.tell() - 64 * 1024)
            log.seek(start)
            output = log.read(64 * 1024).decode("utf-8", errors="replace")
        if start > log_offset:
            output = "[earlier output truncated]\n" + output
    except OSError as exc:
        output = f"[cannot read captured output: {exc}]"
    return EnvironmentError(
        f"{message}; exit code {code}\nCaptured output ({log_path}):\n" + (output or "[no output]")
    )


def stop_engine(root: Path, name: str, *, generation: int | None = None) -> None:
    """Stop only the supervisor whose live status proves its recorded identity."""
    _validate_name(name)
    root = Path(root)
    selected = generation if generation is not None else read_current(root)
    if selected is None:
        raise EnvironmentError("environment has no active snapshot")
    record = _load_record(root, name, selected)
    if record is None:
        return
    lifecycle = cast("Path", record["_processState"])
    deadline = time.monotonic() + _STOP_TIMEOUT
    while True:
        claimed = lifecycle.is_dir() and cancel_pending_process(
            lifecycle, cast("str", record["instance"])
        )
        inspected = _inspect(record)
        if inspected["state"] == "stopped":
            if sys.platform == "darwin":
                from .macos_process_family import remove_supervisor

                remove_supervisor(lifecycle)
            return
        if _identity_matches(inspected):
            break
        if not claimed or time.monotonic() >= deadline:
            raise EnvironmentError(
                "engine identity is unreachable or mismatched; refusing to stop it"
            )
        time.sleep(_POLL_INTERVAL)
    headers = {
        "X-Dinkster-Instance": cast("str", inspected["instance"]),
        "X-Dinkster-Shutdown-Token": cast("str", inspected["shutdownToken"]),
    }
    try:
        shutdown_url = cast("str", inspected["url"]) + "/supervisor/shutdown"
        _request(
            shutdown_url, method="POST", headers=headers, listener=inspected.get("publicListener")
        )
    except (HTTPError, URLError, ValueError):
        # GracefulExit may close the listener before an HTTP response reaches us.
        pass
    deadline = time.monotonic() + _STOP_TIMEOUT
    while time.monotonic() < deadline:
        if not _process_alive(inspected):
            if (
                lifecycle.is_dir()
                and process_state(
                    lifecycle, cast("str", inspected["instance"]), pid=cast("int", inspected["pid"])
                )
                != "stopped"
            ):
                raise EnvironmentError("supervisor exited without proving child cleanup")
            if sys.platform == "darwin":
                from .macos_process_family import remove_supervisor

                remove_supervisor(lifecycle)
            inspected["state"] = "stopped"
            inspected["pid"] = 0
            _write_record(root, name, inspected)
            from .environment_collection import automatic_generation_collection

            automatic_generation_collection(root)
            return
        time.sleep(_POLL_INTERVAL)
    raise EnvironmentError("authenticated supervisor did not stop")


def engine_status(root: Path) -> tuple[dict[str, object], ...]:
    """Return the selected generation's engines without trusting a reused PID."""
    runtime = Path(root) / "runtime"
    if not runtime.is_dir():
        return ()
    rows: list[dict[str, object]] = []
    selected = read_current(Path(root))
    paths = (
        runtime.glob("*/*.json") if selected is None else (runtime / str(selected)).glob("*.json")
    )
    for path in sorted(paths):
        rows.append(_public_row(_inspect(_parse_record(path))))
    return tuple(rows)


def engine_provenance(root: Path, name: str) -> dict[str, object] | None:
    """The ready local runtime's verified identity, never an HTTP claim alone."""
    if _ENGINE_NAME.fullmatch(name) is None:
        return None
    generation = read_current(root)
    if generation is None:
        return None
    record = _load_record(root, name, generation)
    identity = read_environment_identity(root)
    if record is None or identity is None or record.get("environmentId") != identity:
        return None
    inspected = _inspect(record)
    library = Path(library_root(root))
    if (
        inspected["state"] != "ready"
        or not _identity_matches(inspected)
        or not isinstance(inspected["_engineInstance"], str)
        or not inspected["_engineInstance"]
        or not library.is_absolute()
    ):
        return None
    library = library.resolve()
    identity_stat = library.stat()
    filesystem = {
        "path": str(library),
        "device": identity_stat.st_dev,
        "inode": identity_stat.st_ino,
    }
    if record.get("library") != filesystem:
        return None
    return {
        "installation": {"id": identity, "root": str(root.resolve())},
        "engine": {
            "name": name,
            "generation": generation,
            "instance": inspected["instance"],
            "url": inspected["url"],
        },
        "library": filesystem,
        "processInstance": inspected["_engineInstance"],
    }


def engine_assignments(root: Path, generation: int) -> list[dict[str, object]]:
    """Snapshot the assignments that a generation transaction must preserve."""
    rows: list[dict[str, object]] = []
    for path in sorted((root / "runtime" / str(generation)).glob("*.json")):
        record = _inspect(_parse_record(path))
        if record["state"] not in {"ready", "stopped"}:
            raise EnvironmentError("cannot preserve an ambiguous engine assignment")
        rows.append(
            {key: record[key] for key in ("name", "port", "gpus")}
            | {
                "running": record["state"] == "ready",
                "sandboxPacks": record.get("sandboxPacks", False),
            }
        )
    return rows


def generation_runtime_reason(
    root: Path, generation: int, *, restart_proofs: list[Path] | None = None
) -> str | None:
    """A supervisor may restart its child even while its HTTP engine status is stopped."""
    directory = root / "runtime" / str(generation)
    for path in (directory.parent, directory):
        if path.is_symlink() or path.is_junction():
            raise EnvironmentError("collection refuses linked runtime records")
    for path in sorted(directory.glob("*.json")):
        if path.is_symlink() or path.is_junction():
            raise EnvironmentError("collection refuses linked runtime records")
        record = _parse_record(path)
        lifecycle = cast("Path", record["_processState"])
        if not lifecycle.is_dir():
            return "managed engine has no child-cleanup receipt"
    for processes in sorted(directory.glob("*.process")):
        if processes.is_symlink() or processes.is_junction() or not processes.is_dir():
            raise EnvironmentError("collection refuses linked lifecycle records")
        for lifecycle in sorted(processes.iterdir()):
            if lifecycle.is_symlink() or lifecycle.is_junction() or not lifecycle.is_dir():
                raise EnvironmentError("collection refuses linked lifecycle records")
            state_path = lifecycle / "state.json"
            if state_path.is_symlink():
                raise EnvironmentError("collection refuses linked lifecycle state")
            instance = read_document(state_path).get("instance")
            if (
                not isinstance(instance, str)
                or not instance
                or (lifecycle.name != hashlib.sha256(instance.encode()).hexdigest())
            ):
                raise EnvironmentError("lifecycle directory and instance disagree")
            state = process_state(lifecycle, instance)
            if state not in {"stopped", "cancelled"}:
                return "managed supervisor is live or its child cleanup is unproven"
            if (
                state == "stopped"
                and read_document(state_path).get("descendantsStopped") is not True
            ):
                if sys.platform == "darwin":
                    from .macos_process_family import RESTART_REQUIRED, supersede_stopped_receipt

                    with writer_lock(lifecycle):
                        if supersede_stopped_receipt(
                            lifecycle, allow_unpublished=restart_proofs is not None
                        ):
                            if restart_proofs is not None:
                                restart_proofs.append(lifecycle)
                            continue
                    if restart_proofs is not None:
                        raise EnvironmentError(RESTART_REQUIRED)
                return "managed worker descendant cleanup is unproven"
    if any(binding["running"] for binding in engine_assignments(root, generation)):
        return "live managed engine"
    return None


def preserve_stopped_engine(root: Path, name: str, previous: int, target: int, port: int) -> None:
    """Carry a stopped binding forward without launching it during activation."""
    with generation_runtime_lock(root):
        record = _load_record(root, name, previous)
        existing = _load_record(root, name, target)
        if record is None or _inspect(record)["state"] != "stopped":
            raise EnvironmentError("previous binding is not safely stopped")
        if existing is not None and _inspect(existing)["state"] != "stopped":
            raise EnvironmentError("target binding is not safely stopped")
        instance = secrets.token_urlsafe(24)
        lifecycle = _process_directory(_runtime_path(root, name, target), instance)
        prepare_process(lifecycle, instance)
        # This target binding never launched; cancellation must not impersonate child shutdown.
        cancel_pending_process(lifecycle, instance)
        _write_record(
            root,
            name,
            record
            | {
                "generation": target,
                "port": port,
                "url": PublicListener.read(record.get("publicListener", {})).url(port),
                "instance": instance,
                "shutdownToken": secrets.token_urlsafe(32),
                "pid": 0,
                "state": "stopped",
            },
        )


def require_idle(root: Path) -> None:
    """Fail closed unless every live ready engine reports an empty queue."""
    runtime = Path(root) / "runtime"
    if not runtime.is_dir():
        return
    for path in sorted(runtime.glob("*/*.json")):
        record = _parse_record(path)
        inspected = _inspect(record)
        if inspected["state"] == "stopped":
            continue
        if not _identity_matches(inspected) or inspected["state"] != "ready":
            raise EnvironmentError("cannot verify an engine is idle")
        try:
            queue = _request(
                cast("str", inspected["url"]) + "/api/queue/idle",
                listener=inspected.get("publicListener"),
            )
        except (HTTPError, URLError, ValueError) as exc:
            raise EnvironmentError("cannot verify an engine is idle") from exc
        if not isinstance(queue, dict):
            raise EnvironmentError("cannot verify an engine is idle")
        document = cast("dict[str, object]", queue)
        idle = document.get("idle")
        if type(idle) is not bool:
            raise EnvironmentError("cannot verify an engine is idle")
        if not idle:
            raise EnvironmentError("Finish or cancel the running job before updating.")


def validate_assignment(generation: int, name: str, port: int, gpus: tuple[str, ...]) -> None:
    if type(generation) is not int or generation < 1:
        raise EnvironmentError("snapshot must be a positive integer")
    _validate_name(name)
    if type(port) is not int or not 1 <= port <= 65535:
        raise EnvironmentError("port must be in 1..65535")
    if any(_GPU_ID.fullmatch(gpu) is None for gpu in gpus):
        raise EnvironmentError("GPU ids must be decimal indexes or GPU UUIDs")


def _validate_name(name: str) -> None:
    if type(name) is not str or _ENGINE_NAME.fullmatch(name) is None:
        raise EnvironmentError("engine name must be a portable engine id")


def engine_log_tail(root: Path, name: str) -> dict[str, str | bool]:
    """Read at most 64 KiB of the current snapshot's engine output, even after stop."""
    _validate_name(name)
    generation = read_current(root)
    empty: dict[str, str | bool] = {"available": False, "text": "", "truncated": False}
    if generation is None:
        return empty
    try:
        with _runtime_path(root, name, generation).with_suffix(".log").open("rb") as log:
            log.seek(0, os.SEEK_END)
            start = max(0, log.tell() - 64 * 1024)
            log.seek(start)
            output = log.read(64 * 1024).decode("utf-8", errors="replace")
    except FileNotFoundError:
        return empty
    return {"available": True, "text": output, "truncated": start > 0}


def node_table_record(root: Path, name: str) -> Path:
    """Where an engine records its composed node table for the always-on process."""
    return root / "node-tables" / f"{name}.json"


def _runtime_path(root: Path, name: str, generation: int) -> Path:
    return root / "runtime" / str(generation) / f"{name}.json"


def _process_directory(path: Path, instance: str) -> Path:
    return path.with_suffix(".process") / hashlib.sha256(instance.encode()).hexdigest()


def _load_record(root: Path, name: str, generation: int) -> dict[str, object] | None:
    path = _runtime_path(root, name, generation)
    return None if not path.is_file() else _parse_record(path)


def _parse_record(path: Path) -> dict[str, object]:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EnvironmentError(f"cannot read runtime record {path.name}") from exc
    if not isinstance(raw, dict):
        raise EnvironmentError(f"runtime record {path.name} must be an object")
    record = cast("dict[str, object]", raw)
    name, instance, token, url, state = (
        record.get("name"),
        record.get("instance"),
        record.get("shutdownToken"),
        record.get("url"),
        record.get("state"),
    )
    generation, port, pid, gpus = (
        record.get("generation"),
        record.get("port"),
        record.get("pid"),
        record.get("gpus"),
    )
    if not isinstance(gpus, list):
        raise EnvironmentError(f"runtime record {path.name} is invalid")
    gpu_values = cast("list[object]", gpus)
    if any(not isinstance(gpu, str) for gpu in gpu_values):
        raise EnvironmentError(f"runtime record {path.name} is invalid")
    if type(generation) is not int or not isinstance(name, str) or type(port) is not int:
        raise EnvironmentError(f"runtime record {path.name} is invalid")
    gpu_ids = tuple(cast("str", gpu) for gpu in gpu_values)
    validate_assignment(generation, name, port, gpu_ids)
    if path.stem != name or path.parent.name != str(generation):
        raise EnvironmentError(f"runtime record {path.name} does not match its assignment")
    if (
        not isinstance(instance, str)
        or not instance
        or not isinstance(token, str)
        or not token
        or not isinstance(url, str)
        or url != PublicListener.read(record.get("publicListener", {})).url(port)
        or state not in _STATES
        or type(pid) is not int
        or pid < 0
        or type(record.get("sandboxPacks", False)) is not bool
    ):
        raise EnvironmentError(f"runtime record {path.name} is invalid")
    record["_processState"] = _process_directory(path, instance)
    return record


def _write_record(root: Path, name: str, record: dict[str, object]) -> None:
    path = _runtime_path(root, name, cast("int", record["generation"]))
    persisted = {key: value for key, value in record.items() if not key.startswith("_")}
    atomic_write(path, json.dumps(persisted, sort_keys=True, ensure_ascii=True) + "\n")
    if os.name != "nt":
        path.chmod(0o600)


def _inspect(record: dict[str, object]) -> dict[str, object]:
    inspected = dict(record)
    inspected["_verifiedIdentity"] = False
    inspected["_engineExitCode"] = None
    inspected["_engineInstance"] = None
    lifecycle = cast("Path", record["_processState"])
    if lifecycle.is_dir():
        state = process_state(
            lifecycle, cast("str", record["instance"]), pid=cast("int", record["pid"])
        )
        if state in {"stopped", "cancelled"}:
            inspected.update(state="stopped", pid=0)
            return inspected
    if inspected["state"] == "stopped":
        if _process_alive(inspected):
            inspected["state"] = "failed"
        return inspected
    if inspected["pid"] != 0 and not _process_alive(inspected):
        # A crashed supervisor does not prove that its engine child exited.
        inspected["state"] = "failed"
        return inspected
    listener = lifecycle / "listener.json"
    if listener.is_file():
        binding = read_document(listener)
        port = binding.get("port")
        if binding.get("instance") != inspected["instance"] or type(port) is not int:
            raise EnvironmentError("invalid supervisor listener record")
        validate_assignment(
            cast("int", inspected["generation"]),
            cast("str", inspected["name"]),
            port,
            tuple(cast("list[str]", inspected["gpus"])),
        )
        inspected.update(
            port=port, url=PublicListener.read(record.get("publicListener", {})).url(port)
        )
    try:
        status = _request(
            cast("str", inspected["url"]) + "/supervisor/status",
            listener=inspected.get("publicListener"),
        )
    except (HTTPError, URLError, ValueError):
        inspected["state"] = "failed"
        return inspected
    if not isinstance(status, dict):
        inspected["state"] = "failed"
        return inspected
    status_document = cast("dict[str, object]", status)
    if status_document.get("instance") != inspected["instance"]:
        inspected["state"] = "failed"
        return inspected
    supervisor = status_document.get("supervisor")
    if not isinstance(supervisor, dict):
        inspected["state"] = "failed"
        return inspected
    supervisor_pid = cast("dict[str, object]", supervisor).get("pid")
    if type(supervisor_pid) is not int or supervisor_pid < 1:
        inspected["state"] = "failed"
        return inspected
    recorded_pid = cast("int", inspected["pid"])
    if recorded_pid not in {0, supervisor_pid}:
        inspected["state"] = "failed"
        return inspected
    inspected["pid"] = supervisor_pid
    state = status_document.get("state")
    if state not in _STATES:
        inspected["state"] = "failed"
        return inspected
    inspected["state"] = state
    inspected["_verifiedIdentity"] = True
    engine = status_document.get("engine")
    if isinstance(engine, dict):
        inspected["_engineExitCode"] = cast("dict[str, object]", engine).get("exitCode")
        inspected["_engineInstance"] = cast("dict[str, object]", engine).get("instance")
    return inspected


def _identity_matches(record: dict[str, object]) -> bool:
    return record.get("_verifiedIdentity") is True and _process_alive(record)


def _health_is_ready(record: dict[str, object]) -> bool:
    if not _identity_matches(record):
        return False
    try:
        health = _request(
            cast("str", record["url"]) + "/api/health", listener=record.get("publicListener")
        )
    except (HTTPError, URLError, ValueError):
        return False
    return (
        isinstance(health, dict)
        and cast("dict[str, object]", health).get("ok") is True
        and (not record.get("excludedNodeTypes") or "composition" not in health)
    )


def _process_alive(record: dict[str, object]) -> bool:
    pid = record.get("pid")
    if type(pid) is not int or pid < 1:
        return False
    if os.name == "nt":
        return _windows_process_alive(pid)
    if sys.platform == "linux":
        try:
            stat = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
            if stat.rsplit(")", 1)[1].split()[0] == "Z":
                return False
        except (FileNotFoundError, ProcessLookupError):
            return False
        except OSError as exc:
            raise EnvironmentError("cannot determine supervisor process state") from exc
    else:
        # Signal zero also succeeds for an exited, unreaped child.
        try:
            if os.waitpid(pid, os.WNOHANG)[0] == pid:
                return False
        except ChildProcessError:
            pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _windows_process_alive(pid: int) -> bool:
    import ctypes
    from ctypes import wintypes

    process_query_limited_information = 0x1000
    still_active = 259
    win_dll = getattr(ctypes, "WinDLL")  # noqa: B009 - Linux stubs omit it
    kernel32: Any = win_dll("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        get_last_error = getattr(ctypes, "get_last_error")  # noqa: B009 - Linux stubs omit it
        if get_last_error() == 87:
            return False
        raise EnvironmentError("cannot query supervisor process")
    try:
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            raise EnvironmentError("cannot query supervisor exit code")
        return exit_code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def _assignment(record: dict[str, object]) -> tuple[int, int, tuple[str, ...]]:
    return (
        cast("int", record["generation"]),
        cast("int", record["port"]),
        tuple(cast("list[str]", record["gpus"])),
    )


def _public_row(record: dict[str, object]) -> dict[str, object]:
    return (
        {key: record[key] for key in ("name", "generation", "state", "url", "gpus")}
        | ({"sandboxPacks": True} if record.get("sandboxPacks") is True else {})
        | ({"runtimeOffline": True} if record.get("runtimeOffline") is True else {})
    )


def _request(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    listener: object = None,
) -> object:
    request = Request(url, method=method, headers=headers or {})
    opener = OpenerDirector()
    policy = PublicListener.read(listener or {})
    if policy.certificate is not None:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        # Lifecycle clients trust the operator's exact certificate, not its DNS name.
        # The public URL can use a numeric interface while the certificate names a DNS host.
        context.check_hostname = False
        context.load_verify_locations(cafile=policy.certificate)
        context.verify_flags |= ssl.VERIFY_X509_PARTIAL_CHAIN
        opener.add_handler(HTTPSHandler(context=context))
    for handler in (
        HTTPHandler(),
        HTTPDefaultErrorHandler(),
        HTTPRedirectHandler(),
        HTTPErrorProcessor(),
        UnknownHandler(),
    ):
        opener.add_handler(handler)
    try:
        with opener.open(request, timeout=2) as response:
            return json.loads(response.read().decode("utf-8"))
    except (TimeoutError, ConnectionError) as exc:
        # Disconnects and response-body timeouts can escape urllib without wrapping.
        raise URLError(exc) from exc
