"""Named install roots sharing engine objects but never supervisors or user data."""

from __future__ import annotations

import json
import os
import re
import subprocess
import uuid
from pathlib import Path

from dinkster_registry import InstallError

from .engine_install import EngineInstaller, owned_path
from .installer import Installer


def home() -> Path:
    return Path(os.environ.get("DINKSTER_HOME", Path.home() / ".dinkster")).expanduser().resolve()


def project_root(name: str) -> Path:
    if re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", name) is None:
        raise InstallError("project name must be 1-64 lowercase letters, digits or hyphens")
    return owned_path(home(), f"projects/{name}")


def create(name: str, data_root: Path | None = None) -> dict[str, str]:
    root = project_root(name)
    data = (data_root or home() / "data" / name).expanduser().resolve()
    store = owned_path(home(), "engine-store")
    if any(data.is_relative_to(path) or path.is_relative_to(data) for path in (root.parent, store)):
        raise InstallError("project data root must be separate from install and content roots")
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir()
    record = {"name": name, "root": str(root), "dataRoot": str(data)}
    Installer(root, shared_store=store)
    (root / "project.json").write_text(json.dumps(record, sort_keys=True) + "\n")
    return record


def read(name: str) -> dict[str, str]:
    root = project_root(name)
    path = owned_path(root, "project.json")
    if not path.is_file():
        raise InstallError(f"project {name!r} does not exist; create it first")
    record = json.loads(path.read_text())
    if (
        not isinstance(record, dict)
        or record.get("name") != name
        or record.get("root") != str(root)
        or not isinstance(record.get("dataRoot"), str)
    ):
        raise InstallError("invalid project record")
    data = Path(record["dataRoot"])
    namespaces = (root.parent, owned_path(home(), "engine-store"))
    if not data.is_absolute() or any(
        data.resolve().is_relative_to(path) or path.is_relative_to(data.resolve())
        for path in namespaces
    ):
        raise InstallError("project data root must be absolute and outside its install root")
    return record


def list_projects() -> list[dict[str, str]]:
    directory = owned_path(home(), "projects")
    if not directory.exists():
        return []
    return [read(path.name) for path in sorted(directory.iterdir()) if path.is_dir()]


def supervisor_command(
    root: Path,
    data_root: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 3639,
    instance: str | None = None,
    engine_args: tuple[str, ...] = (),
) -> list[str]:
    root, data_root = root.resolve(), data_root.resolve()
    if instance == "":
        raise InstallError("supervisor instance must not be empty")
    if data_root.is_relative_to(root) or root.is_relative_to(data_root):
        raise InstallError("data root must be separate from the engine install root")
    installer = Installer(root)
    store = installer.shared_store
    if store is not None and (
        data_root.is_relative_to(store.resolve()) or store.resolve().is_relative_to(data_root)
    ):
        raise InstallError("data root must be separate from the engine content store")
    number = installer.current_number()
    if number is None:
        raise InstallError("project has no active generation; install an engine first")
    control, execution = EngineInstaller(installer).interpreters(number)
    return [
        str(control),
        "-I",
        "-m",
        "dinkster_supervisor.supervisor",
        "--host",
        host,
        "--port",
        str(port),
        "--instance",
        instance if instance is not None else uuid.uuid4().hex,
        "--",
        str(control),
        "-I",
        "-m",
        "dinkster.serve",
        "--install-root",
        str(root),
        "--library-root",
        str(data_root),
        "--execution-python",
        str(execution),
        "--prepare-stale-catalogs",
        "--disable-p2p",
        "--allow-mount-changes",
        *engine_args,
    ]


def serve(
    root: Path,
    data_root: Path,
    *,
    host: str,
    port: int,
    instance: str | None = None,
    args: tuple[str, ...] = (),
) -> int:
    return subprocess.call(
        supervisor_command(
            root,
            data_root,
            host=host,
            port=port,
            instance=instance,
            engine_args=args,
        )
    )
